#include "dragonedge/unet_runner.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#include "dragonedge/clock.h"
#include "dragonedge/log.h"
#include "dragonedge/mem_accounting.h"

namespace dragonedge {

static const std::set<std::string> kPersistent = {"ehs", "add_proj", "core_out", kLatent};

ArenaLayout plan_unet_arena(const Manifest& m, uint64_t alignment) {
  ArenaLayout out;
  out.tensors = m.tensors;
  const TensorSpec& sample = m.tensor("sample");
  TensorSpec latent;
  latent.name = kLatent;
  latent.shape = sample.shape;
  latent.elem_bytes = 4;
  latent.role = "host";
  out.tensors.push_back(latent);

  const int host_in = 0;
  const int host_out = static_cast<int>(m.pieces.size()) + 1;
  std::map<std::string, std::pair<int, int>> span;
  auto touch = [&](const std::string& n, int op) {
    auto it = span.find(n);
    if (it == span.end()) span[n] = {op, op};
    else {
      it->second.first = std::min(it->second.first, op);
      it->second.second = std::max(it->second.second, op);
    }
  };
  for (const auto& t : out.tensors)
    if (t.role == "external") touch(t.name, host_in);
  for (size_t i = 0; i < m.pieces.size(); ++i) {
    for (const auto& n : m.pieces[i].inputs) touch(n, static_cast<int>(i) + 1);
    for (const auto& n : m.pieces[i].outputs) touch(n, static_cast<int>(i) + 1);
  }
  touch("eps", host_out);
  for (const auto& t : out.tensors) {
    if (kPersistent.count(t.name)) span[t.name] = {host_in, host_out};
    auto it = span.find(t.name);
    if (it == span.end()) throw std::runtime_error("arena: tensor " + t.name + " is never used");
    out.ranges.push_back({t.name, t.bytes(), it->second.first, it->second.second});
  }
  out.plan = plan_memory(out.ranges, alignment);
  return out;
}

UNetRunner::UNetRunner(const Manifest& m, Backend& backend, int slots, MemAccounting* mem)
    : m_(m), backend_(backend), mem_(mem), streamer_(backend, m.stage(Stage::Core), slots, mem) {}

UNetRunner::~UNetRunner() { unload_shell(); }

void UNetRunner::load_shell() {
  if (shell_down_ && shell_up_) return;
  if (!shell_down_) shell_down_ = backend_.load(*m_.stage(Stage::ShellDown).at(0));
  if (!shell_up_) shell_up_ = backend_.load(*m_.stage(Stage::ShellUp).at(0));
  DE_INFO("shell", "pinned shell_down (%.1f ms) and shell_up (%.1f ms)", shell_down_->timing.total_ms,
          shell_up_->timing.total_ms);
  if (mem_) mem_->mark("shell pinned");
}

void UNetRunner::unload_shell() {
  if (shell_up_) backend_.unload(std::move(shell_up_));
  if (shell_down_) backend_.unload(std::move(shell_down_));
}

const LoadTiming& UNetRunner::shell_down_timing() const { return shell_down_->timing; }
const LoadTiming& UNetRunner::shell_up_timing() const { return shell_up_->timing; }

StepStats UNetRunner::run_full_step(int index) {
  if (!shell_down_ || !shell_up_) throw std::logic_error("runner: load_shell() first");
  StepStats st;
  st.index = index;
  st.begin_ms = now_ms();
  st.core = streamer_.run_pass([&] {
    backend_.execute(*shell_down_);
    st.shell_down_ms = now_ms() - st.begin_ms;
  });
  const double up0 = now_ms();
  st.core_ms = up0 - st.begin_ms - st.shell_down_ms;
  backend_.execute(*shell_up_);
  st.end_ms = now_ms();
  st.shell_up_ms = st.end_ms - up0;
  if (mem_) {
    MemSnapshot s = mem_->mark("step " + std::to_string(index) + " done");
    DE_INFO("step", "step %d: total %.1f ms | shell_down %.1f | core %.1f (stall %.1f) | shell_up %.1f | rss %.1f MB "
            "dmabuf %.1f MB avail %.1f MB",
            index, st.end_ms - st.begin_ms, st.shell_down_ms, st.core_ms, st.core.stall_ms, st.shell_up_ms,
            s.rss / 1048576.0, s.dmabuf_fd_bytes / 1048576.0, s.mem_available / 1048576.0);
  } else {
    DE_INFO("step", "step %d: total %.1f ms | shell_down %.1f | core %.1f (stall %.1f) | shell_up %.1f", index,
            st.end_ms - st.begin_ms, st.shell_down_ms, st.core_ms, st.core.stall_ms, st.shell_up_ms);
  }
  return st;
}

Conditioning zero_conditioning(const Manifest& m) {
  Conditioning c;
  c.ehs.assign(m.tensor("ehs").numel(), 0.f);
  c.pooled.assign(static_cast<size_t>(m.pooled_dim), 0.f);
  const float r = static_cast<float>(m.resolution);
  const float ids[6] = {r, r, 0.f, 0.f, r, r};
  std::copy(ids, ids + 6, c.time_ids);
  return c;
}

DenoiseResult denoise(UNetRunner& runner, Arena& arena, const Manifest& m, const EulerSchedule& sched,
                      const Conditioning& cond, const std::vector<float>& initial_noise) {
  const size_t n = m.tensor("sample").numel();
  if (initial_noise.size() != n) throw std::invalid_argument("denoise: noise has wrong size");
  runner.load_shell();

  arena.write_f32("ehs", cond.ehs.data(), cond.ehs.size());
  std::vector<float> add = sdxl_add_proj(cond.pooled, cond.time_ids, m.add_time_proj);
  arena.write_f32("add_proj", add.data(), add.size());

  float* latent = reinterpret_cast<float*>(arena.slot(kLatent).data);
  for (size_t i = 0; i < n; ++i) latent[i] = initial_noise[i] * sched.init_noise_sigma;

  DenoiseResult res;
  std::vector<float> scratch(n);
  std::vector<float> tproj(static_cast<size_t>(m.time_proj.dim));
  for (size_t i = 0; i < sched.timesteps.size(); ++i) {
    const float sigma = sched.sigmas[i];
    scale_model_input(latent, scratch.data(), n, sigma);
    arena.write_f32("sample", scratch.data(), n);
    timestep_embedding(sched.timesteps[i], m.time_proj, tproj.data());
    arena.write_f32("t_proj", tproj.data(), tproj.size());

    res.steps.push_back(runner.run_full_step(static_cast<int>(i)));

    arena.read_f32("eps", scratch.data(), n);
    euler_step(latent, scratch.data(), n, sigma, sched.sigmas[i + 1]);
  }
  res.latent.assign(latent, latent + n);
  return res;
}

}
