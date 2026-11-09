#include "dragonedge/manifest.h"

#include <map>
#include <set>
#include <stdexcept>

#include "dragonedge/file_io.h"
#include "dragonedge/json.h"

namespace dragonedge {

const char* stage_name(Stage s) {
  switch (s) {
    case Stage::ShellDown: return "shell_down";
    case Stage::Core: return "core";
    case Stage::ShellUp: return "shell_up";
  }
  return "?";
}

uint64_t TensorSpec::numel() const {
  uint64_t n = 1;
  for (int64_t d : shape) n *= static_cast<uint64_t>(d);
  return n;
}

static Stage parse_stage(const std::string& s) {
  if (s == "shell_down") return Stage::ShellDown;
  if (s == "core") return Stage::Core;
  if (s == "shell_up") return Stage::ShellUp;
  throw std::runtime_error("manifest: unknown stage '" + s + "'");
}

static std::vector<std::string> strings(const json::Value& v) {
  std::vector<std::string> out;
  for (const auto& x : v.as_array()) out.push_back(x.as_string());
  return out;
}

static TimeProjSpec parse_time_proj(const json::Value& v) {
  TimeProjSpec t;
  t.dim = static_cast<int>(v.at("dim").as_int());
  t.flip_sin_to_cos = v.at("flip_sin_to_cos").as_bool();
  t.freq_shift = static_cast<float>(v.at("freq_shift").as_number());
  return t;
}

static std::string dirname_of(const std::string& path) {
  size_t p = path.find_last_of("/\\");
  return p == std::string::npos ? std::string(".") : path.substr(0, p);
}

Manifest Manifest::load(const std::string& path, const std::string& context_dir) {
  std::vector<uint8_t> bytes = read_file(path);
  json::Value root = json::Value::parse(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  if (root.at("format").as_string() != "dragonedge-manifest")
    throw std::runtime_error("manifest: " + path + " is not a dragonedge manifest");
  if (root.at("version").as_int() != 1) throw std::runtime_error("manifest: unsupported version");

  Manifest m;
  m.path = path;
  m.resolution = static_cast<int>(root.at("resolution").as_int());
  m.latent = static_cast<int>(root.at("latent").as_int());
  const json::Value& cond = root.at("conditioning");
  m.seq_len = static_cast<int>(cond.at("seq_len").as_int());
  m.pooled_dim = static_cast<int>(cond.at("pooled_dim").as_int());
  m.time_proj = parse_time_proj(cond.at("time_proj"));
  m.add_time_proj = parse_time_proj(cond.at("add_time_proj"));

  for (const auto& t : root.at("tensors").as_array()) {
    TensorSpec s;
    s.name = t.at("name").as_string();
    for (const auto& d : t.at("shape").as_array()) s.shape.push_back(d.as_int());
    s.elem_bytes = static_cast<uint32_t>(t.at("elem_bytes").as_uint());
    s.role = t.at("role").as_string();
    if (m.find_tensor(s.name)) throw std::runtime_error("manifest: duplicate tensor " + s.name);
    m.tensors.push_back(std::move(s));
  }

  const std::string dir = context_dir.empty() ? dirname_of(path) : context_dir;
  for (const auto& p : root.at("pieces").as_array()) {
    PieceSpec s;
    s.name = p.at("name").as_string();
    s.stage = parse_stage(p.at("stage").as_string());
    s.context_path = dir + "/" + p.at("context").as_string();
    s.params = p.at("params").as_uint();
    s.est_weight_bytes = p.at("est_w8_bytes").as_uint();
    s.units = strings(p.at("units"));
    s.inputs = strings(p.at("inputs"));
    s.outputs = strings(p.at("outputs"));
    m.pieces.push_back(std::move(s));
  }

  if (m.pieces.size() < 3 || m.pieces.front().stage != Stage::ShellDown || m.pieces.back().stage != Stage::ShellUp)
    throw std::runtime_error("manifest: expected shell_down, core shards, shell_up");
  for (size_t i = 1; i + 1 < m.pieces.size(); ++i)
    if (m.pieces[i].stage != Stage::Core) throw std::runtime_error("manifest: core shards must be contiguous");

  std::map<std::string, std::string> producer;
  std::set<std::string> consumed;
  for (const auto& p : m.pieces) {
    for (const auto& n : p.inputs) {
      const TensorSpec& t = m.tensor(n);
      if (t.role != "external" && !producer.count(n))
        throw std::runtime_error("manifest: " + p.name + " reads " + n + " before it is produced");
      consumed.insert(n);
    }
    for (const auto& n : p.outputs) {
      m.tensor(n);
      if (!producer.emplace(n, p.name).second)
        throw std::runtime_error("manifest: " + n + " produced by both " + producer[n] + " and " + p.name);
    }
  }
  for (const auto& t : m.tensors)
    if (t.role != "output" && !consumed.count(t.name))
      throw std::runtime_error("manifest: tensor " + t.name + " is never consumed");
  return m;
}

const TensorSpec* Manifest::find_tensor(std::string_view name) const {
  for (const auto& t : tensors)
    if (t.name == name) return &t;
  return nullptr;
}

const TensorSpec& Manifest::tensor(std::string_view name) const {
  const TensorSpec* t = find_tensor(name);
  if (!t) throw std::runtime_error("manifest: unknown tensor '" + std::string(name) + "'");
  return *t;
}

const PieceSpec& Manifest::piece(std::string_view name) const {
  for (const auto& p : pieces)
    if (p.name == name) return p;
  throw std::runtime_error("manifest: unknown piece '" + std::string(name) + "'");
}

std::vector<const PieceSpec*> Manifest::stage(Stage s) const {
  std::vector<const PieceSpec*> out;
  for (const auto& p : pieces)
    if (p.stage == s) out.push_back(&p);
  return out;
}

}
