#include "dragonedge/sampler.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace dragonedge {

EulerSchedule euler_trailing(int steps, int train_steps, double beta_start, double beta_end) {
  if (steps < 1 || steps > train_steps) throw std::invalid_argument("euler_trailing: bad step count");
  std::vector<double> alphas_cumprod(train_steps);
  const double a = std::sqrt(beta_start);
  const double b = std::sqrt(beta_end);
  double prod = 1.0;
  for (int i = 0; i < train_steps; ++i) {
    double s = a + (b - a) * i / (train_steps - 1);
    prod *= 1.0 - s * s;
    alphas_cumprod[i] = prod;
  }
  EulerSchedule out;
  const double ratio = static_cast<double>(train_steps) / steps;
  for (int i = 0; i < steps; ++i) {
    double t = std::round(train_steps - i * ratio) - 1.0;
    int ti = static_cast<int>(t);
    out.timesteps.push_back(static_cast<float>(t));
    out.sigmas.push_back(static_cast<float>(std::sqrt((1.0 - alphas_cumprod[ti]) / alphas_cumprod[ti])));
  }
  out.sigmas.push_back(0.f);
  out.init_noise_sigma = out.sigmas.front();
  return out;
}

void timestep_embedding(float t, const TimeProjSpec& spec, float* out) {
  const int half = spec.dim / 2;
  const float denom = static_cast<float>(half) - spec.freq_shift;
  const float log_max = std::log(10000.0f);
  for (int i = 0; i < half; ++i) {
    float freq = std::exp(-log_max * static_cast<float>(i) / denom);
    float arg = t * freq;
    float s = std::sin(arg);
    float c = std::cos(arg);
    if (spec.flip_sin_to_cos) {
      out[i] = c;
      out[half + i] = s;
    } else {
      out[i] = s;
      out[half + i] = c;
    }
  }
  if (spec.dim % 2) out[spec.dim - 1] = 0.f;
}

std::vector<float> sdxl_add_proj(const std::vector<float>& pooled, const float time_ids[6], const TimeProjSpec& spec) {
  std::vector<float> out(pooled);
  out.resize(pooled.size() + 6 * static_cast<size_t>(spec.dim));
  for (int k = 0; k < 6; ++k) timestep_embedding(time_ids[k], spec, out.data() + pooled.size() + k * spec.dim);
  return out;
}

void scale_model_input(const float* latent, float* out, size_t n, float sigma) {
  const float s = 1.f / std::sqrt(sigma * sigma + 1.f);
  for (size_t i = 0; i < n; ++i) out[i] = latent[i] * s;
}

void euler_step(float* latent, const float* eps, size_t n, float sigma, float sigma_next) {
  const float dt = sigma_next - sigma;
  for (size_t i = 0; i < n; ++i) latent[i] += eps[i] * dt;
}

void fill_normal(float* out, size_t n, uint64_t seed, float stddev) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.f, stddev);
  for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
}

}
