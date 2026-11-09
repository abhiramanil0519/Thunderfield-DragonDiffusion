#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dragonedge/manifest.h"

namespace dragonedge {

struct EulerSchedule {
  std::vector<float> timesteps;
  std::vector<float> sigmas;
  float init_noise_sigma = 1.f;
};

EulerSchedule euler_trailing(int steps, int train_steps = 1000, double beta_start = 0.00085, double beta_end = 0.012);

void timestep_embedding(float t, const TimeProjSpec& spec, float* out);

std::vector<float> sdxl_add_proj(const std::vector<float>& pooled, const float time_ids[6], const TimeProjSpec& spec);

void scale_model_input(const float* latent, float* out, size_t n, float sigma);
void euler_step(float* latent, const float* eps, size_t n, float sigma, float sigma_next);

void fill_normal(float* out, size_t n, uint64_t seed, float stddev);

}
