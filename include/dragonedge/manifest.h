#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dragonedge {

enum class Stage { ShellDown, Core, ShellUp };

const char* stage_name(Stage s);

struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  uint32_t elem_bytes = 2;
  std::string role;

  uint64_t numel() const;
  uint64_t bytes() const { return numel() * elem_bytes; }
};

struct PieceSpec {
  std::string name;
  Stage stage = Stage::Core;
  std::string context_path;
  uint64_t params = 0;
  uint64_t est_weight_bytes = 0;
  std::vector<std::string> units;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct TimeProjSpec {
  int dim = 0;
  bool flip_sin_to_cos = true;
  float freq_shift = 0.f;
};

struct Manifest {
  std::string path;
  int resolution = 0;
  int latent = 0;
  int seq_len = 0;
  int pooled_dim = 0;
  TimeProjSpec time_proj;
  TimeProjSpec add_time_proj;
  std::vector<TensorSpec> tensors;
  std::vector<PieceSpec> pieces;

  static Manifest load(const std::string& path, const std::string& context_dir = "");

  const TensorSpec& tensor(std::string_view name) const;
  const TensorSpec* find_tensor(std::string_view name) const;
  const PieceSpec& piece(std::string_view name) const;
  std::vector<const PieceSpec*> stage(Stage s) const;
};

}
