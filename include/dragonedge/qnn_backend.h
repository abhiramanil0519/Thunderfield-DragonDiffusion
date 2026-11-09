#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dragonedge/arena.h"
#include "dragonedge/backend.h"
#include "dragonedge/manifest.h"

namespace dragonedge {

class RpcMemAllocator : public Allocator {
 public:
  RpcMemAllocator();
  ~RpcMemAllocator() override;
  const char* name() const override { return "rpcmem"; }
  void* allocate(uint64_t bytes) override;
  void release(void* p) override;
  int fd(void* p) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct QnnOptions {
  std::string backend_lib = "libQnnHtp.so";
  std::string system_lib = "libQnnSystem.so";
  bool share_spill_fill = true;
  uint64_t max_spill_fill_override = 0;
  bool burst = true;
  int log_level = 2;
};

struct QnnDeviceInfo {
  bool valid = false;
  uint32_t soc_model = 0;
  uint32_t arch = 0;
  uint64_t vtcm_size = 0;
};

struct QnnBinaryInfo {
  std::string graph_name;
  uint64_t spill_fill_bytes = 0;
  uint64_t vtcm_bytes = 0;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  double inspect_ms = 0;
};

struct QnnSelfTest {
  uint64_t elems = 0;
  double alloc_ms = 0;
  double build_ms = 0;
  double finalize_ms = 0;
  double register_ms = 0;
  double first_exec_ms = 0;
  double exec_ms = 0;
  double gb_per_s = 0;
  float max_abs_err = 0;
  bool pass = false;
  uint32_t arena_slots = 0;
  double arena_register_ms = 0;
};

class QnnBackend : public Backend {
 public:
  QnnBackend(Arena& arena, const Manifest& m, QnnOptions opts);
  ~QnnBackend() override;

  std::string name() const override { return "qnn-htp"; }
  std::unique_ptr<LoadedPiece> load(const PieceSpec& spec) override;
  void execute(LoadedPiece& piece) override;
  void unload(std::unique_ptr<LoadedPiece> piece) override;

  const QnnDeviceInfo& device_info() const;
  std::vector<QnnBinaryInfo> inspect(const std::string& context_path) const;
  QnnSelfTest self_test(uint64_t elems, int reps);
  uint64_t max_spill_fill() const;
  uint64_t spill_fill(const std::string& piece) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}
