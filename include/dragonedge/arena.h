#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "dragonedge/manifest.h"
#include "dragonedge/memory_plan.h"

namespace dragonedge {

enum class DType { F16, F32, UFixed16, UFixed8 };

const char* dtype_name(DType d);
uint32_t dtype_bytes(DType d);

struct Encoding {
  DType dtype = DType::F16;
  float scale = 1.f;
  int32_t offset = 0;
  bool nhwc = false;

  bool operator==(const Encoding& o) const {
    return dtype == o.dtype && scale == o.scale && offset == o.offset && nhwc == o.nhwc;
  }
  bool operator!=(const Encoding& o) const { return !(*this == o); }
};

uint16_t f32_to_f16(float f);
float f16_to_f32(uint16_t h);
void encode_f32(const float* src, void* dst, size_t n, const Encoding& enc);
void decode_f32(const void* src, float* dst, size_t n, const Encoding& enc);

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual const char* name() const = 0;
  virtual void* allocate(uint64_t bytes) = 0;
  virtual void release(void* p) = 0;
  virtual int fd(void* p) const { (void)p; return -1; }
};

class HostAllocator : public Allocator {
 public:
  const char* name() const override { return "host"; }
  void* allocate(uint64_t bytes) override;
  void release(void* p) override;
};

struct Slot {
  TensorSpec spec;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  uint8_t* data = nullptr;
  Encoding enc;
  std::string enc_owner;
};

class Arena {
 public:
  Arena(const std::vector<TensorSpec>& specs, const MemoryPlan& plan, Allocator& alloc);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  Slot& slot(std::string_view name);
  const Slot& slot(std::string_view name) const;
  const Slot* find(std::string_view name) const;
  const std::vector<Slot>& slots() const { return slots_; }

  uint8_t* base() const { return base_; }
  uint64_t size() const { return plan_.total_bytes; }
  int fd() const { return alloc_.fd(base_); }
  const MemoryPlan& plan() const { return plan_; }

  void bind_encoding(std::string_view name, const Encoding& enc, const std::string& owner);

  void write_f32(std::string_view name, const float* src, size_t n);
  void read_f32(std::string_view name, float* dst, size_t n) const;

 private:
  MemoryPlan plan_;
  Allocator& alloc_;
  uint8_t* base_ = nullptr;
  std::vector<Slot> slots_;
  std::mutex bind_mu_;
};

}
