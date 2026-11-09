#include "dragonedge/arena.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace dragonedge {

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F16: return "f16";
    case DType::F32: return "f32";
    case DType::UFixed16: return "ufixed16";
    case DType::UFixed8: return "ufixed8";
  }
  return "?";
}

uint32_t dtype_bytes(DType d) {
  switch (d) {
    case DType::F16: return 2;
    case DType::F32: return 4;
    case DType::UFixed16: return 2;
    case DType::UFixed8: return 1;
  }
  return 0;
}

uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t absx = x & 0x7FFFFFFFu;
  if (absx >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0u));
  if (absx >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);
  if (absx < 0x38800000u) {
    float af;
    std::memcpy(&af, &absx, 4);
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(af * 16777216.0f)));
  }
  uint32_t h = ((absx - 0x38000000u) >> 13);
  const uint32_t rem = absx & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(sign | h);
}

float f16_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 0x1Fu;
  uint32_t m = h & 0x3FFu;
  uint32_t x;
  if (e == 0) {
    if (m == 0) {
      x = sign;
    } else {
      int ee = -1;
      do {
        ++ee;
        m <<= 1;
      } while ((m & 0x400u) == 0);
      x = sign | static_cast<uint32_t>(127 - 15 - ee) << 23 | (m & 0x3FFu) << 13;
    }
  } else if (e == 31) {
    x = sign | 0x7F800000u | m << 13;
  } else {
    x = sign | (e + 112) << 23 | m << 13;
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

void encode_f32(const float* src, void* dst, size_t n, const Encoding& enc) {
  switch (enc.dtype) {
    case DType::F32:
      std::memcpy(dst, src, n * 4);
      return;
    case DType::F16: {
      auto* d = static_cast<uint16_t*>(dst);
      for (size_t i = 0; i < n; ++i) d[i] = f32_to_f16(src[i]);
      return;
    }
    case DType::UFixed16:
    case DType::UFixed8: {
      const float qmax = enc.dtype == DType::UFixed16 ? 65535.f : 255.f;
      const float inv = 1.f / enc.scale;
      for (size_t i = 0; i < n; ++i) {
        float q = std::nearbyint(src[i] * inv) - static_cast<float>(enc.offset);
        q = std::min(std::max(q, 0.f), qmax);
        if (enc.dtype == DType::UFixed16) static_cast<uint16_t*>(dst)[i] = static_cast<uint16_t>(q);
        else static_cast<uint8_t*>(dst)[i] = static_cast<uint8_t>(q);
      }
      return;
    }
  }
}

void decode_f32(const void* src, float* dst, size_t n, const Encoding& enc) {
  switch (enc.dtype) {
    case DType::F32:
      std::memcpy(dst, src, n * 4);
      return;
    case DType::F16: {
      auto* s = static_cast<const uint16_t*>(src);
      for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(s[i]);
      return;
    }
    case DType::UFixed16:
      for (size_t i = 0; i < n; ++i)
        dst[i] = (static_cast<float>(static_cast<const uint16_t*>(src)[i]) + enc.offset) * enc.scale;
      return;
    case DType::UFixed8:
      for (size_t i = 0; i < n; ++i)
        dst[i] = (static_cast<float>(static_cast<const uint8_t*>(src)[i]) + enc.offset) * enc.scale;
      return;
  }
}

void* HostAllocator::allocate(uint64_t bytes) {
  const uint64_t a = 4096;
  const uint64_t n = (bytes + a - 1) / a * a;
#if defined(_WIN32)
  void* p = _aligned_malloc(n, a);
#else
  void* p = std::aligned_alloc(a, n);
#endif
  if (!p) throw std::bad_alloc();
  return p;
}

void HostAllocator::release(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

Arena::Arena(const std::vector<TensorSpec>& specs, const MemoryPlan& plan, Allocator& alloc)
    : plan_(plan), alloc_(alloc) {
  check_plan(plan_);
  base_ = static_cast<uint8_t*>(alloc_.allocate(plan_.total_bytes));
  std::memset(base_, 0, plan_.total_bytes);
  for (const auto& s : specs) {
    const Placement* p = plan_.find(s.name);
    if (!p) throw std::runtime_error("arena: no placement for " + s.name);
    if (p->bytes < s.bytes()) throw std::runtime_error("arena: placement too small for " + s.name);
    Slot slot;
    slot.spec = s;
    slot.offset = p->offset;
    slot.bytes = s.bytes();
    slot.data = base_ + p->offset;
    slot.enc.dtype = s.elem_bytes == 4 ? DType::F32 : DType::F16;
    slots_.push_back(std::move(slot));
  }
}

Arena::~Arena() { alloc_.release(base_); }

const Slot* Arena::find(std::string_view name) const {
  for (const auto& s : slots_)
    if (s.spec.name == name) return &s;
  return nullptr;
}

const Slot& Arena::slot(std::string_view name) const {
  const Slot* s = find(name);
  if (!s) throw std::runtime_error("arena: unknown tensor '" + std::string(name) + "'");
  return *s;
}

Slot& Arena::slot(std::string_view name) { return const_cast<Slot&>(static_cast<const Arena*>(this)->slot(name)); }

void Arena::bind_encoding(std::string_view name, const Encoding& enc, const std::string& owner) {
  std::lock_guard<std::mutex> lock(bind_mu_);
  Slot& s = slot(name);
  if (dtype_bytes(enc.dtype) * s.spec.numel() != s.bytes)
    throw std::runtime_error("arena: " + owner + " binds " + s.spec.name + " as " + dtype_name(enc.dtype) +
                             " but the slot holds " + std::to_string(s.bytes) + " bytes");
  if (s.enc_owner.empty()) {
    s.enc = enc;
    s.enc_owner = owner;
    return;
  }
  if (s.enc != enc)
    throw std::runtime_error("arena: encoding mismatch on " + s.spec.name + ": " + s.enc_owner + " uses " +
                             dtype_name(s.enc.dtype) + " scale=" + std::to_string(s.enc.scale) +
                             " offset=" + std::to_string(s.enc.offset) + (s.enc.nhwc ? " nhwc" : "") + ", " + owner +
                             " uses " + dtype_name(enc.dtype) + " scale=" + std::to_string(enc.scale) +
                             " offset=" + std::to_string(enc.offset) + (enc.nhwc ? " nhwc" : ""));
}

static void nchw_to_nhwc(const float* src, float* dst, const std::vector<int64_t>& shape) {
  const int64_t N = shape[0], C = shape[1], HW = shape[2] * shape[3];
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t i = 0; i < HW; ++i) dst[(n * HW + i) * C + c] = src[(n * C + c) * HW + i];
}

static void nhwc_to_nchw(const float* src, float* dst, const std::vector<int64_t>& shape) {
  const int64_t N = shape[0], C = shape[1], HW = shape[2] * shape[3];
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t i = 0; i < HW; ++i) dst[(n * C + c) * HW + i] = src[(n * HW + i) * C + c];
}

void Arena::write_f32(std::string_view name, const float* src, size_t n) {
  Slot& s = slot(name);
  if (n != s.spec.numel()) throw std::runtime_error("arena: size mismatch writing " + s.spec.name);
  if (s.enc.nhwc && s.spec.shape.size() == 4) {
    std::vector<float> tmp(n);
    nchw_to_nhwc(src, tmp.data(), s.spec.shape);
    encode_f32(tmp.data(), s.data, n, s.enc);
    return;
  }
  encode_f32(src, s.data, n, s.enc);
}

void Arena::read_f32(std::string_view name, float* dst, size_t n) const {
  const Slot& s = slot(name);
  if (n != s.spec.numel()) throw std::runtime_error("arena: size mismatch reading " + s.spec.name);
  if (s.enc.nhwc && s.spec.shape.size() == 4) {
    std::vector<float> tmp(n);
    decode_f32(s.data, tmp.data(), n, s.enc);
    nhwc_to_nchw(tmp.data(), dst, s.spec.shape);
    return;
  }
  decode_f32(s.data, dst, n, s.enc);
}

}
