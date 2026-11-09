#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dragonedge {

struct LiveRange {
  std::string name;
  uint64_t bytes = 0;
  int first = 0;
  int last = 0;
};

struct Placement {
  std::string name;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  int first = 0;
  int last = 0;
};

struct MemoryPlan {
  uint64_t alignment = 0;
  uint64_t total_bytes = 0;
  uint64_t unshared_bytes = 0;
  uint64_t peak_live_bytes = 0;
  std::vector<Placement> placements;

  const Placement* find(std::string_view name) const;
};

MemoryPlan plan_memory(const std::vector<LiveRange>& ranges, uint64_t alignment);

void check_plan(const MemoryPlan& plan);

}
