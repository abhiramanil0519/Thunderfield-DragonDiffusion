#include "dragonedge/memory_plan.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace dragonedge {

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static bool overlaps_in_time(const Placement& a, const LiveRange& b) {
  return a.first <= b.last && b.first <= a.last;
}

const Placement* MemoryPlan::find(std::string_view name) const {
  for (const auto& p : placements)
    if (p.name == name) return &p;
  return nullptr;
}

MemoryPlan plan_memory(const std::vector<LiveRange>& ranges, uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    throw std::invalid_argument("plan_memory: alignment must be a power of two");

  std::set<std::string> names;
  for (const auto& r : ranges) {
    if (r.first > r.last) throw std::invalid_argument("plan_memory: bad live range for " + r.name);
    if (!names.insert(r.name).second) throw std::invalid_argument("plan_memory: duplicate tensor " + r.name);
  }

  std::vector<size_t> order(ranges.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const auto& x = ranges[a];
    const auto& y = ranges[b];
    if (x.bytes != y.bytes) return x.bytes > y.bytes;
    if (x.first != y.first) return x.first < y.first;
    return x.name < y.name;
  });

  MemoryPlan plan;
  plan.alignment = alignment;
  std::vector<Placement> placed;
  for (size_t idx : order) {
    const LiveRange& r = ranges[idx];
    const uint64_t size = align_up(std::max<uint64_t>(r.bytes, 1), alignment);

    std::vector<const Placement*> live;
    for (const auto& p : placed)
      if (overlaps_in_time(p, r)) live.push_back(&p);
    std::sort(live.begin(), live.end(), [](const Placement* a, const Placement* b) { return a->offset < b->offset; });

    uint64_t best = std::numeric_limits<uint64_t>::max();
    uint64_t best_gap = std::numeric_limits<uint64_t>::max();
    uint64_t cursor = 0;
    for (const Placement* p : live) {
      if (p->offset >= cursor + size) {
        uint64_t gap = p->offset - cursor;
        if (gap < best_gap) {
          best_gap = gap;
          best = cursor;
        }
      }
      cursor = std::max(cursor, align_up(p->offset + p->bytes, alignment));
    }
    if (best == std::numeric_limits<uint64_t>::max()) best = cursor;

    placed.push_back({r.name, best, size, r.first, r.last});
    plan.total_bytes = std::max(plan.total_bytes, best + size);
    plan.unshared_bytes += size;
  }

  std::map<int, uint64_t> delta;
  for (const auto& p : placed) {
    delta[p.first] += p.bytes;
    delta[p.last + 1] -= p.bytes;
  }
  uint64_t live_bytes = 0;
  for (const auto& [t, d] : delta) {
    live_bytes += d;
    plan.peak_live_bytes = std::max(plan.peak_live_bytes, live_bytes);
  }

  std::map<std::string, size_t> pos;
  for (size_t i = 0; i < ranges.size(); ++i) pos[ranges[i].name] = i;
  std::sort(placed.begin(), placed.end(), [&](const Placement& a, const Placement& b) { return pos[a.name] < pos[b.name]; });
  plan.placements = std::move(placed);
  return plan;
}

void check_plan(const MemoryPlan& plan) {
  const auto& ps = plan.placements;
  for (size_t i = 0; i < ps.size(); ++i) {
    if (ps[i].offset % plan.alignment != 0) throw std::logic_error("plan: misaligned " + ps[i].name);
    if (ps[i].offset + ps[i].bytes > plan.total_bytes) throw std::logic_error("plan: out of bounds " + ps[i].name);
    for (size_t j = i + 1; j < ps.size(); ++j) {
      const auto& a = ps[i];
      const auto& b = ps[j];
      bool time = a.first <= b.last && b.first <= a.last;
      bool space = a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
      if (time && space) throw std::logic_error("plan: " + a.name + " and " + b.name + " overlap while both live");
    }
  }
}

}
