#pragma once

#include <memory>
#include <string>

#include "dragonedge/manifest.h"

namespace dragonedge {

struct LoadTiming {
  uint64_t file_bytes = 0;
  double read_ms = 0;
  double create_ms = 0;
  double register_ms = 0;
  double total_ms = 0;
};

class LoadedPiece {
 public:
  explicit LoadedPiece(const PieceSpec& s) : spec(&s) {}
  virtual ~LoadedPiece() = default;

  const PieceSpec* spec;
  LoadTiming timing;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  virtual std::unique_ptr<LoadedPiece> load(const PieceSpec& spec) = 0;
  virtual void execute(LoadedPiece& piece) = 0;
  virtual void unload(std::unique_ptr<LoadedPiece> piece) = 0;
};

}
