#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dragonedge {

std::vector<uint8_t> read_file(const std::string& path);
void write_file(const std::string& path, const void* data, size_t bytes);
bool file_exists(const std::string& path);
uint64_t file_size(const std::string& path);

void evict_page_cache(const std::string& path);

void prefetch_page_cache(const std::string& path);

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  uint8_t* data() const { return data_; }
  uint64_t size() const { return size_; }
  uint64_t touch() const;

 private:
  uint8_t* data_ = nullptr;
  uint64_t size_ = 0;
  bool mapped_ = false;
  std::vector<uint8_t> copy_;
};

}
