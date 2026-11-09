#include "dragonedge/file_io.h"

#include <cstdio>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#define DRAGON_POSIX 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dragonedge {

std::vector<uint8_t> read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> out;
  uint8_t buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  bool err = std::ferror(f) != 0;
  std::fclose(f);
  if (err) throw std::runtime_error("read error on " + path);
  return out;
}

void write_file(const std::string& path, const void* data, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot create " + path);
  size_t n = std::fwrite(data, 1, bytes, f);
  bool ok = n == bytes && std::fclose(f) == 0;
  if (!ok) throw std::runtime_error("write error on " + path);
}

bool file_exists(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

uint64_t file_size(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
#if defined(_WIN32)
  _fseeki64(f, 0, SEEK_END);
  int64_t n = _ftelli64(f);
#else
  fseeko(f, 0, SEEK_END);
  int64_t n = ftello(f);
#endif
  std::fclose(f);
  if (n < 0) throw std::runtime_error("cannot size " + path);
  return static_cast<uint64_t>(n);
}

void evict_page_cache(const std::string& path) {
#if defined(DRAGON_POSIX) && !defined(__APPLE__)
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  fdatasync(fd);
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
#else
  (void)path;
#endif
}

void prefetch_page_cache(const std::string& path) {
#if defined(DRAGON_POSIX) && !defined(__APPLE__)
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
  close(fd);
#else
  (void)path;
#endif
}

MappedFile::MappedFile(const std::string& path) {
#if defined(DRAGON_POSIX)
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = static_cast<uint64_t>(st.st_size);
  void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  madvise(p, size_, MADV_SEQUENTIAL);
  data_ = static_cast<uint8_t*>(p);
  mapped_ = true;
#else
  copy_ = read_file(path);
  data_ = copy_.data();
  size_ = copy_.size();
#endif
}

MappedFile::~MappedFile() {
#if defined(DRAGON_POSIX)
  if (mapped_) munmap(data_, size_);
#endif
}

uint64_t MappedFile::touch() const {
  volatile uint64_t sum = 0;
  for (uint64_t off = 0; off < size_; off += 4096) sum = sum + data_[off];
  return sum;
}

}
