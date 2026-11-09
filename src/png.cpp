#include "dragonedge/png.h"

#include <algorithm>
#include <vector>

#include "dragonedge/file_io.h"

namespace dragonedge {

namespace {

uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t v = i;
      for (int k = 0; k < 8; ++k) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
      table[i] = v;
    }
    init = true;
  }
  for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c;
}

void put32(std::vector<uint8_t>& out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v >> 24));
  out.push_back(static_cast<uint8_t>(v >> 16));
  out.push_back(static_cast<uint8_t>(v >> 8));
  out.push_back(static_cast<uint8_t>(v));
}

void chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
  put32(out, static_cast<uint32_t>(data.size()));
  const size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  put32(out, crc32(out.data() + start, out.size() - start) ^ 0xFFFFFFFFu);
}

}

void write_png_rgb8(const std::string& path, int width, int height, const uint8_t* rgb) {
  const size_t row = static_cast<size_t>(width) * 3 + 1;
  std::vector<uint8_t> raw(row * height);
  for (int y = 0; y < height; ++y) {
    raw[y * row] = 0;
    std::copy(rgb + static_cast<size_t>(y) * width * 3, rgb + static_cast<size_t>(y + 1) * width * 3, raw.begin() + y * row + 1);
  }

  std::vector<uint8_t> z = {0x78, 0x01};
  uint32_t a = 1, b = 0;
  for (uint8_t v : raw) {
    a = (a + v) % 65521;
    b = (b + a) % 65521;
  }
  for (size_t off = 0; off < raw.size(); off += 65535) {
    const size_t n = std::min<size_t>(65535, raw.size() - off);
    z.push_back(off + n == raw.size() ? 1 : 0);
    z.push_back(static_cast<uint8_t>(n));
    z.push_back(static_cast<uint8_t>(n >> 8));
    z.push_back(static_cast<uint8_t>(~n));
    z.push_back(static_cast<uint8_t>(~n >> 8));
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
  }
  put32(z, (b << 16) | a);

  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  std::vector<uint8_t> ihdr;
  put32(ihdr, static_cast<uint32_t>(width));
  put32(ihdr, static_cast<uint32_t>(height));
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
  chunk(out, "IHDR", ihdr);
  chunk(out, "IDAT", z);
  chunk(out, "IEND", {});
  write_file(path, out.data(), out.size());
}

}
