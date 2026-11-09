#pragma once

#include <cstdint>
#include <string>

namespace dragonedge {

void write_png_rgb8(const std::string& path, int width, int height, const uint8_t* rgb);

}
