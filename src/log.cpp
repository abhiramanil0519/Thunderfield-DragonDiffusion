#include "dragonedge/log.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <stdexcept>

#include "dragonedge/clock.h"

namespace dragonedge {

namespace {

std::mutex g_mu;
LogLevel g_level = LogLevel::Info;
FILE* g_file = nullptr;

const char* level_tag(LogLevel l) {
  switch (l) {
    case LogLevel::Debug: return "D";
    case LogLevel::Info: return "I";
    case LogLevel::Warn: return "W";
    case LogLevel::Error: return "E";
  }
  return "?";
}

}

void log_set_level(LogLevel level) { g_level = level; }
LogLevel log_level() { return g_level; }

void log_set_file(const std::string& path) {
  std::lock_guard<std::mutex> l(g_mu);
  if (g_file) std::fclose(g_file);
  g_file = path.empty() ? nullptr : std::fopen(path.c_str(), "a");
  if (!path.empty() && !g_file) throw std::runtime_error("cannot open log file " + path);
}

void log_write(LogLevel level, const char* tag, const char* fmt, ...) {
  if (level < g_level) return;
  char msg[4096];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, args);
  va_end(args);

  const auto now = std::chrono::system_clock::now();
  const std::time_t secs = std::chrono::system_clock::to_time_t(now);
  const int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &secs);
#else
  localtime_r(&secs, &tm);
#endif
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);

  std::lock_guard<std::mutex> l(g_mu);
  std::fprintf(stderr, "%s.%03d +%9.3fs %s [%-6s] %s\n", stamp, ms, now_ms() / 1000.0, level_tag(level), tag, msg);
  if (g_file) {
    std::fprintf(g_file, "%s.%03d +%9.3fs %s [%-6s] %s\n", stamp, ms, now_ms() / 1000.0, level_tag(level), tag, msg);
    std::fflush(g_file);
  }
}

}
