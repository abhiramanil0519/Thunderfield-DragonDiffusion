#pragma once

#include <string>

namespace dragonedge {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void log_set_level(LogLevel level);
LogLevel log_level();
void log_set_file(const std::string& path);
void log_write(LogLevel level, const char* tag, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

}

#define DE_DEBUG(tag, ...) ::dragonedge::log_write(::dragonedge::LogLevel::Debug, tag, __VA_ARGS__)
#define DE_INFO(tag, ...) ::dragonedge::log_write(::dragonedge::LogLevel::Info, tag, __VA_ARGS__)
#define DE_WARN(tag, ...) ::dragonedge::log_write(::dragonedge::LogLevel::Warn, tag, __VA_ARGS__)
#define DE_ERROR(tag, ...) ::dragonedge::log_write(::dragonedge::LogLevel::Error, tag, __VA_ARGS__)
