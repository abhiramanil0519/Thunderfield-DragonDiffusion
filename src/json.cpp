#include "dragonedge/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace dragonedge::json {

class Parser {
 public:
  explicit Parser(std::string_view s) : s_(s) {}

  Value document() {
    Value v = value();
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  std::string_view s_;
  size_t pos_ = 0;
  int depth_ = 0;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error("json: " + std::string(what) + " at offset " + std::to_string(pos_));
  }

  void skip_ws() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) ++pos_;
  }

  char peek() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++pos_;
  }

  bool consume_literal(std::string_view lit) {
    if (s_.substr(pos_, lit.size()) == lit) {
      pos_ += lit.size();
      return true;
    }
    return false;
  }

  Value value() {
    if (++depth_ > 256) fail("nesting too deep");
    Value v;
    char c = peek();
    if (c == '{') {
      v = object();
    } else if (c == '[') {
      v = array();
    } else if (c == '"') {
      v.type_ = Value::Type::String;
      v.string_ = string();
    } else if (c == 't' && consume_literal("true")) {
      v.type_ = Value::Type::Bool;
      v.bool_ = true;
    } else if (c == 'f' && consume_literal("false")) {
      v.type_ = Value::Type::Bool;
      v.bool_ = false;
    } else if (c == 'n' && consume_literal("null")) {
      v.type_ = Value::Type::Null;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      v = number();
    } else {
      fail("unexpected character");
    }
    --depth_;
    return v;
  }

  Value object() {
    Value v;
    v.type_ = Value::Type::Object;
    expect('{');
    if (peek() == '}') {
      ++pos_;
      return v;
    }
    for (;;) {
      if (peek() != '"') fail("expected object key");
      v.keys_.push_back(string());
      expect(':');
      v.items_.push_back(value());
      char c = peek();
      ++pos_;
      if (c == '}') return v;
      if (c != ',') fail("expected ',' or '}'");
    }
  }

  Value array() {
    Value v;
    v.type_ = Value::Type::Array;
    expect('[');
    if (peek() == ']') {
      ++pos_;
      return v;
    }
    for (;;) {
      v.items_.push_back(value());
      char c = peek();
      ++pos_;
      if (c == ']') return v;
      if (c != ',') fail("expected ',' or ']'");
    }
  }

  Value number() {
    size_t start = pos_;
    if (s_[pos_] == '-') ++pos_;
    while (pos_ < s_.size() && std::string_view("0123456789+-.eE").find(s_[pos_]) != std::string_view::npos) ++pos_;
    std::string tmp(s_.substr(start, pos_ - start));
    char* end = nullptr;
    double d = std::strtod(tmp.c_str(), &end);
    if (end != tmp.c_str() + tmp.size()) fail("bad number");
    Value v;
    v.type_ = Value::Type::Number;
    v.number_ = d;
    return v;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  uint32_t hex4() {
    if (pos_ + 4 > s_.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    for (;;) {
      if (pos_ >= s_.size()) fail("unterminated string");
      char c = s_[pos_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos_ >= s_.size()) fail("unterminated string");
      char e = s_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && consume_literal("\\u")) {
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
};

Value Value::parse(std::string_view text) { return Parser(text).document(); }

static void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("json: value is not ") + what);
}

bool Value::as_bool() const {
  require(type_ == Type::Bool, "a bool");
  return bool_;
}

double Value::as_number() const {
  require(type_ == Type::Number, "a number");
  return number_;
}

int64_t Value::as_int() const {
  double d = as_number();
  require(std::floor(d) == d, "an integer");
  return static_cast<int64_t>(d);
}

uint64_t Value::as_uint() const {
  int64_t i = as_int();
  require(i >= 0, "non-negative");
  return static_cast<uint64_t>(i);
}

const std::string& Value::as_string() const {
  require(type_ == Type::String, "a string");
  return string_;
}

const std::vector<Value>& Value::as_array() const {
  require(type_ == Type::Array, "an array");
  return items_;
}

const std::vector<std::string>& Value::keys() const {
  require(type_ == Type::Object, "an object");
  return keys_;
}

const std::vector<Value>& Value::values() const {
  require(type_ == Type::Object, "an object");
  return items_;
}

const Value* Value::find(std::string_view key) const {
  if (type_ != Type::Object) return nullptr;
  for (size_t i = 0; i < keys_.size(); ++i)
    if (keys_[i] == key) return &items_[i];
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  if (!v) throw std::runtime_error("json: missing key '" + std::string(key) + "'");
  return *v;
}

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += '"';
  return out;
}

}
