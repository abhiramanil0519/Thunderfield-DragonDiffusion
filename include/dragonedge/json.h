#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dragonedge::json {

class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  static Value parse(std::string_view text);

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool as_bool() const;
  double as_number() const;
  int64_t as_int() const;
  uint64_t as_uint() const;
  const std::string& as_string() const;
  const std::vector<Value>& as_array() const;
  const std::vector<std::string>& keys() const;
  const std::vector<Value>& values() const;
  const Value* find(std::string_view key) const;
  const Value& at(std::string_view key) const;

 private:
  friend class Parser;
  Type type_ = Type::Null;
  bool bool_ = false;
  double number_ = 0;
  std::string string_;
  std::vector<Value> items_;
  std::vector<std::string> keys_;
};

std::string quote(std::string_view s);

}
