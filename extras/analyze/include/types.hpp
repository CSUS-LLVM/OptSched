#pragma once

#include <pybind11/pybind11.h>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ev {
using Number = std::variant<std::int64_t, std::uint64_t, double>;

struct EventId {
  std::string_view Value;

  bool operator==(const EventId &) const = default;
};

enum class Type {
  Number,
  String,
  Bool,
};

union Value {
  Number Num;
  std::string_view Str;
  bool Bool;
};

struct EventSchema {
  EventId Id;
  std::vector<std::string_view> Parameters;
  std::vector<Type> ParamTypes;

  bool operator==(const EventSchema &) const = default;
  bool operator==(EventId Rhs) const { return Id == Rhs; }
};

struct Event {
  EventId Id;
  const std::deque<Value> *Values;
  std::size_t Start; // Indices into values.
  std::size_t End;

  bool operator==(EventId Rhs) const { return Id == Rhs; }
};

struct EventIdHash {
  using is_transparent = void;

  std::size_t operator()(std::string_view Id) const noexcept {
    return std::hash<std::string_view>()(Id);
  }

  std::size_t operator()(EventId Id) const noexcept {
    return (*this)(Id.Value);
  }

  std::size_t operator()(const EventSchema &Schema) const noexcept {
    return (*this)(Schema.Id);
  }

  std::size_t operator()(const std::vector<Event> &Events) const noexcept {
    assert(!Events.empty());
    return (*this)(Events.front());
  }

  std::size_t operator()(const Event &Event) const noexcept {
    return (*this)(Event.Id);
  }
};

struct EventIdEq {
  using is_transparent = void;

  template <typename T, typename U>
  bool operator()(const T &Lhs, const U &Rhs) const {
    return Lhs == Rhs;
  }

  template <typename T, typename U>
  bool operator()(const T &Lhs, const std::vector<U> &Rhs) const {
    assert(!Rhs.empty());
    return (*this)(Lhs, Rhs.front());
  }

  template <typename T, typename U>
  bool operator()(const std::vector<T> &Lhs, const U &Rhs) const {
    assert(!Lhs.empty());
    return (*this)(Lhs.front(), Rhs);
  }

  template <typename T, typename U>
  bool operator()(const std::vector<T> &Lhs, const std::vector<U> &Rhs) const {
    assert(!Lhs.empty() && !Rhs.empty());
    return (*this)(Lhs.front(), Rhs.front());
  }

  bool operator()(const Event &Lhs, const Event &Rhs) const {
    return (*this)(Lhs.Id, Rhs.Id);
  }
};

using BlockEventMap =
    std::unordered_set<std::vector<Event>, EventIdHash, EventIdEq>;

struct BlockEvents {
  const BlockEventMap *Events;
};

struct UnloadedRawLog {
  std::size_t Offset;
  std::size_t Size;
};

using RawLog = std::variant<UnloadedRawLog, std::string>;

struct Block {
  std::string_view Name;
  BlockEventMap Events;
  ev::RawLog RawLog;
};

struct Benchmark {
  std::string Name;
  std::vector<Block> Blocks;
};

struct Logs {
  std::vector<Benchmark> Benchmarks;
};

void defTypes(pybind11::module &Mod);
} // namespace ev
