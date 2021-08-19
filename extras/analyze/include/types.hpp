#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <mio/mmap.hpp>
#include <pybind11/pybind11.h>

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
};

struct Event {
  EventId Id;
  const std::deque<Value> *Values;
  std::size_t Start; // Indices into values.
  std::size_t End;
};

inline EventId getId(EventId Id) { return Id; }

template <typename T>
requires requires(const T &It) {
  { It.Id } -> std::convertible_to<EventId>;
}
EventId getId(const T &It) { return It.Id; }

template <typename T> EventId getId(const std::vector<T> &Vec) {
  assert(!Vec.empty());
  return getId(Vec.front());
}

struct EventIdHash {
  using is_transparent = void;

  std::size_t operator()(std::string_view Id) const noexcept {
    return std::hash<std::string_view>()(Id);
  }

  std::size_t operator()(EventId Id) const noexcept {
    return (*this)(Id.Value);
  }

  template <typename T> std::size_t operator()(const T &It) const noexcept {
    return (*this)(getId(It));
  }
};

struct EventIdEq {
  using is_transparent = void;

  bool operator()(EventId Lhs, EventId Rhs) const { return Lhs == Rhs; }

  template <typename T, typename U>
  bool operator()(const T &Lhs, const U &Rhs) const {
    return getId(Lhs) == getId(Rhs);
  }
};

using BlockEventMap =
    std::unordered_set<std::vector<Event>, EventIdHash, EventIdEq>;

struct Logs;
struct Benchmark;

struct Block {
  std::string_view Name;
  BlockEventMap Events;
  // Offset & size into the mmapped file.
  std::size_t Offset;
  std::size_t Size;

  std::string UniqueId;

  ev::Benchmark *Bench;

  std::string File; // Which file was compiled for this block
};

struct Benchmark {
  std::string Name;
  std::vector<Block> Blocks;
  // Offset & size into the mmapped file.
  std::size_t Offset;
  std::size_t Size;

  // Keep the memory around so that we can detect if the Logs object was
  // destroyed, giving the Python user a good error message.
  std::weak_ptr<ev::Logs> Logs;
};

struct Logs {
  std::string LogFile;
  mio::mmap_source MMap;
  std::vector<std::shared_ptr<Benchmark>> Benchmarks;
};

void defTypes(pybind11::module &Mod);
} // namespace ev
