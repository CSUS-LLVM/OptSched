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
