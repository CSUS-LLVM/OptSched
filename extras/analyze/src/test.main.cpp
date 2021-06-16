#include <algorithm>
#include <cassert>
#include <charconv>
#include <deque>
#include <execution>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mio/mmap.hpp>

using namespace std::literals;

enum class Type {
  Int,
  NegInt,
  Float,
  String,
  Bool,
};

union Value {
  std::uint64_t I;
  std::int64_t SI;
  double F;
  std::string_view S;
  bool B;
};

struct EventId {
  std::string_view value;

  bool operator==(const EventId &) const = default;
};

struct EventSchema {
  EventId id;
  std::vector<std::string_view> parameters;
  std::vector<Type> param_types;

  bool operator==(const EventSchema &) const = default;
  bool operator==(EventId rhs) const { return id == rhs; }
};

namespace std {
template <> struct hash<EventId> {
  std::size_t operator()(EventId id) const noexcept {
    return std::hash<std::string_view>()(id.value);
  }
};
template <> struct hash<EventSchema> {
  std::size_t operator()(const EventSchema &schema) const noexcept {
    return std::hash<EventId>()(schema.id);
  }
};
} // namespace std

struct TransparentHash {
  using is_transparent = void;

  template <typename T>
  std::size_t operator()(const T &it) const
      noexcept(noexcept(std::hash<T>{}(it))) {
    return std::hash<T>{}(it);
  }
};

thread_local std::deque<Value> Values;

struct Event {
  EventId id;
  std::deque<Value> const *values;
  std::size_t start; // Indices into Values.
  std::size_t end;
};

struct Block {
  std::string_view name;
  std::unordered_map<EventId, std::vector<Event>> events;
  std::string_view raw_log;
};

static constexpr std::string_view RegionNameEv =
    R"("event_id": "ProcessDag", "name": ")";
static const std::boyer_moore_horspool_searcher
    BlockNameSearcher(RegionNameEv.begin(), RegionNameEv.end());

std::string_view parse_name(const std::string_view block_log) {
  auto it = std::search(block_log.begin(), block_log.end(), BlockNameSearcher);
  it += RegionNameEv.size();
  auto end = std::find(it, block_log.end(), '"');

  return std::string_view(it, end);
}

EventSchema parse_event_schema(
    EventId id,
    const std::vector<std::pair<std::string_view, std::string_view>> &init) {
  EventSchema result;
  result.id = id;
  result.param_types.reserve(init.size() - 1);
  result.parameters.reserve(init.size() - 1);

  for (std::size_t index = 0; index < init.size() - 1; ++index) {
    result.parameters.push_back(init[index + 1].first);
    assert(!init[index + 1].second.empty());
    if (init[index + 1].second.front() == '"') {
      result.param_types.push_back(Type::String);
    } else if (init[index + 1].second == "true"sv ||
               init[index + 1].second == "false"sv) {
      result.param_types.push_back(Type::Bool);
    } else if (init[index + 1].second.front() == '-') {
      result.param_types.push_back(Type::NegInt);
    } else {
      result.param_types.push_back(Type::Int);
    }
  }

  return result;
}

static std::unordered_set<EventSchema, TransparentHash, std::equal_to<>>
    MasterSchemas;
std::mutex MasterSchemaMutex;
thread_local std::unordered_set<EventSchema, TransparentHash, std::equal_to<>>
    Schemas;

void update_schema_structures(EventId id, EventSchema schema) {
  std::scoped_lock sl(MasterSchemaMutex);
  if (MasterSchemas.find(id) == MasterSchemas.end())
    MasterSchemas.emplace_hint(MasterSchemas.end(), std::move(schema));
  Schemas = MasterSchemas;
}

Event parse_event(const std::string_view event) {
  const auto end = event.rfind('}');
  auto begin = event.find('{');

  std::vector<std::pair<std::string_view, std::string_view>> result;

  while (begin < end) {
    const auto key_f = event.find('"', begin + 1) + 1;
    if (key_f == std::string_view::npos)
      break;
    const auto key_e = event.find('"', key_f);
    if (key_e == std::string_view::npos)
      break;
    const std::string_view key = event.substr(key_f, key_e - key_f);
    const auto val_f =
        event.find_first_not_of(" \t\n", event.find(':', key_e + 1) + 1);
    if (val_f == std::string_view::npos)
      break;
    const auto val_e = event[val_f] == '"'
                           ? event.find('"', val_f + 1) + 1
                           : event.find_first_of(",} \t\n", val_f + 1);
    if (val_e == std::string_view::npos)
      break;
    std::string_view val = event.substr(val_f, val_e - val_f);

    result.emplace_back(key, val);
    begin = event.find_first_of(",}", val_e + 1);
    if (begin == std::string_view::npos)
      break;
    begin += 1;
  }

  assert(result[0].first == "event_id"sv);
  EventId id(result[0].second);

  auto it = Schemas.find(id);
  if (it == Schemas.end()) {
    auto sch = ::parse_event_schema(id, result);
    ::update_schema_structures(id, std::move(sch));
    it = Schemas.find(id);
  }

  assert(it->param_types.size() == result.size() - 1);
  std::size_t start = Values.size();
  for (std::size_t index = 0; index < result.size() - 1; ++index) {
    Values.push_back([&]() -> Value {
      switch (it->param_types[index]) {
      case Type::Int: {
        std::uint64_t val;
        [[maybe_unused]] const auto r = std::from_chars(
            result[index + 1].second.data(),
            result[index + 1].second.data() + result[index + 1].second.size(),
            val);

        assert(r.ptr == result[index + 1].second.data() +
                            result[index + 1].second.size());
        assert(r.ec == std::errc());

        return Value{.I = val};
      }
      case Type::NegInt: {
        std::int64_t val;
        [[maybe_unused]] const auto r = std::from_chars(
            result[index + 1].second.data(),
            result[index + 1].second.data() + result[index + 1].second.size(),
            val);

        assert(r.ptr == result[index + 1].second.data() +
                            result[index + 1].second.size());
        assert(r.ec == std::errc());

        return Value{.SI = val};
      }
      case Type::Float: {
        double val;
        [[maybe_unused]] const auto r = std::from_chars(
            result[index + 1].second.data(),
            result[index + 1].second.data() + result[index + 1].second.size(),
            val);

        assert(r.ptr == result[index + 1].second.data() +
                            result[index + 1].second.size());
        assert(r.ec == std::errc());

        return Value{.F = val};
      }
      case Type::String:
        return Value{.S = result[index + 1].second.substr(
                         1, result[index + 1].second.size() - 2)};
      case Type::Bool:
        return Value{.B = result[index + 1].second == "true"sv};
      }
      std::abort();
    }());
  }
  std::size_t iend = Values.size();

  return Event{
      .id = id,
      .values = &Values,
      .start = start,
      .end = iend,
  };
}

static constexpr std::string_view EventTag = R"(EVENT: {)";
static const std::boyer_moore_horspool_searcher
    EventTagSearcher(EventTag.begin(), EventTag.end());
std::unordered_map<EventId, std::vector<Event>>
parse_events(const std::string_view block_log) {
  std::unordered_map<EventId, std::vector<Event>> result;

  const auto e = block_log.end();
  auto b = std::search(block_log.begin(), e, EventTagSearcher);
  while (b != e) {
    auto line_end = std::find(b + EventTag.size() - 1, e, '\n');

    std::string_view event(b, line_end);

    Event ev = ::parse_event(event);
    result[ev.id].push_back(ev);

    b = std::search(line_end, e, EventTagSearcher);
  }

  return result;
}

Block parse_block(const std::string_view block_log) {
  return Block{
      .name = ::parse_name(block_log),
      .events = ::parse_events(block_log),
      .raw_log = block_log,
  };
}

std::vector<std::string_view> split_blocks(const std::string_view file) {
  static constexpr std::string_view RegionDelimiter =
      "********** Opt Scheduling **********";
  const std::boyer_moore_horspool_searcher searcher(RegionDelimiter.begin(),
                                                    RegionDelimiter.end());

  std::vector<std::string_view> result;

  const auto e = file.end();
  auto b = std::search(file.begin(), e, searcher);
  while (b != e) {
    auto it = std::search(b + RegionDelimiter.size(), e, searcher);
    result.emplace_back(file.data() + std::distance(file.begin(), b),
                        std::distance(b, it));
    b = it;
  }

  return result;
}

int main(int argc, char **argv) {
  mio::mmap_source mmap(argv[1]);
  std::string_view file(mmap.data(), mmap.size());
  const auto raw_blocks = ::split_blocks(file);
  std::vector<Block> blocks(raw_blocks.size());
  std::transform(
#if HAS_TBB
      std::execution::par_unseq,
#endif
      raw_blocks.begin(), raw_blocks.end(), blocks.begin(),
      [](std::string_view blk) { return ::parse_block(blk); });

  std::cout << "done " << blocks.size() << std::endl;
  std::cout << "done " << MasterSchemas.size() << std::endl;
}
