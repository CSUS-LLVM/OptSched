#include "parse.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <deque>
#include <execution>
#include <iostream>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mio/mmap.hpp>
#include <pybind11/stl.h>

#include "types.hpp"

using namespace std::literals;
using namespace ev;
namespace py = pybind11;

thread_local std::deque<Value> Values;
thread_local std::unordered_set<std::string> Strings;

static constexpr std::string_view RegionNameEv =
    R"("event_id": "ProcessDag", "name": ")";
static const std::boyer_moore_horspool_searcher
    BlockNameSearcher(RegionNameEv.begin(), RegionNameEv.end());

static std::string_view parseName(const std::string_view BlockLog) {
  auto It = std::search(BlockLog.begin(), BlockLog.end(), BlockNameSearcher);
  It += RegionNameEv.size();
  auto End = std::find(It, BlockLog.end(), '"');

  return std::string_view(It, End);
}

static EventSchema parse_event_schema(
    EventId Id,
    const std::vector<std::pair<std::string_view, std::string_view>> &Init) {
  EventSchema Result;
  Result.Id = EventId(*Strings.insert(std::string(Id.Value)).first);
  Result.ParamTypes.reserve(Init.size() - 1);
  Result.Parameters.reserve(Init.size() - 1);

  for (std::size_t Index = 0; Index < Init.size() - 1; ++Index) {
    Result.Parameters.push_back(
        *Strings.insert(std::string(Init[Index + 1].first)).first);
    assert(!Init[Index + 1].second.empty());
    if (Init[Index + 1].second.front() == '"') {
      Result.ParamTypes.push_back(Type::String);
    } else if (Init[Index + 1].second == "true"sv ||
               Init[Index + 1].second == "false"sv) {
      Result.ParamTypes.push_back(Type::Bool);
    } else {
      Result.ParamTypes.push_back(Type::Number);
    }
  }

  return Result;
}

static std::unordered_set<EventSchema, EventIdHash, EventIdEq> MasterSchemas;
static std::mutex MasterSchemaMutex;
thread_local std::unordered_set<EventSchema, EventIdHash, EventIdEq> Schemas;

static void update_schema_structures(EventId Id, EventSchema schema) {
  std::scoped_lock Lock(MasterSchemaMutex);
  if (MasterSchemas.find(Id) == MasterSchemas.end())
    MasterSchemas.emplace_hint(MasterSchemas.end(), std::move(schema));
  Schemas = MasterSchemas;
}

static Event parseEvent(const std::string_view Event) {
  const auto End = Event.rfind('}');
  auto Begin = Event.find('{');

  std::vector<std::pair<std::string_view, std::string_view>> Result;

  while (Begin < End) {
    const auto KeyF = Event.find('"', Begin + 1) + 1;
    if (KeyF == std::string_view::npos)
      break;
    const auto KeyE = Event.find('"', KeyF);
    if (KeyE == std::string_view::npos)
      break;
    const std::string_view Key = Event.substr(KeyF, KeyE - KeyF);
    const auto ValF =
        Event.find_first_not_of(" \t\n", Event.find(':', KeyE + 1) + 1);
    if (ValF == std::string_view::npos)
      break;
    const auto ValE = Event[ValF] == '"'
                          ? Event.find('"', ValF + 1) + 1
                          : Event.find_first_of(",} \t\n", ValF + 1);
    if (ValE == std::string_view::npos)
      break;
    std::string_view Val = Event.substr(ValF, ValE - ValF);

    Result.emplace_back(Key, Val);
    Begin = Event.find_first_of(",}", ValE + 1);
    if (Begin == std::string_view::npos)
      break;
    Begin += 1;
  }

  assert(Result[0].first == "event_id"sv);
  EventId Id(Result[0].second);

  auto It = Schemas.find(Id);
  if (It == Schemas.end()) {
    auto Sch = ::parse_event_schema(Id, Result);
    ::update_schema_structures(Id, std::move(Sch));
    It = Schemas.find(Id);
  }

  Id = It->Id; // Update to the non-dangling Id.

  assert(It->ParamTypes.size() == Result.size() - 1);
  std::size_t start = Values.size();
  for (std::size_t Index = 0; Index < Result.size() - 1; ++Index) {
    const std::string_view Data = Result[Index + 1].second;
    Values.push_back([&]() -> Value {
      switch (It->ParamTypes[Index]) {
      case Type::Number: {
        std::int64_t I64;
        [[maybe_unused]] const auto Ri64 =
            std::from_chars(Data.data(), Data.data() + Data.size(), I64);
        if (Ri64.ec == std::errc() && Ri64.ptr == Data.data() + Data.size()) {
          return Value{.Num = Number(I64)};
        }

        std::uint64_t U64;
        [[maybe_unused]] const auto Ru64 =
            std::from_chars(Data.data(), Data.data() + Data.size(), U64);
        if (Ru64.ec == std::errc() && Ru64.ptr == Data.data() + Data.size()) {
          return Value{.Num = Number(U64)};
        }

        double Fl;
        [[maybe_unused]] const auto Rfl =
            std::from_chars(Data.data(), Data.data() + Data.size(), Fl);
        if (Rfl.ec == std::errc() && Rfl.ptr == Data.data() + Data.size()) {
          return Value{.Num = Number(Fl)};
        }
      }
      case Type::String:
        return Value{
            .Str = *Strings.insert(std::string(Data.substr(1, Data.size() - 2)))
                        .first};
      case Type::Bool:
        return Value{.Bool = Data == "true"sv};
      }
      std::abort();
    }());
  }
  std::size_t iend = Values.size();

  return ev::Event{
      .Id = Id,
      .Values = &Values,
      .Start = start,
      .End = iend,
  };
}

static constexpr std::string_view EventTag = R"(EVENT: {)";
static const std::boyer_moore_horspool_searcher
    EventTagSearcher(EventTag.begin(), EventTag.end());

static BlockEventMap parseEvents(const std::string_view BlockLog) {
  std::unordered_map<EventId, std::vector<Event>, EventIdHash, EventIdEq>
      Result;

  const auto E = BlockLog.end();
  auto B = std::search(BlockLog.begin(), E, EventTagSearcher);
  while (B != E) {
    auto line_end = std::find(B + EventTag.size() - 1, E, '\n');

    std::string_view Event(B, line_end);

    ev::Event Ev = ::parseEvent(Event);
    Result[Ev.Id].push_back(Ev);

    B = std::search(line_end, E, EventTagSearcher);
  }

  auto Vals = std::ranges::views::values(Result);

  return BlockEventMap(std::make_move_iterator(Vals.begin()),
                       std::make_move_iterator(Vals.end()));
}

static Block parseBlock(const std::string_view WholeFile,
                        const std::string_view BlockLog) {
  return Block{
      .Name = ::parseName(BlockLog),
      .Events = ::parseEvents(BlockLog),
      .RawLog =
          UnloadedRawLog{
              .Offset = BlockLog.data() - WholeFile.data(),
              .Size = BlockLog.size(),
          },
  };
}

static std::vector<std::string_view> split_blocks(const std::string_view file) {
  static constexpr std::string_view RegionDelimiter =
      "********** Opt Scheduling **********";
  const std::boyer_moore_horspool_searcher searcher(RegionDelimiter.begin(),
                                                    RegionDelimiter.end());

  std::vector<std::string_view> Result;

  const auto E = file.end();
  auto B = std::search(file.begin(), E, searcher);
  while (B != E) {
    auto It = std::search(B + RegionDelimiter.size(), E, searcher);
    Result.emplace_back(file.data() + std::distance(file.begin(), B),
                        std::distance(B, It));
    B = It;
  }

  return Result;
}

void ev::defParse(py::module &Mod) {
  Mod.def("parse_blocks", [](const std::string &path) {
    mio::mmap_source MMap(path.c_str());
    std::string_view File(MMap.data(), MMap.size());
    const auto RawBlocks = ::split_blocks(File);
    std::vector<Block> Blocks(RawBlocks.size());
    std::transform(
#if HAS_TBB
        std::execution::par_unseq,
#endif
        RawBlocks.begin(), RawBlocks.end(), Blocks.begin(),
        [File](std::string_view Blk) { return ::parseBlock(File, Blk); });

    return Blocks;
  });
}

const EventSchema *ev::getSchema(std::string_view Id) {
  auto It = MasterSchemas.find(EventId(Id));
  if (It == MasterSchemas.end())
    return nullptr;
  return &*It;
}
