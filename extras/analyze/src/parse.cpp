#include "parse.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <execution>
#include <filesystem>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "py.hpp"
#include "types.hpp"

using namespace std::literals;
using namespace ev;
namespace py = pybind11;
namespace fs = std::filesystem;

static constexpr std::string_view RegionNameEv =
    R"("event_id": "ProcessDag", "name": ")";
static const std::boyer_moore_horspool_searcher
    BlockNameSearcher(RegionNameEv.begin(), RegionNameEv.end());

// Extracts the name of the block
static std::string_view parseName(const std::string_view BlockLog) {
  auto It = std::search(BlockLog.begin(), BlockLog.end(), BlockNameSearcher);
  It += RegionNameEv.size();
  auto End = std::find(It, BlockLog.end(), '"');

  return std::string_view(It, End);
}

// Parses out an EventSchema, which is shared for all events of that EventId.
static EventSchema parseEventSchema(
    EventId Id,
    const std::vector<std::pair<std::string_view, std::string_view>> &Init) {
  EventSchema Result;
  Result.Id = EventId(Id.Value);
  Result.ParamTypes.reserve(Init.size() - 1);
  Result.Parameters.reserve(Init.size() - 1);

  for (std::size_t Index = 0; Index < Init.size() - 1; ++Index) {
    Result.Parameters.push_back(Init[Index + 1].first);
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

// Schemas are globally loaded.
// This static/thread_local dance is to make it appropriately thread safe but
// still fast.
static absl::flat_hash_set<EventSchema, EventIdHash, EventIdEq> MasterSchemas;
static std::mutex MasterSchemaMutex;
thread_local absl::flat_hash_set<EventSchema, EventIdHash, EventIdEq> Schemas;

static void updateSchemaStructures(EventId Id, EventSchema schema) {
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
    auto Sch = ::parseEventSchema(Id, Result);
    ::updateSchemaStructures(Id, std::move(Sch));
    It = Schemas.find(Id);
  }

  Id = It->Id; // Update to the non-dangling Id.

  assert(It->ParamTypes.size() == Result.size() - 1);
  std::vector<Value> Values;

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
        std::abort();
      }
      case Type::String:
        return Value{.Str = Data.substr(1, Data.size() - 2)};
      case Type::Bool:
        return Value{.Bool = Data == "true"sv};
      }
      std::abort();
    }());
  }

  return ev::Event{.Id = Id, .Values = std::move(Values)};
}

static constexpr std::string_view EventTag = R"(EVENT: {)";
static const std::boyer_moore_horspool_searcher
    EventTagSearcher(EventTag.begin(), EventTag.end());

static BlockEventMap parseEvents(const std::string_view BlockLog) {
  absl::flat_hash_map<EventId, std::vector<Event>, EventIdHash, EventIdEq>
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

static Block parseBlock(ev::Benchmark *Bench, const std::string_view BlockLog) {
  std::string_view Name = ::parseName(BlockLog);
  BlockEventMap Events = ::parseEvents(BlockLog);
  std::string UniqueId = Bench->Name + ':' + std::string(Name);
  auto PF = Events.find(EventId("PassFinished"));
  if (PF != Events.end()) {
    UniqueId +=
        ",pass=" +
        std::to_string(std::get<std::int64_t>(PF->front().Values.front().Num));
  }
  return Block{
      .Name = std::move(Name),
      .Events = std::move(Events),
      .RawLog = BlockLog,
      .UniqueId = std::move(UniqueId),
      .Bench = Bench,
      // Extracting file info costs quite a bit of time, and we never use it
      // anyway.
      .File = "",
  };
}

static std::vector<std::string_view> splitBlocks(const std::string_view file) {
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

namespace {
struct BenchmarkRegion {
  std::string BenchmarkName;
  // The offset in the file
  std::size_t Offset;
};

enum class BenchmarkRE : int {
  Spec,
};
} // namespace

static std::shared_ptr<Benchmark> parse(std::weak_ptr<ev::Logs> Logs,
                                        const std::string_view File,
                                        BenchmarkRegion Bench) {
  const auto RawBlocks = ::splitBlocks(File);
  std::vector<Block> Blocks(RawBlocks.size());

  auto Result = std::make_shared<Benchmark>();
  Result->Name = Bench.BenchmarkName;
  Result->Logs = std::move(Logs);
  Result->RawLog = File;

  std::transform(
#if HAS_TBB
      std::execution::par_unseq,
#endif
      RawBlocks.begin(), RawBlocks.end(), Blocks.begin(),
      [Bench = Result.get()](std::string_view Blk) {
        return ::parseBlock(Bench, Blk);
      });

  Result->Blocks = std::move(Blocks);

  return Result;
}

static constexpr std::string_view SpecBenchmarkRegion = R"(  Building )";
static const std::boyer_moore_horspool_searcher
    SpecBenchmarkSearcher(SpecBenchmarkRegion.begin(),
                          SpecBenchmarkRegion.end());
static std::vector<BenchmarkRegion> splitSpecBenchmarks(std::string_view File) {
  std::vector<BenchmarkRegion> Result;

  auto B = File.begin();
  auto E = File.end();
  while (B != E) {
    auto It = std::search(B, E, SpecBenchmarkSearcher);
    if (It == E)
      break;
    It += SpecBenchmarkRegion.size();
    auto EndOfName = std::find(It, E, ' ');

    const auto Name = std::string_view(It, EndOfName);
    const std::size_t Offset = It - File.begin();

    Result.emplace_back(std::string(Name), Offset);

    B = It;
  }

  return Result;
}

void ev::defParse(py::module &Mod) {
  // static constexpr std::string_view BenchmarkRE
  Mod.attr("SPEC_BENCH_RE") = (int)BenchmarkRE::Spec;

  Mod.def("parse_blocks", [](const fs::path &Path,
                             // One of the RE types.
                             int REChoice) {
    if (REChoice != (int)BenchmarkRE::Spec) {
      throw py::value_error("Unknown regular expression number " +
                            std::to_string(REChoice));
    }
    auto Logs = std::make_shared<ev::Logs>();
    Logs->LogFile = std::move(Path);
    Logs->MMap = mio::mmap_source(Logs->LogFile.string());
    Logs->RawLog = std::string_view(Logs->MMap.data(), Logs->MMap.size());
    const std::string_view File = Logs->RawLog;

    const std::vector<BenchmarkRegion> BenchmarkSections =
        [&]() -> std::vector<BenchmarkRegion> {
      switch ((BenchmarkRE)REChoice) {
      case BenchmarkRE::Spec:
        return splitSpecBenchmarks(File);
      }
      std::abort();
    }();

    Logs->Benchmarks.reserve(BenchmarkSections.size());
    for (std::size_t Index = 0; Index < BenchmarkSections.size(); ++Index) {
      const std::size_t Offset = BenchmarkSections[Index].Offset;
      const std::size_t OffsetEnd = Index + 1 < BenchmarkSections.size()
                                        ? BenchmarkSections[Index + 1].Offset
                                        : File.size();

      const std::string_view Section = File.substr(Offset, OffsetEnd - Offset);

      Logs->Benchmarks.push_back(
          ::parse(Logs, Section, std::move(BenchmarkSections[Index])));
    }

    return Logs;
  });
  Mod.def("parse_blocks", [](const fs::path &Path,
                             // A single benchmark name for the whole logs.
                             std::string_view BenchmarkName) {
    auto Logs = std::make_shared<ev::Logs>();
    Logs->LogFile = std::move(Path);
    Logs->MMap = mio::mmap_source(Logs->LogFile.string());
    Logs->RawLog = std::string_view(Logs->MMap.data(), Logs->MMap.size());
    const std::string_view File = Logs->RawLog;

    Logs->Benchmarks.push_back(
        ::parse(Logs, File, BenchmarkRegion{std::string(BenchmarkName), 0}));

    return Logs;
  });
}

const EventSchema *ev::getSchema(std::string_view Id) {
  auto It = MasterSchemas.find(EventId(Id));
  if (It == MasterSchemas.end())
    return nullptr;
  return &*It;
}
