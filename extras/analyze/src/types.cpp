#include "types.hpp"

#include <span>
#include <sstream>

#include "parse.hpp"
#include "py.hpp"

using namespace std::literals;
namespace py = pybind11;

namespace {
template <typename T>
const T &index_into(std::span<const T> Span, std::int64_t index) {
  if (index < 0) {
    // Negative index indexes from the end
    index += Span.size();
  }
  if (index < 0 || static_cast<std::uint64_t>(index) >= Span.size()) {
    throw py::index_error("Index out of bounds: " + std::to_string(index) +
                          "/" + std::to_string(Span.size()));
  }
  return Span[index];
}
} // namespace

void ev::defTypes(py::module &Mod) {
  py::class_<Event>(Mod, "_Event")
      .def("__getitem__",
           [](const Event &Event, std::string_view Property) -> py::object {
             const EventSchema *Schema = ev::getSchema(Event.Id.Value);
             if (!Schema) {
               throw py::key_error("Unknown event " +
                                   std::string(Event.Id.Value));
             }
             auto Index =
                 std::distance(Schema->Parameters.begin(),
                               std::find(Schema->Parameters.begin(),
                                         Schema->Parameters.end(), Property));

             const Value Val = Event.Values[Index];
             switch (Schema->ParamTypes[Index]) {
             case Type::Number:
               return std::visit(
                   []<typename T>(T x) -> py::object {
                     if constexpr (std::same_as<T, double>)
                       return py::float_(x);
                     else
                       return py::int_(x);
                   },
                   Val.Num);
             case Type::String:
               return py::str(std::string(Val.Str));
             case Type::Bool:
               return py::bool_(Val.Bool);
             }
             std::abort();
           })
      .def("__repr__", [](const Event &Event) {
        const EventSchema *Schema = ev::getSchema(Event.Id.Value);
        if (!Schema) {
          throw py::key_error("Unknown event " + std::string(Event.Id.Value));
        }

        std::ostringstream out;
        out << '{';
        for (std::size_t Index = 0; Index < Schema->Parameters.size();
             ++Index) {
          if (Index != 0)
            out << ", ";
          out << '\'' << Schema->Parameters[Index] << "': ";
          const Value Val = Event.Values[Index];
          switch (Schema->ParamTypes[Index]) {
          case Type::Number:
            std::visit([&out](auto x) { out << x; }, Val.Num);
            break;
          case Type::String:
            out << '\'' << Val.Str << '\'';
            break;
          case Type::Bool:
            out << std::boolalpha << Val.Bool;
            break;
          }
        }
        out << '}';

        return out.str();
      });

  py::class_<Block>(Mod, "Block")
      .def_readonly("name", &Block::Name)
      .def_readonly("raw_log", &Block::RawLog)
      .def("__getitem__",
           [](const Block &Blk,
              std::string_view EvId) -> const std::vector<Event> & {
             auto It = Blk.Events.find(EventId(EvId));
             if (It != Blk.Events.end()) {
               return *It;
             } else {
               throw py::key_error(std::string(EvId));
             }
           })
      .def("_event_names",
           [](const Block &Blk) {
             std::vector<std::string_view> Names;
             Names.reserve(Blk.Events.size());

             for (const auto &Events : Blk.Events) {
               Names.push_back(ev::getId(Events).Value);
             }

             return Names;
           })
      .def_readonly("uniqueid", &Block::UniqueId)
      .def("__contains__",
           [](const Block &Blk, std::string_view EvId) {
             return Blk.Events.contains(EventId(EvId));
           })
      .def("__repr__", [](const Block &Blk) {
        return "<Block(bench="s + Blk.Bench->Name + ", file="s + Blk.File +
               ", "s + std::to_string(Blk.Events.size()) + " events)>";
      });

  struct BenchmarkBlocks {
    std::span<const Block> Blocks;
  };

  py::class_<BenchmarkBlocks>(Mod, "_Blocks")
      .def("__getitem__",
           [](const BenchmarkBlocks &Blocks, std::int64_t index) {
             return ::index_into(Blocks.Blocks, index);
           })
      .def("__len__",
           [](const BenchmarkBlocks &Blocks) { return Blocks.Blocks.size(); });

  py::class_<Benchmark, std::shared_ptr<Benchmark>>(Mod, "Benchmark")
      .def_readonly("name", &Benchmark::Name)
      .def_readonly("raw_log", &Benchmark::RawLog)
      .def_property_readonly(
          "blocks",
          [](const Benchmark &Bench) { return BenchmarkBlocks{Bench.Blocks}; })
      .def("__repr__", [](const Benchmark &Bench) {
        return "<Benchmark(name=" + Bench.Name + ", " +
               std::to_string(Bench.Blocks.size()) + " blocks)>";
      });

  struct LogsBenchmarks {
    std::span<const std::shared_ptr<Benchmark>> Benchmarks;
  };
  py::class_<LogsBenchmarks>(Mod, "_Benchmarks")
      .def("__getitem__",
           [](const LogsBenchmarks &Benchmarks, std::int64_t index) {
             return ::index_into(Benchmarks.Benchmarks, index);
           })
      .def("__len__", [](const LogsBenchmarks &Benchmarks) {
        return Benchmarks.Benchmarks.size();
      });

  py::class_<Logs, std::shared_ptr<Logs>>(Mod, "Logs")
      .def_property_readonly(
          "benchmarks",
          [](const ev::Logs &Logs) { return LogsBenchmarks{Logs.Benchmarks}; })
      .def_readonly("raw_log", &Logs::RawLog)
      .def("benchmark",
           [](const ev::Logs &Logs, const std::string_view BenchName) {
             auto It =
                 std::find_if(Logs.Benchmarks.begin(), Logs.Benchmarks.end(),
                              [BenchName](const auto &Bench) {
                                return Bench->Name == BenchName;
                              });

             if (It == Logs.Benchmarks.end()) {
               throw py::key_error("No benchmark `" + std::string(BenchName) +
                                   "` in this Logs");
             } else {
               return It->get();
             }
           })
      .def("__iter__", [](py::handle Logs) { return Logs.attr("benchmarks"); })
      .def("__repr__", [](const ev::Logs &Logs) {
        std::string Result = "<Logs(";
        bool First = true;
        for (const auto &Bench : Logs.Benchmarks) {
          if (!First)
            Result += ", ";
          First = false;

          Result += Bench->Name;
        }

        return Result + ")>";
      });
}
