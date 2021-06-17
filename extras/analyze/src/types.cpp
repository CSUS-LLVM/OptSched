#include "types.hpp"

#include <sstream>

#include <pybind11/stl.h>

#include "parse.hpp"

using namespace std::literals;
namespace py = pybind11;

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

             const Value Val = (*Event.Values)[Event.Start + Index];
             switch (Schema->ParamTypes[Index]) {
             case Type::Number:
               return std::visit(
                   []<typename T>(T x)->py::object {
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
          const Value Val = (*Event.Values)[Event.Start + Index];
          switch (Schema->ParamTypes[Index]) {
          case Type::Number:
            std::visit([&out](auto x) { out << x; }, Val.Num);
          case Type::String:
            out << '\'' << Val.Str << '\'';
          case Type::Bool:
            out << std::boolalpha << Val.Bool;
          }
        }
        out << '}';

        return out.str();
      });

  py::class_<BlockEvents>(Mod, "_BlockEvents")
      .def("__getitem__",
           [](const BlockEvents &Events,
              std::string_view EvId) -> const std::vector<Event> & {
             auto It = Events.Events->find(EventId(EvId));
             if (It != Events.Events->end()) {
               return *It;
             } else {
               throw py::key_error(std::string(EvId));
             }
           });

  py::class_<Block>(Mod, "Block")
      .def_readonly("name", &Block::Name)
      .def_readonly("raw_log", &Block::RawLog)
      .def_property_readonly(
          "events", [](const Block &Blk) { return BlockEvents(&Blk.Events); })
      .def("single",
           [](const Block &Blk, std::string_view EvId) {
             auto It = Blk.Events.find(EventId(EvId));
             if (It != Blk.Events.end()) {
               if (It->size() != 1) {
                 throw std::invalid_argument("Multiple events for " +
                                             std::string(EvId));
               }
               return It->front();
             } else {
               throw py::key_error(std::string(EvId));
             }
           })
      .def("__contains__",
           [](const Block &Blk, std::string_view EvId) {
             return Blk.Events.contains(EventId(EvId));
           })
      .def("__repr__", [](const Block &Blk) {
        return "<Block(" + ", "s + std::to_string(Blk.Events.size()) +
               " events>";
      });

  py::class_<Logs>(Mod, "Logs");
  py::class_<Benchmark>(Mod, "Benchmark");
}
