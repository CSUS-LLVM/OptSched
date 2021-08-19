#include "parse.hpp"
#include "py.hpp"
#include "types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(eventanalyze, Mod) {
  Mod.doc() = "C++-accelerated event logging types and parser";

  Mod.attr("VERSION") = std::tuple(1, 0, 0);

  ev::defTypes(Mod);
  ev::defParse(Mod);
}
