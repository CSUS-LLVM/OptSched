#include <pybind11/pybind11.h>

#include "types.hpp"
#include "parse.hpp"

namespace py = pybind11;

PYBIND11_MODULE(eventanalyze, Mod) {
  Mod.doc() = "C++-accelerated event logging types and parser";

  Mod.attr("VERSION") = std::tuple(1, 0);

  ev::defTypes(Mod);
  ev::defParse(Mod);
}
