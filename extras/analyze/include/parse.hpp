#pragma once

#include <pybind11/pybind11.h>

namespace ev {
void defParse(pybind11::module &Mod);

struct EventSchema;

const EventSchema *getSchema(std::string_view Id);
} // namespace ev
