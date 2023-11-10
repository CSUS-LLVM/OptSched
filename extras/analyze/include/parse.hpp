#pragma once

#include "py.hpp"

namespace ev {
void defParse(pybind11::module &Mod);

struct EventSchema;

const EventSchema *getSchema(std::string_view Id);
} // namespace ev
