#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <string>
#include <string_view>

#include <iostream>

namespace pybind11::detail {
template <> struct type_caster<std::filesystem::path> {
public:
  PYBIND11_TYPE_CASTER(std::filesystem::path, _("pathlib.Path | str"));

  // Python -> C++
  bool load(handle Src, bool) {
    // If !isinstance(Src, str):
    if (!PyUnicode_Check(Src.ptr())) {
      object PyPath = module::import("pathlib").attr("Path");

      if (!PyObject_IsInstance(Src.ptr(), PyPath.ptr()))
        return false;
    }
    this->value = std::filesystem::path(std::string(str(Src)));
    return true;
  }

  static handle cast(const std::filesystem::path &Path, return_value_policy,
                     handle) {
    object PyPath = module::import("pathlib").attr("Path");
    return PyPath(str(Path.string()));
  }
};
} // namespace pybind11::detail
