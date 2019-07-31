/*******************************************************************************
Description:  Contains common includes, constants, typedefs and enums.
Author:       Ghassan Shobaki
Created:      Oct. 1997
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_DEFINES_H
#define OPTSCHED_GENERIC_DEFINES_H

// Define basic constants like NULL.
#include <cstddef>

// For integral types of specific byte length.
// The new standard <cinttypes> is still not supported everywhere.
#include <cassert>
#include <stdint.h>

namespace llvm {
namespace opt_sched {

// The standard time unit.
typedef int64_t Milliseconds;

// Instruction count.
typedef int InstCount;

// A generic sentinel value. Should be used with care.
// TODO(max): Get rid of this in favor of type- or purpose-specific sentinels.
const int INVALID_VALUE = -1;

// Possible function call outcomes.
enum FUNC_RESULT {
  // The function encountered an error.
  RES_ERROR = -1,
  // The function consciously failed.
  RES_FAIL = 0,
  // The function succeeded.
  RES_SUCCESS = 1,
  // The function reached the end of the resource (e.g. file) it operated on.
  RES_END = 2,
  // The function did not finish in the time allocated for it.
  RES_TIMEOUT = 3
};

} // namespace opt_sched
} // namespace llvm

#endif
