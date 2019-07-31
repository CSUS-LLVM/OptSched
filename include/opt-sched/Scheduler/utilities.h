/*******************************************************************************
Description:  Contains a few generic utility functions.
Author:       Ghassan Shobaki
Created:      Oct. 1997
Last Update:  Mar. 2017
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_UTILITIES_H
#define OPTSCHED_GENERIC_UTILITIES_H

#include "opt-sched/Scheduler/defines.h"
#include <chrono>

namespace llvm {
namespace opt_sched {

namespace Utilities {
// Calculates the minimum number of bits that can hold a given integer value.
uint16_t clcltBitsNeededToHoldNum(uint64_t value);
// Returns the time that has passed since the start of the process, in
// milliseconds.
Milliseconds GetProcessorTime();
// Returns a reference to an object that is supposed to initialized with the
// start time of the process
extern std::chrono::high_resolution_clock::time_point startTime;
} // namespace Utilities

inline uint16_t Utilities::clcltBitsNeededToHoldNum(uint64_t value) {
  uint16_t bitsNeeded = 0;

  while (value) {
    value >>= 1;
    bitsNeeded++;
  }
  return bitsNeeded;
}

inline Milliseconds Utilities::GetProcessorTime() {
  auto currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = currentTime - startTime;
  return elapsed.count();
}

} // namespace opt_sched
} // namespace llvm

#endif
