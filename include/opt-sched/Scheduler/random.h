/*******************************************************************************
Description:  Implements the Kahan's random number generator, with a period of
              2 ** 40.
Author:       Ghassan Shobaki
Created:      Unknown
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_RANDOM_H
#define OPTSCHED_GENERIC_RANDOM_H

#include "opt-sched/Scheduler/defines.h"

namespace llvm {
namespace opt_sched {

namespace RandomGen {
// Initialize the random number generator with a seed.
void SetSeed(int32_t iseed);
// Get a random 32-bit value.
uint32_t GetRand32();
// Get a random 32-bit value within a given range, inclusive.
uint32_t GetRand32WithinRange(uint32_t min, uint32_t max);
// Get a random 64-bit value.
uint64_t GetRand64();
// Fill a buffer with a specified number of random bits, rounded to the
// nearest byte boundary.
void GetRandBits(uint16_t bitCnt, unsigned char *dest);
} // namespace RandomGen

} // namespace opt_sched
} // namespace llvm

#endif
