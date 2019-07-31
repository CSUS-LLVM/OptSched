#include "opt-sched/Scheduler/random.h"
// For memcpy().
#include <algorithm>
#include <cstring>

using namespace llvm::opt_sched;

// Magic numbers used in the generator formula.
static const uint32_t A = 0x2faf071d; // 8 * (10 ** 8 - 29) + 5
static const uint32_t C = 0x3b9ac9c1; // 10 ** 9 - 63

// Magic lookup table.
static uint32_t Z[] = {
    0x8ca0df45, 0x37334f23, 0x4a5901d2, 0xaeede075, 0xd84bd3cf, 0xa1ce3350,
    0x35074a8f, 0xfd4e6da0, 0xe2c22e6f, 0x045de97e, 0x0e6d45b9, 0x201624a2,
    0x01e10dca, 0x2810aef2, 0xea0be721, 0x3a3781e4, 0xa3602009, 0xd2ffcf69,
    0xff7102e9, 0x36fab972, 0x5c3650ff, 0x8cd44c9c, 0x25a4a676, 0xbd6385ce,
    0xcd55c306, 0xec8a31f5, 0xa87b24ce, 0x1e025786, 0x53d713c9, 0xb29d308f,
    0x0dc6cf3f, 0xf11139c9, 0x3afb3780, 0x0ed6b24c, 0xef04c8fe, 0xab53d825,
    0x3ca69893, 0x35460fb1, 0x058ead73, 0x0b567c59, 0xfdddca3f, 0x6317e77d,
    0xaa5febe5, 0x655f73e2, 0xd42455bb, 0xe845a8bb, 0x351e4a67, 0xa36a9dfb,
    0x3e0ac91d, 0xbaa0de01, 0xec60dc66, 0xdb29309e, 0xcfa52971, 0x1f3eddaf,
    0xe14aae61,
};

// The current generator state. Magical starting values.
static long j = 23;
static long k = 54;
static uint32_t y[] = {
    0x8ca0df45, 0x37334f23, 0x4a5901d2, 0xaeede075, 0xd84bd3cf, 0xa1ce3350,
    0x35074a8f, 0xfd4e6da0, 0xe2c22e6f, 0x045de97e, 0x0e6d45b9, 0x201624a2,
    0x01e10dca, 0x2810aef2, 0xea0be721, 0x3a3781e4, 0xa3602009, 0xd2ffcf69,
    0xff7102e9, 0x36fab972, 0x5c3650ff, 0x8cd44c9c, 0x25a4a676, 0xbd6385ce,
    0xcd55c306, 0xec8a31f5, 0xa87b24ce, 0x1e025786, 0x53d713c9, 0xb29d308f,
    0x0dc6cf3f, 0xf11139c9, 0x3afb3780, 0x0ed6b24c, 0xef04c8fe, 0xab53d825,
    0x3ca69893, 0x35460fb1, 0x058ead73, 0x0b567c59, 0xfdddca3f, 0x6317e77d,
    0xaa5febe5, 0x655f73e2, 0xd42455bb, 0xe845a8bb, 0x351e4a67, 0xa36a9dfb,
    0x3e0ac91d, 0xbaa0de01, 0xec60dc66, 0xdb29309e, 0xcfa52971, 0x1f3eddaf,
    0xe14aae61,
};

// The last random number.
static uint32_t randNum;

void GenerateNextNumber() {
  randNum = y[j] + y[k];
  y[k] = randNum;
  if (--j < 0)
    j = 54;
  if (--k < 0)
    k = 54;
  randNum &= 0x7fffffff;
}

void RandomGen::SetSeed(int32_t iseed) {
  j = 23;
  k = 54;

  if (iseed == 0) {
    for (int32_t i = 0; i < 55; i++) {
      y[i] = Z[i];
    }
  } else {
    y[0] = (A * iseed + C) >> 1;
    for (int32_t i = 1; i < 55; i++) {
      y[i] = (A * y[i - 1] + C) >> 1;
    }
  }
}

uint32_t RandomGen::GetRand32WithinRange(uint32_t min, uint32_t max) {
  GenerateNextNumber();
  return randNum % (max - min + 1) + min;
}

uint32_t RandomGen::GetRand32() {
  GenerateNextNumber();
  return randNum;
}

uint64_t RandomGen::GetRand64() {
  uint64_t rand64;

  GenerateNextNumber();
  rand64 = randNum;
  rand64 <<= 32;

  GenerateNextNumber();
  rand64 += randNum;

  return rand64;
}

void RandomGen::GetRandBits(uint16_t bitCnt, unsigned char *dest) {
  uint16_t bytesNeeded = (bitCnt + 7) / 8;
  uint16_t index = 0;

  while (bytesNeeded > 0) {
    GenerateNextNumber();
    uint16_t bytesConsumed = std::min(bytesNeeded, (uint16_t)4);
    memcpy(dest + index, &randNum, bytesConsumed);
    index += bytesConsumed;
    bytesNeeded -= bytesConsumed;
  }
}
