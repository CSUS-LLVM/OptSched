#ifndef OPTSCHED_SIMPLE_MACHINE_MODEL_H
#define OPTSCHED_SIMPLE_MACHINE_MODEL_H

#include <string.h> // strdup is in the C header, but not the C++ header

#include "opt-sched/Scheduler/buffers.h"
#include "opt-sched/Scheduler/machine_model.h"

inline llvm::opt_sched::MachineModel simpleMachineModel() {
  static constexpr const char SimpleModel[] = R"(
MODEL_NAME: Simple

# The limit on the total number of instructions that can be issued in one cycle
ISSUE_RATE: 1

# Each instruction must have an issue type, i.e. a function unit that the instruction uses.
ISSUE_TYPE_COUNT: 1

# Default issue type for LLVM instructions.
Default 1

DEP_LATENCY_ANTI: 0
DEP_LATENCY_OUTPUT: 1
DEP_LATENCY_OTHER: 1

# This will not be used. Reg type info will be taken from the compiler.
REG_TYPE_COUNT: 2
I 1
F 1

# Set this to the total number of instructions
INST_TYPE_COUNT: 2

INST_TYPE: artificial
ISSUE_TYPE: Default
LATENCY: 0
PIPELINED: YES
BLOCKS_CYCLE: NO
SUPPORTED: NO

INST_TYPE: Inst
ISSUE_TYPE: Default
LATENCY: 1
PIPELINED: YES
BLOCKS_CYCLE: NO
SUPPORTED: YES
  )";

  llvm::opt_sched::SpecsBuffer Buf(strdup(SimpleModel), sizeof(SimpleModel));
  llvm::opt_sched::MachineModel Model(Buf);
  return Model;
}

#endif
