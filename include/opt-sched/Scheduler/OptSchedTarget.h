//===- OptSchedTarget.h - OptSched Target -----------------------*- C++-*--===//
//
// Interface for target specific functionality in OptSched. This is a workaround
// to avoid needing to modify or use target code in the trunk.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_TARGET_H
#define LLVM_OPT_SCHED_TARGET_H

#include "opt-sched/Scheduler/machine_model.h"

namespace llvm {
namespace opt_sched {

class OptSchedTarget {
public:
  MachineModel *MM;

  OptSchedTarget(MachineModel *MM_) : MM(MM_) {}
  virtual ~OptSchedTarget() = default;

  virtual void initRegion() = 0;
  virtual void finalizeRegion() = 0;
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_OPT_SCHED_TARGET_H
