//===- OptSchedGenericTarget.cpp - Generic Target -------------------------===//
//
// Implements a generic target stub.
//
//===----------------------------------------------------------------------===//
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace llvm::opt_sched;

namespace {

class OptSchedGenericTarget : public OptSchedTarget {
public:
  OptSchedGenericTarget(::opt_sched::MachineModel *MM)
    : OptSchedTarget(MM) {}

  void initRegion() override {}
  void finalizeRegion() override {}
};

} // end anonymous namespace

namespace llvm {
namespace opt_sched {

std::unique_ptr<OptSchedTarget>
createOptSchedGenericTarget(::opt_sched::MachineModel *MM) {
  return llvm::make_unique<OptSchedGenericTarget>(MM);
}

OptSchedTargetRegistry
    OptSchedGenericTargetRegistry("generic", createOptSchedGenericTarget);

} // namespace opt_sched
} // namespace llvm
