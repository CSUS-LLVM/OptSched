//===- OptSchedGenericTarget.cpp - Generic Target -------------------------===//
//
// Implements a generic target stub.
//
//===----------------------------------------------------------------------===//
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace llvm;
using namespace llvm::opt_sched;

namespace {

class OptSchedGenericTarget : public OptSchedTarget {
public:
  OptSchedGenericTarget(OptSchedMachineModel *MM)
    : OptSchedTarget(MM) {}

  void initRegion() override {}
  void finalizeRegion() override {}
};

} // end anonymous namespace

namespace llvm {
namespace opt_sched {

std::unique_ptr<OptSchedTarget>
createOptSchedGenericTarget(OptSchedMachineModel *MM) {
  return llvm::make_unique<OptSchedGenericTarget>(MM);
}

} // namespace opt_sched
} // namespace llvm
