//===- OptSchedGenericTarget.cpp - Generic Target -------------------------===//
//
// Implements a generic target stub.
//
//===----------------------------------------------------------------------===//
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace llvm;
using namespace llvm::opt_sched;

namespace {

class OptSchedGenericTarget : public OptSchedTarget {
public:
  std::unique_ptr<OptSchedMachineModel>
  createMachineModel(const char *ConfigPath) override {
    return llvm::make_unique<OptSchedMachineModel>(ConfigPath);
  }

  std::unique_ptr<OptSchedDDGWrapperBase>
  createDDGWrapper(llvm::MachineSchedContext *Context, ScheduleDAGOptSched *DAG,
                   OptSchedMachineModel *MM, LATENCY_PRECISION LatencyPrecision,
                   GraphTransTypes GraphTransTypes,
                   const std::string &RegionID) override {
    return llvm::make_unique<OptSchedDDGWrapperBasic>(
        Context, DAG, MM, LatencyPrecision, GraphTransTypes, RegionID);
  }

  void initRegion(const llvm::MachineSchedContext *Context,
                  const MachineModel *MM) override {}
  void finalizeRegion(const InstSchedule *Schedule) override {}
};

} // end anonymous namespace

namespace llvm {
namespace opt_sched {

std::unique_ptr<OptSchedTarget> createOptSchedGenericTarget() {
  return llvm::make_unique<OptSchedGenericTarget>();
}

OptSchedTargetRegistry
    OptSchedGenericTargetRegistry("generic", createOptSchedGenericTarget);

} // namespace opt_sched
} // namespace llvm
