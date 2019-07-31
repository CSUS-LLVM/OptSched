//===- OptSchedGenericTarget.cpp - Generic Target -------------------------===//
//
// Implements a generic target stub.
//
//===----------------------------------------------------------------------===//
#include "OptSchedDDGWrapperBasic.h"
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include <memory>

using namespace llvm;
using namespace llvm::opt_sched;

OptSchedRegistry<OptSchedTargetRegistry::OptSchedTargetFactory>
    OptSchedTargetRegistry::Registry;

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
                   const std::string &RegionID) override {
    return llvm::make_unique<OptSchedDDGWrapperBasic>(
        Context, DAG, MM, LatencyPrecision, RegionID);
  }

  void initRegion(llvm::ScheduleDAGInstrs *DAG, MachineModel *MM_) override {
    MM = MM_;
  }
  void finalizeRegion(const InstSchedule *Schedule) override {}
  // For generic target find total PRP.
  InstCount getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const override;
};

} // end anonymous namespace

InstCount OptSchedGenericTarget::getCost(
    const llvm::SmallVectorImpl<unsigned> &PRP) const {
  InstCount TotalPRP = 0;
  for (int16_t T = 0; T < MM->GetRegTypeCnt(); ++T)
    TotalPRP += PRP[T];
  return TotalPRP;
}

namespace llvm {
namespace opt_sched {

std::unique_ptr<OptSchedTarget> createOptSchedGenericTarget() {
  return llvm::make_unique<OptSchedGenericTarget>();
}

OptSchedTargetRegistry
    OptSchedGenericTargetRegistry("generic", createOptSchedGenericTarget);

} // namespace opt_sched
} // namespace llvm
