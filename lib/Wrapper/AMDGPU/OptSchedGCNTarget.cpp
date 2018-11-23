//===- OptSchedGCNTarget.cpp - AMDGCN Target ------------------------------===//
//
// AMDGCN OptSched target.
//
//===----------------------------------------------------------------------===//
#include "OptSchedDDGWrapperGCN.h"
#include "Wrapper/OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace llvm;
using namespace llvm::opt_sched;

namespace {

class OptSchedGCNTarget : public OptSchedTarget {
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
		return llvm::make_unique<OptSchedDDGWrapperGCN>(
      Context, DAG, MM, LatencyPrecision, GraphTransTypes, RegionID);
  }

  void initRegion() override {}
  void finalizeRegion() override {}
};

} // end anonymous namespace

namespace llvm {
namespace opt_sched {

std::unique_ptr<OptSchedTarget>
createOptSchedGCNTarget() {
  return llvm::make_unique<OptSchedGCNTarget>();
}

OptSchedTargetRegistry
    OptSchedGCNTargetRegistry("amdgcn", createOptSchedGCNTarget);

} // namespace opt_sched
} // namespace llvm
