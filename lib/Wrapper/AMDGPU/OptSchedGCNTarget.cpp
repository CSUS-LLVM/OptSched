//===- OptSchedGCNTarget.cpp - AMDGCN Target ------------------------------===//
//
// AMDGCN OptSched target.
//
//===----------------------------------------------------------------------===//
#include "OptSchedDDGWrapperGCN.h"
#include "Wrapper/OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

using namespace llvm;
using namespace llvm::opt_sched;

#define DEBUG_TYPE "optsched"

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

  void initRegion(const llvm::MachineSchedContext *Context,
                  const MachineModel *MM_) override;
  void finalizeRegion(const InstSchedule *Schedule) override;

  void dumpOccupancyInfo(const InstSchedule *Schedule) const;

private:
  const llvm::MachineFunction *MF;
  const MachineModel *MM;
};

std::unique_ptr<OptSchedTarget> createOptSchedGCNTarget() {
  return llvm::make_unique<OptSchedGCNTarget>();
}

} // end anonymous namespace

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void OptSchedGCNTarget::dumpOccupancyInfo(const InstSchedule *Schedule) const {
  auto &ST = MF->getSubtarget<GCNSubtarget>();
  unsigned MaxOccLDS = ST.getOccupancyWithLocalMemSize(*MF);

  const InstCount *PRP;
  Schedule->GetPeakRegPressures(PRP);
  unsigned SGPR32Count = PRP[MM->GetRegTypeByName("SGPR32")];
  auto MaxOccSGPR = ST.getOccupancyWithNumSGPRs(SGPR32Count);

  unsigned VGPR32Count = PRP[MM->GetRegTypeByName("VGPR32")];
  auto MaxOccVGPR = ST.getOccupancyWithNumVGPRs(VGPR32Count);

  dbgs() << "Estimated Max Occupancy After Scheduling: "
         << std::min(std::min(MaxOccSGPR, MaxOccVGPR), MaxOccLDS) << "\n";
  ;
  dbgs() << "Max Occ with LDS: " << MaxOccLDS << "\n";
  dbgs() << "Max Occ with Num SGPRs: " << MaxOccSGPR << "\n";
  dbgs() << "Max Occ with Num VGPRs: " << MaxOccVGPR << "\n";
}
#endif

void OptSchedGCNTarget::initRegion(const llvm::MachineSchedContext *Context,
                                   const MachineModel *MM_) {
  MF = Context->MF;
  MM = MM_;
}

void OptSchedGCNTarget::finalizeRegion(const InstSchedule *Schedule) {
  LLVM_DEBUG(dumpOccupancyInfo(Schedule));
}

namespace llvm {
namespace opt_sched {

OptSchedTargetRegistry OptSchedGCNTargetRegistry("amdgcn",
                                                 createOptSchedGCNTarget);

} // namespace opt_sched
} // namespace llvm
