//===- OptSchedGCNTarget.cpp - AMDGCN Target ------------------------------===//
//
// AMDGCN OptSched target.
//
//===----------------------------------------------------------------------===//
#include "OptSchedDDGWrapperGCN.h"
#include "SIMachineFunctionInfo.h"
#include "Wrapper/OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include <algorithm>
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

  void initRegion(llvm::ScheduleDAGInstrs *DAG, MachineModel *MM_) override;

  void finalizeRegion(const InstSchedule *Schedule) override;

  // Returns occupancy cost with number of VGPRs and SGPRs from PRP for
  // a partial or complete schedule.
  InstCount getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const override;

  void dumpOccupancyInfo(const InstSchedule *Schedule) const;

private:
  const llvm::MachineFunction *MF;
  SIMachineFunctionInfo *MFI;
  ScheduleDAGOptSched *DAG;

  // Max occupancy with local memory size;
  unsigned MaxOccLDS;

  // In RP only (max occupancy) scheduling mode we should try to find
  // a min-RP schedule without considering perf hints which suggest limiting
  // occupancy. Returns true if we should consider perf hints.
  bool shouldLimitWaves() const;

  // Find occupancy with spill cost.
  unsigned getOccupancyWithCost(const InstCount Cost) const;
};

std::unique_ptr<OptSchedTarget> createOptSchedGCNTarget() {
  return llvm::make_unique<OptSchedGCNTarget>();
}

} // end anonymous namespace

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void OptSchedGCNTarget::dumpOccupancyInfo(const InstSchedule *Schedule) const {
  auto &ST = MF->getSubtarget<GCNSubtarget>();

  const InstCount *PRP;
  Schedule->GetPeakRegPressures(PRP);
  unsigned SGPR32Count = PRP[MM->GetRegTypeByName("SGPR32")];
  auto MaxOccSGPR = ST.getOccupancyWithNumSGPRs(SGPR32Count);

  unsigned VGPR32Count = PRP[MM->GetRegTypeByName("VGPR32")];
  auto MaxOccVGPR = ST.getOccupancyWithNumVGPRs(VGPR32Count);

  dbgs() << "Estimated Max Occupancy After Scheduling: "
         << std::min(std::min(MaxOccSGPR, MaxOccVGPR), MaxOccLDS) << "\n"
         << "Max Occ with LDS: " << MaxOccLDS << "\n"
         << "Max Occ with Num SGPRs: " << MaxOccSGPR << "\n"
         << "Max Occ with Num VGPRs: " << MaxOccVGPR << "\n";

  if (MFI->getMinAllowedOccupancy() < MFI->getOccupancy())
    dbgs() << "Occupancy is limited by perf hints:"
           << " MemoryBound=" << MFI->isMemoryBound()
           << " WaveLimiterHint=" << MFI->needsWaveLimiter()
           << "\n";
}
#endif

void OptSchedGCNTarget::initRegion(llvm::ScheduleDAGInstrs *DAG_,
                                   MachineModel *MM_) {
  DAG = static_cast<ScheduleDAGOptSched *>(DAG_);
  MF = &DAG->MF;
  MFI =
      const_cast<SIMachineFunctionInfo *>(MF->getInfo<SIMachineFunctionInfo>());
  MM = MM_;

  auto &ST = MF->getSubtarget<GCNSubtarget>();
  MaxOccLDS = ST.getOccupancyWithLocalMemSize(*MF);
}

bool OptSchedGCNTarget::shouldLimitWaves() const {
  // FIXME: Consider machine model here as well.
  //return DAG->getLatencyType() != LTP_UNITY;
  return false;
}

unsigned OptSchedGCNTarget::getOccupancyWithCost(const InstCount Cost) const {
  if (shouldLimitWaves())
    return MFI->getMinAllowedOccupancy() - Cost;

  return 10 - Cost;
}

void OptSchedGCNTarget::finalizeRegion(const InstSchedule *Schedule) {
  LLVM_DEBUG(dumpOccupancyInfo(Schedule));

  unsigned WavesAfter = getOccupancyWithCost(Schedule->GetSpillCost());
  if (WavesAfter < MFI->getOccupancy()) {
    LLVM_DEBUG(dbgs() << "Limiting occupancy to " << WavesAfter << " waves.\n");

    MFI->limitOccupancy(WavesAfter);
  }
}

InstCount
OptSchedGCNTarget::getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const {
  // FIXME: It's bad to asssume that the reg types for SGPR32/VGPR32 are
  // fixed, but we avoid doing an expensive string compare here with
  // GetRegTypeByName since updating the cost happens so often. We should
  // replace OptSched register types completely with PSets to fix both issues.
  const auto &ST = MF->getSubtarget<GCNSubtarget>();

  const unsigned ErrorMargin = 3;
  unsigned SGPR32Count = PRP[OptSchedDDGWrapperGCN::SGPR32] + ErrorMargin;
  auto MaxOccSGPR = ST.getOccupancyWithNumSGPRs(SGPR32Count);

  unsigned VGPR32Count = PRP[OptSchedDDGWrapperGCN::VGPR32] + ErrorMargin;
  auto MaxOccVGPR = ST.getOccupancyWithNumVGPRs(VGPR32Count);

  auto Occ = std::min(std::min(MaxOccSGPR, MaxOccVGPR), MaxOccLDS);
  auto MinOcc = shouldLimitWaves() ? MFI->getMinAllowedOccupancy() : MFI->getOccupancy();
  // RP cost is the difference between the minimum allowed occupancy for the
  // function and the current occupancy.
  return Occ >= MinOcc ? 0 : MinOcc - Occ;
}

namespace llvm {
namespace opt_sched {

OptSchedTargetRegistry OptSchedGCNTargetRegistry("amdgcn",
                                                 createOptSchedGCNTarget);

} // namespace opt_sched
} // namespace llvm
