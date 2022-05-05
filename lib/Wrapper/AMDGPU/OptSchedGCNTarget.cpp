//===- OptSchedGCNTarget.cpp - AMDGCN Target ------------------------------===//
//
// AMDGCN OptSched target.
//
//===----------------------------------------------------------------------===//
#include "OptSchedDDGWrapperGCN.h"
#include "SIMachineFunctionInfo.h"
#include "OptSchedGCNTarget.h"
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

#ifndef NDEBUG
static unsigned getOccupancyWeight(unsigned Occupancy) {
  if (Occupancy == 1)
    return 100;
  if (Occupancy == 2)
    return 300;
  if (Occupancy == 3)
    return 450;
  if (Occupancy == 4)
    return 583;
  if (Occupancy == 5)
    return 708;
  if (Occupancy == 6)
    return 828;
  if (Occupancy == 7)
    return 944;
  if (Occupancy == 8)
    return 1059;
  if (Occupancy == 9)
    return 1172;
  if (Occupancy == 10)
    return 1283;

  llvm_unreachable("Occupancy must be between 1 and 10");
}
#endif

static unsigned getAdjustedOccupancy(const GCNSubtarget *ST, unsigned VGPRCount,
                                     unsigned SGPRCount, unsigned MaxOccLDS) {
  unsigned MaxOccVGPR =
      ST->getOccupancyWithNumVGPRs(VGPRCount + GPRErrorMargin);
  unsigned MaxOccSGPR =
      ST->getOccupancyWithNumSGPRs(SGPRCount + GPRErrorMargin);
  return std::min(MaxOccLDS, std::min(MaxOccVGPR, MaxOccSGPR));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void OptSchedGCNTarget::dumpOccupancyInfo(const InstSchedule *Schedule) const {
  const InstCount *PRP;
  Schedule->GetPeakRegPressures(PRP);

  unsigned SGPR32Count = PRP[MM->GetRegTypeByName("SGPR32")] + GPRErrorMargin;
  auto MaxOccSGPR = ST->getOccupancyWithNumSGPRs(SGPR32Count);

  unsigned VGPR32Count = PRP[MM->GetRegTypeByName("VGPR32")] + GPRErrorMargin;
  auto MaxOccVGPR = ST->getOccupancyWithNumVGPRs(VGPR32Count);
  auto Occ = std::min(std::min(MaxOccSGPR, MaxOccVGPR), MaxOccLDS);

  dbgs() << "Estimated Max Occupancy After Scheduling: " << Occ << "\n"
         << "Weight: " << getOccupancyWeight(Occ) << "\n"
         << "Max Occ with LDS: " << MaxOccLDS << "\n"
         << "Max Occ with Num SGPRs: " << MaxOccSGPR << "\n"
         << "Max Occ with Num VGPRs: " << MaxOccVGPR << "\n";

  if (MFI->getMinAllowedOccupancy() < MFI->getOccupancy())
    dbgs() << "Occupancy is limited by perf hints:"
           << " MemoryBound=" << MFI->isMemoryBound()
           << " WaveLimiterHint=" << MFI->needsWaveLimiter() << "\n";
}
#endif

void OptSchedGCNTarget::initRegion(llvm::ScheduleDAGInstrs *DAG_,
                                   MachineModel *MM_) {
  DAG = static_cast<ScheduleDAGOptSched *>(DAG_);
  MF = &DAG->MF;
  MFI =
      const_cast<SIMachineFunctionInfo *>(MF->getInfo<SIMachineFunctionInfo>());
  MM = MM_;
  ST = &MF->getSubtarget<GCNSubtarget>();
  MaxOccLDS = ST->getOccupancyWithLocalMemSize(*MF);

  GCNDownwardRPTracker RPTracker(*DAG->getLIS());
  RPTracker.advance(DAG->begin(), DAG->end(), nullptr);
  const GCNRegPressure &P = RPTracker.moveMaxPressure();
  RegionStartingOccupancy =
      getAdjustedOccupancy(ST, P.getVGPRNum(ST->hasGFX90AInsts()), P.getSGPRNum(), MaxOccLDS);
  TargetOccupancy =
      shouldLimitWaves() ? MFI->getMinAllowedOccupancy() : MFI->getOccupancy();

  LLVM_DEBUG(dbgs() << "Region starting occupancy is "
                    << RegionStartingOccupancy << "\n"
                    << "Target occupancy is " << TargetOccupancy << "\n");
}

bool OptSchedGCNTarget::shouldLimitWaves() const {
  // FIXME: Consider machine model here as well.
  // FIXME: Return false because perf hints are not currently strong enough to
  // use as a hard cap. Consider 'OccupancyWeight' heuristic here instead.
  return false;
}

unsigned OptSchedGCNTarget::getOccupancyWithCost(const InstCount Cost) const {
  return TargetOccupancy - Cost;
}

void OptSchedGCNTarget::finalizeRegion(const InstSchedule *Schedule) {
  LLVM_DEBUG(dumpOccupancyInfo(Schedule));

  RegionEndingOccupancy = getOccupancyWithCost(Schedule->GetSpillCost());
  // If we decrease occupancy we may revert scheduling.
  unsigned RegionOccupancy =
      std::max(RegionStartingOccupancy, RegionEndingOccupancy);
  LLVM_DEBUG(if (RegionOccupancy < MFI->getOccupancy()) dbgs()
             << "Limiting occupancy to " << RegionEndingOccupancy
             << " waves.\n");
  MFI->limitOccupancy(RegionOccupancy);
}

InstCount OptSchedGCNTarget::getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const {
  // FIXME: It's bad to asssume that the reg types for SGPR32/VGPR32 are
  // fixed, but we avoid doing an expensive string compare here with
  // GetRegTypeByName since updating the cost happens so often. We should
  // replace OptSched register types completely with PSets to fix both issues.
  auto Occ =
      getAdjustedOccupancy(ST, PRP[OptSchedDDGWrapperGCN::VGPR32],
                           PRP[OptSchedDDGWrapperGCN::SGPR32], MaxOccLDS);
  // RP cost is the difference between the minimum allowed occupancy for the
  // function, and the current occupancy.
  return Occ >= TargetOccupancy ? 0 : TargetOccupancy - Occ;
}

bool OptSchedGCNTarget::shouldKeepSchedule() {
  if (RegionEndingOccupancy >= RegionStartingOccupancy ||
      RegionEndingOccupancy >= TargetOccupancy)
    return true;

  Logger::Info(
      "Reverting Scheduling because of a decrease in occupancy from %d to %d.",
      RegionStartingOccupancy, RegionEndingOccupancy);
  return false;
}

namespace llvm {
namespace opt_sched {

OptSchedTargetRegistry OptSchedGCNTargetRegistry("amdgcn",
                                                 createOptSchedGCNTarget);

OptSchedTargetRegistry OptSchedGCNHSATargetRegistry("amdgcn-amd-amdhsa",
                                                    createOptSchedGCNTarget);

} // namespace opt_sched
} // namespace llvm
