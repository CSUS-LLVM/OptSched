//===- OptSchedGCNTarget.cpp - AMDGCN Target ------------------------------===//
//
// AMDGCN OptSched target.
//
//===----------------------------------------------------------------------===//
#include "OptSchedDDGWrapperGCN.h"
#include "SIMachineFunctionInfo.h"
#include "Wrapper/OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/config.h"
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

// This is necessary because we cannot perfectly predict the number of registers
// of each type that will be allocated.
static const unsigned GPRErrorMargin = 0;
static const unsigned OCCUnlimited = 10;

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

namespace {

class OptSchedGCNTarget : public OptSchedTarget {
public:
  std::unique_ptr<OptSchedMachineModel>
  createMachineModel(const char *ConfigPath) override {
    return std::make_unique<OptSchedMachineModel>(ConfigPath);
  }

  std::unique_ptr<OptSchedDDGWrapperBase>
  createDDGWrapper(llvm::MachineSchedContext *Context, ScheduleDAGOptSched *DAG,
                   OptSchedMachineModel *MM, LATENCY_PRECISION LatencyPrecision,
                   const std::string &RegionID) override {
    return std::make_unique<OptSchedDDGWrapperGCN>(Context, DAG, MM,
                                                   LatencyPrecision, RegionID);
  }

  void initRegion(llvm::ScheduleDAGInstrs *DAG, MachineModel *MM_,
                  Config &OccFile) override;

  void finalizeRegion(const InstSchedule *Schedule) override;

  // Returns occupancy cost with number of VGPRs and SGPRs from PRP for
  // a partial or complete schedule.
  InstCount getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const override;

  void dumpOccupancyInfo(const InstSchedule *Schedule) const;

  // Revert scheduing if we decrease occupancy.
  bool shouldKeepSchedule() override;

  void SetOccupancyLimit(int OccupancyLimitParam) override {
    OccupancyLimit = OccupancyLimitParam;
  }
  void SetShouldLimitOcc(bool ShouldLimitOccParam) override {
    ShouldLimitOcc = ShouldLimitOccParam;
  }
  void SetOccLimitSource(OCC_LIMIT_TYPE LimitTypeParam) override {
    LimitType = LimitTypeParam;
  }

  int getOccupancyLimit(Config &OccFile) const;

private:
  const llvm::MachineFunction *MF;
  SIMachineFunctionInfo *MFI;
  ScheduleDAGOptSched *DAG;
  const GCNSubtarget *ST;

  unsigned RegionStartingOccupancy;
  unsigned RegionEndingOccupancy;
  unsigned TargetOccupancy;

  // Limiting occupancy has shown to greatly increase the performance of some
  // kernels
  int OccupancyLimit;
  bool ShouldLimitOcc;
  OCC_LIMIT_TYPE LimitType;

  // Max occupancy with local memory size;
  unsigned MaxOccLDS;

  // In RP only (max occupancy) scheduling mode we should try to find
  // a min-RP schedule without considering perf hints which suggest limiting
  // occupancy. Returns true if we should consider perf hints.
  bool shouldLimitWaves(llvm::SIMachineFunctionInfo *MFI) const;

  // Find occupancy with spill cost.
  unsigned getOccupancyWithCost(const InstCount Cost) const;
};

std::unique_ptr<OptSchedTarget> createOptSchedGCNTarget() {
  return std::make_unique<OptSchedGCNTarget>();
}

} // end anonymous namespace

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
                                   MachineModel *MM_, Config &OccFile) {
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
  RegionStartingOccupancy = getAdjustedOccupancy(
      ST, P.getVGPRNum(ST->hasGFX90AInsts()), P.getSGPRNum(), MaxOccLDS);

  TargetOccupancy =
      shouldLimitWaves(MFI) ? getOccupancyLimit(OccFile) : MFI->getOccupancy();

  // Do not attempt to hit a higher occupancy if we are limited by another
  // region
  if (TargetOccupancy > MFI->getOccupancy())
    TargetOccupancy = MFI->getOccupancy();

  Logger::Event("TargetOccupancy", "RegionStarting", RegionStartingOccupancy,
                "Target", TargetOccupancy);

  LLVM_DEBUG(dbgs() << "Region starting occupancy is "
                    << RegionStartingOccupancy << "\n"
                    << "Target occupancy is " << TargetOccupancy << "\n");
}

bool OptSchedGCNTarget::shouldLimitWaves(
    llvm::SIMachineFunctionInfo *MFI) const {
  // FIXME: Consider machine model here as well.
  // FIXME: Return false because perf hints are not currently strong enough to
  // use as a hard cap. Consider 'OccupancyWeight' heuristic here instead.
  // TODO(Jeff): Limiting occupancy has shown to have a huge impact on
  // performance. Good heuristics will likely be largely beneficial

  if (ShouldLimitOcc) {
    switch (LimitType) {
    case OLT_NONE:
      return false;
    case OLT_HEUR:
      return MFI->isMemoryBound() || MFI->needsWaveLimiter();
    case OLT_FILE:
      return true;
    }
  }

  return false;
}

int OptSchedGCNTarget::getOccupancyLimit(Config &OccFile) const {
  switch (LimitType) {
  case OLT_NONE:
    return OCCUnlimited;
  case OLT_VALUE:
    return OccupancyLimit;
  case OLT_HEUR:
    return MFI->isMemoryBound() || MFI->needsWaveLimiter() ? 4 : OCCUnlimited;
  case OLT_FILE:
    std::string functionName = MF->getFunction().getName().data();
    int limit = OccFile.GetInt(functionName, -1);
    int AMDHeur =
        MFI->isMemoryBound() || MFI->needsWaveLimiter() ? 4 : OCCUnlimited;
    if (limit != -1) {
      Logger::Event("OccupancyLimits", "File", limit, "AMDHeur", AMDHeur);
    }
    if (limit == -1) {
      limit = OCCUnlimited;
    }
    return limit;
  }
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

InstCount
OptSchedGCNTarget::getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const {
  // FIXME: It's bad to assume that the reg types for SGPR32/VGPR32 are
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
