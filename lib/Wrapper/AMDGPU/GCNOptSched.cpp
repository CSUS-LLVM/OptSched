//===- GCNOptSched.cpp - AMDGCN Combinatorial scheudler -------------------===//
//
// Implements a combinatorial scheduling strategy for AMDGCN.
//
//===----------------------------------------------------------------------===//

#include "GCNOptSched.h"
#include "AMDGPUMacroFusion.h"
#include "GCNSchedStrategy.h"
#include "SIMachineFunctionInfo.h"
#include "OptSchedGCNTarget.h"
//#include "llvm/CodeGen/OptSequential.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "optsched"

using namespace llvm::opt_sched;

// FIXME: Temporary, eliminate
static cl::opt<bool>
    GCNLimitOccWithHints("gcn-limit-occ-with-hints",
                         cl::desc("Limit occpancy target using perf hints."),
                         cl::init(false), cl::Hidden);

static ScheduleDAGInstrs *createOptSchedGCN(MachineSchedContext *C) {
  return new ScheduleDAGOptSchedGCN(
      C, llvm::make_unique<GCNMaxOccupancySchedStrategy>(C));
}

// Register the machine scheduler.
static MachineSchedRegistry
    OptSchedMIRegistry("gcn-optsched", "Use the GCN OptSched scheduler.",
                       createOptSchedGCN);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static void getRealRegionPressure(MachineBasicBlock::const_iterator Begin,
                                  MachineBasicBlock::const_iterator End,
                                  const LiveIntervals *LIS, StringRef Label) {
  GCNDownwardRPTracker RP(*LIS);
  RP.advance(Begin, End, nullptr);
  dbgs() << "Dumping real RP " << Label << "\n";
  RP.moveMaxPressure().dump();
}
#endif

ScheduleDAGOptSchedGCN::ScheduleDAGOptSchedGCN(
    llvm::MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGOptSched(C, std::move(S)) {}

void ScheduleDAGOptSchedGCN::initSchedulers() {
  // Add DAG mutations that apply to both GCN and OptSched DAG's

  addMutation(createLoadClusterDAGMutation(TII, TRI));
  addMutation(createStoreClusterDAGMutation(TII, TRI));
  // addMutation(createAMDGPUMacroFusionDAGMutation());

  // Add passes

  // SchedPasses.push_back(GCNMaxOcc);

  // First
  SchedPasses.push_back(OptSchedMaxOcc);
  // Second
  SchedPasses.push_back(OptSchedBalanced);
  SchedPasses.push_back(OptSchedReschedule);
}   

// Execute scheduling passes.
// Partially copied GCNScheduleDAGMILive::finalizeSchedule
void ScheduleDAGOptSchedGCN::finalizeSchedule() {
  if (TwoPassEnabled && OptSchedEnabled) {
    initSchedulers();
    RescheduleRegions.resize(Regions.size());
    RescheduleRegions.set();

    LLVM_DEBUG(dbgs() << "Starting two pass scheduling approach\n");
    TwoPassSchedulingStarted = true;
    for (const SchedPassStrategy &S : SchedPasses) {
      MachineBasicBlock *MBB = nullptr;
      // Reset
      RegionIdx = 0;
      if (S == OptSchedReschedule) {
        if (RescheduleRegions.none()) {
	  dbgs() << "No regions to reschedule.\n";
	  continue;
	} else {
          auto GCNOST = static_cast<OptSchedGCNTarget *>(OST.get());
          unsigned TargetOccupancy = GCNOST->getTargetOcc();
          if (TargetOccupancy == 1u) {
            dbgs() << "Cannot lower occupancy to below 1.\n";
	    continue;
	  }

          dbgs() << "Beginning rescheduling of regions.\n";
	  unsigned NewTarget = TargetOccupancy - 1u;
	  dbgs() << "Decreasing current target occupancy " << TargetOccupancy
                 << " to new target " << NewTarget << '\n';
	  GCNOST->limitOccupancy(NewTarget);
	}
      }

      for (auto &Region : Regions) {
	/*if (S == OptSchedReschedule && !RescheduleRegions[RegionIdx]) {
	  dbgs() << "Region " << RegionIdx << " does not need to be rescheduled.\n";
	  ++RegionIdx;
	  continue;
	}*/

        RegionBegin = Region.first;
        RegionEnd = Region.second;

        if (RegionBegin->getParent() != MBB) {
          if (MBB)
            finishBlock();
          MBB = RegionBegin->getParent();
          startBlock(MBB);
        }
        unsigned NumRegionInstrs = std::distance(begin(), end());
        enterRegion(MBB, begin(), end(), NumRegionInstrs);

        // Skip empty scheduling regions (0 or 1 schedulable instructions).
        if (begin() == end() || begin() == std::prev(end())) {
          exitRegion();
          continue;
        }
        LLVM_DEBUG(getRealRegionPressure(RegionBegin, RegionEnd, LIS, "Before"));
        runSchedPass(S);
        LLVM_DEBUG(getRealRegionPressure(RegionBegin, RegionEnd, LIS, "After"));
        Region = std::make_pair(RegionBegin, RegionEnd);
        exitRegion();
        ++RegionIdx;
      }
      finishBlock();
    }
  }

  ScheduleDAGMILive::finalizeSchedule();

  LLVM_DEBUG(if (isSimRegAllocEnabled()) {
    dbgs() << "*************************************\n";
    dbgs() << "Function: " << MF.getName()
           << "\nTotal Simulated Spills: " << SimulatedSpills << "\n";
    dbgs() << "*************************************\n";
  });
}

void ScheduleDAGOptSchedGCN::runSchedPass(SchedPassStrategy S) {
  RescheduleRegions[RegionIdx] = false;
  switch (S) {
  case GCNMaxOcc:
    scheduleGCNMaxOcc();
    break;
  case OptSchedMaxOcc:
    scheduleOptSchedMaxOcc();
    break;
  case OptSchedBalanced:
    scheduleOptSchedBalanced();
    break;
  case OptSchedReschedule:
    scheduleOptSchedReschedule();
    break;
  }
}

void ScheduleDAGOptSchedGCN::scheduleGCNMaxOcc() {
  auto &S = (GCNMaxOccupancySchedStrategy &)*SchedImpl;
  if (GCNLimitOccWithHints) {
    const auto &MFI = *MF.getInfo<SIMachineFunctionInfo>();
    S.setTargetOccupancy(MFI.getMinAllowedOccupancy());
  }

  ScheduleDAGMILive::schedule();
}

void ScheduleDAGOptSchedGCN::scheduleOptSchedMaxOcc() {
  ScheduleDAGOptSched::scheduleOptSchedMinRP();
}

void ScheduleDAGOptSchedGCN::scheduleOptSchedBalanced() {
  ScheduleDAGOptSched::scheduleOptSchedBalanced();
}

void ScheduleDAGOptSchedGCN::scheduleOptSchedReschedule() {
  IsThirdPass = true;
  ScheduleDAGOptSched::scheduleOptSchedBalanced();
  Logger::Info("End of third pass through\n");
}

