#include "TwoPassScheduler.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "optsched"

using namespace llvm::opt_sched;

static ScheduleDAGInstrs *createTwoPassOptSched(MachineSchedContext *C) {
  return new ScheduleTwoPassOptSched(
      C, llvm::make_unique<GenericScheduler>(C));
}

// Register the machine scheduler.
static MachineSchedRegistry
    OptSchedMIRegistry("two-pass-optsched", "Use the Two Pass OptSched scheduler.",
                       createTwoPassOptSched);
					   
ScheduleTwoPassOptSched::ScheduleTwoPassOptSched(
    llvm::MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGOptSched(C, std::move(S)) {}

void ScheduleTwoPassOptSched::initSchedulers() {
  // Add passes

  // First
  SchedPasses.push_back(OptSchedMinRP);
  // Second
  SchedPasses.push_back(OptSchedBalanced);
}

// Record scheduling regions.
void ScheduleTwoPassOptSched::schedule() {
  Regions.push_back(std::make_pair(RegionBegin, RegionEnd));
  return;
}

// Execute scheduling passes.
// Partially copied GCNScheduleDAGMILive::finalizeSchedule
void ScheduleTwoPassOptSched::finalizeSchedule() {
  initSchedulers();

  for (const SchedPassStrategy &S : SchedPasses) {
    MachineBasicBlock *MBB = nullptr;
    // Reset
    RegionNumber = ~0u;

    for (auto &Region : Regions) {
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
      runSchedPass(S);
      Region = std::make_pair(RegionBegin, RegionEnd);
      exitRegion();
    }
    finishBlock();
  }
}

void ScheduleTwoPassOptSched::runSchedPass(SchedPassStrategy S) {
  switch (S) {
  case OptSchedMinRP:
    scheduleOptSchedMinRP();
    break;
  case OptSchedBalanced:
    scheduleOptSchedBalanced();
    break;
  }
}

void ScheduleTwoPassOptSched::scheduleOptSchedMinRP() {
  LLVM_DEBUG(dbgs() << "First pass through...\n");
  LatencyPrecision = LTP_UNITY;
  HeurSchedType = SCHED_LIST;
  ScheduleDAGOptSched::schedule();
}

void ScheduleTwoPassOptSched::scheduleOptSchedBalanced() {
  LLVM_DEBUG(dbgs() << "Second pass through...\n");
  secondPass = true;
  LatencyPrecision = LTP_ROUGH;
  // Force the input to the balanced scheduler to be the sequential order of the
  // (hopefully) good max occupancy schedule. We don’t want the list scheduler
  // to mangle the input because of latency or resource constraints.
  HeurSchedType = SCHED_SEQ;
  ScheduleDAGOptSched::schedule();
}

