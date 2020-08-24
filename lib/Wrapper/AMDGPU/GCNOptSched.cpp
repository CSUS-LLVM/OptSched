//===- GCNOptSched.cpp - AMDGCN Combinatorial scheudler -------------------===//
//
// Implements a combinatorial scheduling strategy for AMDGCN.
//
//===----------------------------------------------------------------------===//

#include "GCNOptSched.h"
#include "AMDGPUMacroFusion.h"
#include "GCNSchedStrategy.h"
#include "OptSchedGCNTarget.h"
#include "SIMachineFunctionInfo.h"
//#include "llvm/CodeGen/OptSequential.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <string>

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
    : ScheduleDAGOptSched(C, std::move(S)) {
  MinOcc = getMinOcc();
}

unsigned ScheduleDAGOptSchedGCN::getMinOcc() {
  SchedulerOptions &schedIni = SchedulerOptions::getInstance();
  int MinOcc = schedIni.GetInt("MIN_OCCUPANCY_FOR_RESCHEDULE");
  if (MinOcc <= 10 && MinOcc >= 1)
    return MinOcc;

  llvm::report_fatal_error(
      "Unrecognized option for MIN_OCCUPANCY_FOR_RESCHEDULE setting: %d" +
      std::to_string(MinOcc), false);
}

int ScheduleDAGOptSchedGCN::getMinILPImprovement() {
  SchedulerOptions &schedIni = SchedulerOptions::getInstance();
  int MinIlpImprovement = schedIni.GetInt("MIN_ILP_IMPROVEMENT");
  if (MinIlpImprovement <= 100 && MinIlpImprovement >= 0)
    return MinIlpImprovement;

  llvm::report_fatal_error(
      "Unrecognized option for MIN_OCCUPANCY_FOR_RESCHEDULE setting: %d" +
      std::to_string(MinIlpImprovement), false);
}

void ScheduleDAGOptSchedGCN::initSchedulers() {
  // Add DAG mutations that apply to both GCN and OptSched DAG's

  addMutation(createLoadClusterDAGMutation(TII, TRI));
  addMutation(createStoreClusterDAGMutation(TII, TRI));
  // addMutation(createAMDGPUMacroFusionDAGMutation());

  // Add passes

  // SchedPasses.push_back(GCNMaxOcc);

  // First
  SchedPasses.push_back(OptSchedMaxOcc);
  // Second ILP passes
  SchedPasses.push_back(OptSchedBalanced);
  SchedPasses.push_back(OptSchedLowerOccAnalysis);
  SchedPasses.push_back(OptSchedCommitLowerOcc);
}

// Execute scheduling passes.
// Partially copied GCNScheduleDAGMILive::finalizeSchedule
void ScheduleDAGOptSchedGCN::finalizeSchedule() {
  if (TwoPassEnabled && OptSchedEnabled) {
    initSchedulers();
    RescheduleRegions.resize(Regions.size());
    ILPAnalysis.resize(Regions.size());
    CostAnalysis.resize(Regions.size());
    LowerOccScheds.resize(Regions.size());
    RescheduleRegions.set();

    LLVM_DEBUG(dbgs() << "Starting two pass scheduling approach\n");
    TwoPassSchedulingStarted = true;
    for (const SchedPassStrategy &S : SchedPasses) {
      MachineBasicBlock *MBB = nullptr;
      // Reset
      RegionIdx = 0;

      if (S == OptSchedLowerOccAnalysis) {
        if (RescheduleRegions.none())
          break;
        else {
          auto GCNOST = static_cast<OptSchedGCNTarget *>(OST.get());
          unsigned TargetOccupancy = GCNOST->getTargetOcc();
          if (TargetOccupancy <= MinOcc)
            break;

          unsigned NewTarget = TargetOccupancy - 1u;
          dbgs() << "Decreasing current target occupancy " << TargetOccupancy
                 << " to new target " << NewTarget << '\n';
          GCNOST->limitOccupancy(NewTarget);
        }
      }

      if (S == OptSchedCommitLowerOcc) {
        if (!shouldCommitLowerOccSched()) {
          dbgs()
              << "Lower occupancy schedule did not meet minimum improvement.\n";
          break;
        }
        dbgs() << "Lower occupancy met minimum improvement requirement!\n";
      }

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
        LLVM_DEBUG(
            getRealRegionPressure(RegionBegin, RegionEnd, LIS, "Before"));
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
}

void ScheduleDAGOptSchedGCN::runSchedPass(SchedPassStrategy S) {
  RescheduleRegions[RegionIdx] = false;
  switch (S) {
  case GCNMaxOcc:
    scheduleGCNMaxOcc();
    break;
  case OptSchedMaxOcc:
    scheduleOptSchedMaxOcc();
    Logger::Info("End of first pass through");
    break;
  case OptSchedBalanced:
    scheduleOptSchedBalanced();
    Logger::Info("End of second pass through");
    break;
  case OptSchedLowerOccAnalysis:
    scheduleOptSchedLowerOccAnalysis();
    Logger::Info("End of third pass through");
    break;
  case OptSchedCommitLowerOcc:
    scheduleCommitLowerOcc();
    Logger::Info("End of fourth pass through");
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

void ScheduleDAGOptSchedGCN::scheduleOptSchedLowerOccAnalysis() {
  IsThirdPass = true;
  ScheduleDAGOptSched::scheduleOptSchedBalanced();
  IsThirdPass = false;
}

void ScheduleDAGOptSchedGCN::scheduleCommitLowerOcc() {
  IsFourthPass = true;
  ScheduleDAGOptSched::scheduleOptSchedBalanced();
  IsFourthPass = false;
}

bool ScheduleDAGOptSchedGCN::shouldCommitLowerOccSched() {
  // First analyze ILP improvements
  int FirstPassLengthSum = 0;
  int SecondPassLengthSum = 0;
  int MinILPImprovement = getMinILPImprovement();
  for (std::pair<int, int> &RegionLength : ILPAnalysis) {
    FirstPassLengthSum += RegionLength.first;
    SecondPassLengthSum += RegionLength.second;
  }
  double FirstPassAverageLength = (double)FirstPassLengthSum / Regions.size();
  double SecondPassAverageLength = (double)SecondPassLengthSum / Regions.size();
  double ILPImprovement = ((FirstPassAverageLength - SecondPassAverageLength) /
                           FirstPassAverageLength) *
                          100.0;
  dbgs() << "ILPImprovement from second ILP pass is " << ILPImprovement
         << ", min improvement is: " << MinILPImprovement << '\n';
  if (ILPImprovement - MinILPImprovement >= 0)
    return true;

  return false;
}
