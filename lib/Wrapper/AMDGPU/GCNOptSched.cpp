//===- GCNOptSched.cpp - AMDGCN Combinatorial scheudler -------------------===//
//
// Implements a combinatorial scheduling strategy for AMDGCN.
//
//===----------------------------------------------------------------------===//

#include "GCNOptSched.h"
#include "AMDGPUMacroFusion.h"
#include "GCNSchedStrategy.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/Support/Debug.h"
#include "AMDGPUExportClustering.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include <math.h> 
#include <algorithm>

#define DEBUG_TYPE "optsched"
#define MAX_POSSIBLE_OCCUPANCY 10
#define MAX_DISTANCE_FROM_OCCUPANCY_BOUNDARY 10000
#define RP_WEIGHT 100
#define ILP_WEIGHT 1
#define OCC_WEIGHT 800
#define LD_FACTOR 15
#define COST_THRESHOLD 7
// #define DEBUG_RESET_OCCUPANCY 1
#define DEBUG_TUNING_PARAMETER 1

using namespace llvm::opt_sched;

// FIXME: Temporary, eliminate
static cl::opt<bool>
    GCNLimitOccWithHints("gcn-limit-occ-with-hints",
                         cl::desc("Limit occpancy target using perf hints."),
                         cl::init(false), cl::Hidden);

static ScheduleDAGInstrs *createOptSchedGCN(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG = new ScheduleDAGOptSchedGCN(
      C, std::make_unique<GCNMaxOccupancySchedStrategy>(C));
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createAMDGPUMacroFusionDAGMutation());
  DAG->addMutation(createAMDGPUExportClusteringDAGMutation());
  return DAG;
}

// Register the machine scheduler.
static MachineSchedRegistry
    OptSchedGCNMIRegistry("gcn-optsched", "Use the GCN OptSched scheduler.",
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

static unsigned getVGPRMargin(unsigned VGPRs) {
  if (VGPRs <= 24) {
    return 24 - VGPRs;
  }
  if (VGPRs <= 28) {
    return 28 - VGPRs;
  }
  if (VGPRs <= 32) {
    return 32 - VGPRs;
  }
  if (VGPRs <= 36) {
    return 36 - VGPRs;
  }
  if (VGPRs <= 40) {
    return 40 - VGPRs;
  }
  if (VGPRs <= 48) {
    return 48 - VGPRs;
  }
  if (VGPRs <= 64) {
    return 64 - VGPRs;
  }
  if (VGPRs <= 84) {
    return 84 - VGPRs;
  }
  if (VGPRs <= 128) {
    return 128 - VGPRs;
  }
  return UINT_MAX;
}

ScheduleDAGOptSchedGCN::ScheduleDAGOptSchedGCN(
    llvm::MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGOptSched(C, std::move(S)) {
  SIMachineFunctionInfo *MFI;
  MFI =
      const_cast<SIMachineFunctionInfo *>(C->MF->getInfo<SIMachineFunctionInfo>());
  #ifdef DEBUG_RESET_OCCUPANCY
    printf("Occ before: %d\n", MFI->getOccupancy());
  #endif
  setInitialOccupancy(MFI->getCurrentOccupancy());
  MFI->resetInitialOccupancy(*C->MF);
  #ifdef DEBUG_RESET_OCCUPANCY
    printf("Occ after: %d\n", MFI->getOccupancy());
  #endif
  /*
  printf("Memory Cost: For function: %s\n", MF.getName());
  printf("InstCost = %d\n", MFI->getInstCost());
  printf("MemInstCost = %d\n", MFI->getMemInstCost());
  printf("IndirectAccessInstCost = %d\n", MFI->getIndirectAccessInstCost());
  printf("LargeStrideInstCost = %d\n", MFI->getLargeStrideInstCost());
  printf("End\n");
  */
}
void PrintTuningParameters()
{
  printf("TuningParameters\n");
  printf("RP_WEIGHT = %d\n", RP_WEIGHT);
  printf("ILP_WEIGHT = %d\n", ILP_WEIGHT);
  printf("OCC_WEIGHT = %d\n", OCC_WEIGHT);
  printf("LD_FACTOR = %d\n", LD_FACTOR);
  printf("COST_THRESHOLD = %d\n", COST_THRESHOLD);
  printf("END\n");
}
void ScheduleDAGOptSchedGCN::initSchedulers() {
  // Add passes

  // SchedPasses.push_back(GCNMaxOcc);

  // First
  SchedPasses.push_back(OptSchedMaxOcc);
  // Second
  SchedPasses.push_back(OptSchedBalanced);
}

// Execute scheduling passes.
// Partially copied GCNScheduleDAGMILive::finalizeSchedule
void ScheduleDAGOptSchedGCN::finalizeSchedule() {

  if (TwoPassEnabled && OptSchedEnabled) {
    #ifdef DEBUG_TUNING_PARAMETER
      PrintTuningParameters();
    #endif
    initSchedulers();
    int numOccupancies = 0;
    LLVM_DEBUG(dbgs() << "Starting two pass scheduling approach\n");
    TwoPassSchedulingStarted = true;
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
        LLVM_DEBUG(getRealRegionPressure(RegionBegin, RegionEnd, LIS, "Before"));
        runSchedPass(S);
        LLVM_DEBUG(getRealRegionPressure(RegionBegin, RegionEnd, LIS, "After"));
        Region = std::make_pair(RegionBegin, RegionEnd);
        exitRegion();
        if (S == OptSchedBalanced) {
          auto &SchedEvalForMBB = SchedEvals[RegionNumber];
          numOccupancies = std::max(SchedEvalForMBB.getNumOccupancies(), numOccupancies);

          int ILPWeight = pow(LD_FACTOR, C->MLI->getLoopDepth(MBB));
          SchedEvalForMBB.setILPWeight(ILPWeight);
        }
      }
      finishBlock();
    }
    // Reset
    RegionNumber = ~0u;
    int regionNum = 0;
    // some kernels can contain 0 scheduling regions, need to check it's not empty
    if (!SchedEvals.empty()) {
      Logger::Info("Starting Reverting\n");
      auto &firstSchedEval = SchedEvals[0];

      int64_t ILPSum[numOccupancies];
      for (int i = 0; i < numOccupancies; i++)
        ILPSum[i] = 0;
      unsigned OccTracker[numOccupancies];
      unsigned RegisterMarginTracker[numOccupancies];
      std::fill(OccTracker, OccTracker + numOccupancies, MAX_POSSIBLE_OCCUPANCY);
      std::fill(RegisterMarginTracker, RegisterMarginTracker + numOccupancies, MAX_DISTANCE_FROM_OCCUPANCY_BOUNDARY);

      for (auto &Region : Regions) {
        auto &SchedEval = SchedEvals[regionNum];
        auto ILPWeight = SchedEval.getILPWeight();
        for (int i = 0; i < numOccupancies; i++) {
          ILPSum[i] += SchedEval.getILPAtIndex(i) * ILPWeight;
          OccTracker[i] = std::min(OccTracker[i], SchedEval.getOccAtIndex(i));
          RegisterMarginTracker[i] = std::min(RegisterMarginTracker[i], getVGPRMargin(SchedEval.getVGPRsAtIndex(i)));
          printf("Region Num: %d Choice %d, occ cost: %d, ilp: %d, ILP weight: %d\n", regionNum, i, OccTracker[i], SchedEval.getILPAtIndex(i), ILPWeight);
        }
        regionNum++;
      }
      for (int i = 0; i < numOccupancies; i++) {
        if (i > 0 && ILPSum[i] > ILPSum[i-1])
          printf("ILP Regression\n");
      }

      int schedIndex = 0;
      int weightedCost[numOccupancies + 1];

      auto calcOccCost = [ ] (unsigned Occ, unsigned VGPRMargin) {
        #ifdef calOcc
          printf("OCC_WEIGHT/Occ + RP_WEIGHT/std::pow(2, VGPRMargin)")
        #define calOcc 1
        #endif
	      return OCC_WEIGHT/Occ + RP_WEIGHT/std::pow(2, VGPRMargin);
      };

      int occCost = calcOccCost(OccTracker[0], RegisterMarginTracker[0]);
      int ilpCost = ILP_WEIGHT * ILPSum[0];
      weightedCost[0] = ilpCost + occCost;
      printf("Choice 0, Weighted Cost: %d, occ: %d, occ cost: %d, ilp cost: %d, margin: %d\n", weightedCost[0], OccTracker[0], occCost, ilpCost, RegisterMarginTracker[0]);
      int minCost = weightedCost[0];
      int minIndex = 0;
      for (int i = 1; i < numOccupancies; i++) {
        occCost = calcOccCost(OccTracker[i], RegisterMarginTracker[i]);
        ilpCost = ILP_WEIGHT * ILPSum[i];
        weightedCost[i] = ilpCost + occCost;
        weightedCost[i] += (int) ((COST_THRESHOLD / 100.0) * weightedCost[i]);
        if (weightedCost[i] < minCost) {
          minIndex = i;
          minCost = weightedCost[i];
        }
        printf("Choice %d, Weighted Cost: %d, occ: %d, occ cost: %d, ilp cost: %d, margin: %d\n", i, weightedCost[i], OccTracker[i], occCost, ilpCost, RegisterMarginTracker[i]);
      }
			// minIndex = 3;
      //printf("Min Index is: %d, occupancy is: %d\n", minIndex, OccTracker[minIndex]);
      regionNum = 0;

      // temp to test only use AMD Schedule
      // minIndex = 1;
      // printf("Min Index overridden to: %d\n", minIndex);

      // set up the block beginning and ending in order to revert
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

        auto &SchedEval = SchedEvals[regionNum];
        //Override
        //AMD
        //SchedEval.revertScheduling(1);
        SchedEval.revertScheduling(minIndex);
        regionNum++;

        // Skip empty scheduling regions (0 or 1 schedulable instructions).
        if (begin() == end() || begin() == std::prev(end())) {
          exitRegion();
          continue;
        }
        LLVM_DEBUG(
            getRealRegionPressure(RegionBegin, RegionEnd, LIS, "Before"));
        LLVM_DEBUG(getRealRegionPressure(RegionBegin, RegionEnd, LIS, "After"));
        Region = std::make_pair(RegionBegin, RegionEnd);
        exitRegion();
      }
      finishBlock();
      Logger::Info("Finish Reverting");
    } else {
      printf("No schedEvals?\n");
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
