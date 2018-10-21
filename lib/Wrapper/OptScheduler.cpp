//===- OptScheduler.cpp - Implement the opt scheduler -===//
//
// Integrates an alternative scheduler into LLVM which
// implements a branch and bound scheduling algorithm.
//
//===-------------------------------------------------===//
#include "OptScheduler.h"
#include "OptSchedDagWrapper.h"
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/utilities.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>

#define DEBUG_TYPE "optsched"

using namespace opt_sched;

// hack to print spills
bool OPTSCHED_gPrintSpills;

// Create OptSched ScheduleDAG.
static llvm::ScheduleDAGInstrs *createOptSched(llvm::MachineSchedContext *C) {
  return new opt_sched::ScheduleDAGOptSched(C);
}

// Register the machine scheduler.
static llvm::MachineSchedRegistry
    OptSchedRegistry("optsched", "Use the OptSched scheduler.", createOptSched);

// Command line options for opt-sched.
static llvm::cl::opt<std::string> OptSchedCfg(
    "optsched-cfg", llvm::cl::Hidden,
    llvm::cl::desc(
        "Path to the directory containing configuration files for opt-sched."),
    llvm::cl::init("~/.optsched-cfg/"));

static llvm::cl::opt<std::string> OptSchedCfgS(
    "optsched-cfg-sched", llvm::cl::Hidden,
    llvm::cl::desc(
        "Path to the scheduler options configuration file for opt-sched."));

static llvm::cl::opt<std::string> OptSchedCfgHF(
    "optsched-cfg-hotfuncs", llvm::cl::Hidden,
    llvm::cl::desc(
        "Path to the list of hot functions to schedule using opt-sched."));

static llvm::cl::opt<std::string> OptSchedCfgMM(
    "optsched-cfg-machine-model", llvm::cl::Hidden,
    llvm::cl::desc(
        "Path to the machine model specification file for opt-sched."));

namespace {

// If this iterator is a debug value, increment until reaching the End or a
// non-debug instruction. static function copied from
// llvm/CodeGen/MachineScheduler.cpp
llvm::MachineBasicBlock::iterator
nextIfDebug(llvm::MachineBasicBlock::iterator I,
            llvm::MachineBasicBlock::const_iterator End) {
  for (; I != End; ++I) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

} // end anonymous namespace

ScheduleDAGOptSched::ScheduleDAGOptSched(llvm::MachineSchedContext *C)
    : llvm::ScheduleDAGMILive(C, llvm::make_unique<llvm::GenericScheduler>(C)),
      context(C), model(nullptr), totalSimulatedSpills(0) {

  // Find the native paths to the scheduler configuration files.
  getRealCfgPaths();

  // Setup config object
  Config &schedIni = SchedulerOptions::getInstance();
  // load OptSched ini file
  schedIni.Load(PathCfgS.c_str());

  // load hot functions ini file
  hotFunctions.Load(PathCfgHF.c_str());

  // load machine model file
  model = llvm::make_unique<LLVMMachineModel>(PathCfgMM.c_str());

  // Convert machine model
  model->convertMachineModel(static_cast<llvm::ScheduleDAGInstrs &>(*this),
                             RegClassInfo);

  // Load config files for the OptScheduler
  loadOptSchedConfig();
}

void ScheduleDAGOptSched::SetupLLVMDag() {
  // build DAG
  // Initialize the register pressure tracker used by buildSchedGraph.
  RPTracker.init(&MF, RegClassInfo, LIS, BB, LiveRegionEnd,
                 ShouldTrackLaneMasks, /*TrackUntiedDefs=*/true);

  // Account for liveness generate by the region boundary.
  if (LiveRegionEnd != RegionEnd)
    RPTracker.recede();

  // Build the DAG, and compute current register pressure.
  buildSchedGraph(AA, &RPTracker, &SUPressureDiffs, LIS, ShouldTrackLaneMasks);

  // Initialize top/bottom trackers after computing region pressure.
  initRegPressure();
}

// schedule called for each basic block
void ScheduleDAGOptSched::schedule() {
  // (Chris) Increment the region number here to get unique dag IDs
  // per scheduling region within a machine function.
  ++regionNum;

  // (Chris): This option in the sched.ini file will override USE_OPT_SCHED. It
  // will only apply B&B if the region name belongs in the list of specified
  // regions. Region names are of the form:
  //   funcName:regionNum
  // No leading zeroes in regionNum, and no whitespace.
  // Get config options.
  Config &schedIni = SchedulerOptions::getInstance();
  const bool scheduleSpecificRegions =
      schedIni.GetBool("SCHEDULE_SPECIFIC_REGIONS");
  if (scheduleSpecificRegions) {
    const std::list<std::string> regionList =
        schedIni.GetStringList("REGIONS_TO_SCHEDULE");
    const std::string regionName = context->MF->getFunction().getName().data() +
                                   std::string(":") + std::to_string(regionNum);
    optSchedEnabled = std::find(std::begin(regionList), std::end(regionList),
                                regionName) != std::end(regionList);
  }

#ifdef IS_DEBUG_PEAK_PRESSURE
  SetupLLVMDag();
  Logger::Info("RP before scheduling");
  RPTracker.dump();
#endif

  if (!optSchedEnabled) {
#ifdef IS_DEBUG
    Logger::Info("********** LLVM Scheduling **********\n");
#endif

    ScheduleDAGMILive::schedule();

#ifdef IS_DEBUG_PEAK_PRESSURE
    SetupLLVMDag();
    Logger::Info("RP after llvm scheduling");
    RPTracker.dump();
#endif
    return;
  }

  // Use LLVM's heuristic schedule as input to the B&B scheduler.
  if (llvmScheduling) {

    ScheduleDAGMILive::schedule();

    originalDAG = SUnits;
    // The schedule generated by LLVM
    ISOSchedule.resize(SUnits.size());
    // A vector index by NodeNum. The value
    // is the order in LLVM's schedule.
    std::vector<int> nodeNumMap(SUnits.size());

    // Find the schedule generated by LLVM which is stored in MMB.
    int counter = 0;
    for (llvm::MachineBasicBlock::instr_iterator I = BB->instr_begin(),
                                                 E = BB->instr_end();
         I != E; ++I) {

      llvm::MachineInstr &instr = *I;
      llvm::SUnit *su = getSUnit(&instr);

      if (su != NULL && !su->isBoundaryNode()) {
        int nodeNum = su->NodeNum;
#ifdef IS_DEBUG_ISO
        Logger::Info("Node num %d", nodeNum);
#endif
        ISOSchedule[counter] = nodeNum;
        nodeNumMap[nodeNum] = counter;
        counter++;
      }
    }

    // Update SUnits with the discovered schedule. Re-number SUnits to be in
    // sequential order.
    for (size_t i = 0; i < SUnits.size(); i++) {
      // continue if the SUnit is already in the correct place
      unsigned newNum = ISOSchedule[i];
      unsigned curNum = SUnits[i].NodeNum;
      if (curNum == newNum) {
        SUnits[i].NodeNum = i;
        continue;
      }

      for (size_t j = i + 1; j < SUnits.size(); j++) {
        if (SUnits[j].NodeNum == newNum) {
#ifdef IS_DEBUG_ISO
          Logger::Info("Swapping %d with %d for ISO", SUnits[j].NodeNum,
                       SUnits[i].NodeNum);
#endif
          std::swap(SUnits[j], SUnits[i]);
          SUnits[i].NodeNum = i;
          break;
        }
      }
    }

    // Update edges.
    for (size_t i = 0; i < SUnits.size(); i++) {
      llvm::SUnit *unit = &SUnits[i];
      // Update successor NodeNums.
      for (llvm::SDep &dep : unit->Succs) {
        llvm::SUnit *succ = dep.getSUnit();
        // check if the successor is a boundary node
        if (succ->isBoundaryNode())
          continue;
        dep.setSUnit(&SUnits[nodeNumMap[succ->NodeNum]]);
      }

      // Update predecessor NodeNums.
      for (llvm::SDep &dep : unit->Preds) {
        llvm::SUnit *pred = dep.getSUnit();
        // check if the predecessor is a boundary node
        if (pred->isBoundaryNode())
          continue;
        dep.setSUnit(&SUnits[nodeNumMap[pred->NodeNum]]);
      }
    }
  }

#ifdef IS_DEBUG
  Logger::Info("********** Opt Scheduling **********");
#endif
  // Build LLVM DAG
  SetupLLVMDag();

  // Apply llvm DAG post processing.
  if (enableMutations) {
    postprocessDAG();
  }

  // Ignore empty DAGs
  if (SUnits.empty())
    return;

  // convert dag
  LLVMDataDepGraph dag(context, this, model.get(), latencyPrecision, BB,
                       graphTransTypes, RPTracker.getPressure().MaxSetPressure,
                       treatOrderDepsAsDataDeps, maxDagSizeForLatencyPrecision,
                       regionNum);
  // create region
  SchedRegion *region = new BBWithSpill(
      model.get(), &dag, 0, histTableHashBits, lowerBoundAlgorithm,
      heuristicPriorities, enumPriorities, verifySchedule, prune,
      schedForRPOnly, enumerateStalls, spillCostFactor, spillCostFunction,
      checkSpillCostSum, checkConflicts, fixLiveIn, fixLiveOut, maxSpillCost);

  // count defs, add defs and uses
  region->BuildFromFile();

  // Schedule
  bool isEasy;
  InstCount normBestCost = 0;
  InstCount bestSchedLngth = 0;
  InstCount normHurstcCost = 0;
  InstCount hurstcSchedLngth = 0;
  InstSchedule *sched = NULL;
  FUNC_RESULT rslt;

  if (isTimeoutPerInstruction) {
    // Re-calculate timeout values if timeout setting is per instruction
    // becuase we want a unique value per DAG size
    regionTimeout = schedIni.GetInt("REGION_TIMEOUT") * dag.GetInstCnt();
    lengthTimeout = schedIni.GetInt("LENGTH_TIMEOUT") * dag.GetInstCnt();
  }

  // Setup time before scheduling
  Utilities::startTime = std::chrono::high_resolution_clock::now();

  if (dag.GetInstCnt() < minDagSize || dag.GetInstCnt() > maxDagSize) {
    rslt = RES_FAIL;
    Logger::Error("Dag skipped due to out-of-range size. DAG size = %d, \
                  valid range is [%d, %d]",
                  dag.GetInstCnt(), minDagSize, maxDagSize);
  } else {
    bool filterByPerp = schedIni.GetBool("FILTER_BY_PERP");
    auto blocksToKeep = [&]() {
      auto setting = schedIni.GetString("BLOCKS_TO_KEEP");
      if (setting == "ZERO_COST")
        return BLOCKS_TO_KEEP::ZERO_COST;
      else if (setting == "OPTIMAL")
        return BLOCKS_TO_KEEP::OPTIMAL;
      else if (setting == "IMPROVED")
        return BLOCKS_TO_KEEP::IMPROVED;
      else if (setting == "IMPROVED_OR_OPTIMAL")
        return BLOCKS_TO_KEEP::IMPROVED_OR_OPTIMAL;
      else
        return BLOCKS_TO_KEEP::ALL;
    }();
    rslt = region->FindOptimalSchedule(
        useFileBounds, regionTimeout, lengthTimeout, isEasy, normBestCost,
        bestSchedLngth, normHurstcCost, hurstcSchedLngth, sched, filterByPerp,
        blocksToKeep);
    if ((!(rslt == RES_SUCCESS || rslt == RES_TIMEOUT) || sched == NULL)) {
      Logger::Info("OptSched run failed: rslt=%d, sched=%p. Falling back.",
                   rslt, (void *)sched);

      // Scheduling with opt-sched failed.
      fallbackScheduler();

    } else {
      Logger::Info("OptSched succeeded.");
      // Count simulated spills.
      if (isSimRegAllocEnabled()) {
        totalSimulatedSpills += region->GetSimSpills();
      }

      // Convert back to LLVM.
      // Advance past initial DebugValues.
      CurrentTop = nextIfDebug(RegionBegin, RegionEnd);
      CurrentBottom = RegionEnd;
      InstCount cycle, slot;
      for (InstCount i = sched->GetFrstInst(cycle, slot); i != INVALID_VALUE;
           i = sched->GetNxtInst(cycle, slot)) {
        if (i == SCHD_STALL) {
          ScheduleNode(NULL, cycle);
        } else {
          llvm::SUnit *unit = dag.GetSUnit(i);

          if (unit && unit->isInstr())
            ScheduleNode(unit, cycle);
        }
      }
    } // end OptSched succeeded
  }

#ifdef IS_DEBUG_PEAK_PRESSURE
  Logger::Info("Register pressure after");
  RPTracker.dump();
#endif

  delete region;
}

void ScheduleDAGOptSched::ScheduleNode(llvm::SUnit *SU, unsigned CurCycle) {
#ifdef IS_DEBUG_CONVERT_LLVM
  Logger::Info("*** Scheduling [%lu]: ", CurCycle);
#endif
  if (SU) {
    llvm::MachineInstr *instr = SU->getInstr();
    if (CurrentTop == NULL) {
      Logger::Error("Currenttop is NULL");
      return;
    }
    if (&*CurrentTop == instr)
      CurrentTop = nextIfDebug(++CurrentTop, CurrentBottom);
    else
      moveInstruction(instr, CurrentTop);
  } else {
#ifdef IS_DEBUG_CONVERT_LLVM
    Logger::Info("Stall");
#endif
  }
}

// call the default "Fallback Scheduler" on a region
void ScheduleDAGOptSched::fallbackScheduler() {
  // If the heurisitc is ISO the order of the SUnits will be
  // the order of LLVM's heuristic schedule. Otherwise reset
  // the BB to LLVM's original order, the order of the SUnits,
  // then call their scheduler.
  if (!llvmScheduling)
    ScheduleDAGMILive::schedule();
  // Restore the original llvm schedule that was found earlier.
  else {
    CurrentTop = nextIfDebug(RegionBegin, RegionEnd);
    CurrentBottom = RegionEnd;

    for (size_t i = 0; i < SUnits.size(); i++) {
      llvm::MachineInstr *instr = SUnits[i].getInstr();

      if (CurrentTop == NULL) {
        Logger::Error("Currenttop is NULL");
        return;
      }

      if (&*CurrentTop == instr)
        CurrentTop = nextIfDebug(++CurrentTop, CurrentBottom);
      else
        moveInstruction(instr, CurrentTop);
    }
  }
}

void ScheduleDAGOptSched::loadOptSchedConfig() {
  SchedulerOptions &schedIni = SchedulerOptions::getInstance();
  // setup OptScheduler configuration options
  optSchedEnabled = isOptSchedEnabled();
  latencyPrecision = fetchLatencyPrecision();
  maxDagSizeForLatencyPrecision =
      schedIni.GetInt("MAX_DAG_SIZE_FOR_PRECISE_LATENCY");
  treatOrderDepsAsDataDeps = schedIni.GetBool("TREAT_ORDER_DEPS_AS_DATA_DEPS");

  // should we print spills for the current function
  OPTSCHED_gPrintSpills = shouldPrintSpills();

  // setup pruning
  prune.rlxd = schedIni.GetBool("APPLY_RELAXED_PRUNING");
  prune.nodeSup = schedIni.GetBool("DYNAMIC_NODE_SUPERIORITY");
  prune.histDom = schedIni.GetBool("APPLY_HISTORY_DOMINATION");
  prune.spillCost = schedIni.GetBool("APPLY_SPILL_COST_PRUNING");
  prune.useSuffixConcatenation =
      schedIni.GetBool("ENABLE_SUFFIX_CONCATENATION");

  // setup graph transformations
  graphTransTypes.staticNodeSup = schedIni.GetBool("STATIC_NODE_SUPERIORITY");
  // setup graph transformation flags
  GraphTrans::GRAPHTRANSFLAGS.multiPassNodeSup =
      schedIni.GetBool("MULTI_PASS_NODE_SUPERIORITY");

  schedForRPOnly = schedIni.GetBool("SCHEDULE_FOR_RP_ONLY");
  histTableHashBits =
      static_cast<int16_t>(schedIni.GetInt("HIST_TABLE_HASH_BITS"));
  verifySchedule = schedIni.GetBool("VERIFY_SCHEDULE");
  enableMutations = schedIni.GetBool("LLVM_MUTATIONS");
  enumerateStalls = schedIni.GetBool("ENUMERATE_STALLS");
  spillCostFactor = schedIni.GetInt("SPILL_COST_FACTOR");
  checkSpillCostSum = schedIni.GetBool("CHECK_SPILL_COST_SUM");
  checkConflicts = schedIni.GetBool("CHECK_CONFLICTS");
  fixLiveIn = schedIni.GetBool("FIX_LIVEIN");
  fixLiveOut = schedIni.GetBool("FIX_LIVEOUT");
  maxSpillCost = schedIni.GetInt("MAX_SPILL_COST");
  lowerBoundAlgorithm = parseLowerBoundAlgorithm();
  heuristicPriorities = parseHeuristic(schedIni.GetString("HEURISTIC"));
  // To support old sched.ini files setting NID as the heuristic means LLVM
  // scheduling is enabled.
  llvmScheduling = schedIni.GetBool("LLVM_SCHEDULING", false) ||
                   schedIni.GetString("HEURISTIC") == "NID";
  enumPriorities = parseHeuristic(schedIni.GetString("ENUM_HEURISTIC"));
  spillCostFunction = parseSpillCostFunc();
  regionTimeout = schedIni.GetInt("REGION_TIMEOUT");
  lengthTimeout = schedIni.GetInt("LENGTH_TIMEOUT");
  if (schedIni.GetString("TIMEOUT_PER") == "INSTR")
    isTimeoutPerInstruction = true;
  else
    isTimeoutPerInstruction = false;
  minDagSize = schedIni.GetInt("MIN_DAG_SIZE");
  maxDagSize = schedIni.GetInt("MAX_DAG_SIZE");
  useFileBounds = schedIni.GetBool("USE_FILE_BOUNDS");
  int randomSeed = schedIni.GetInt("RANDOM_SEED", 0);
  if (randomSeed == 0)
    randomSeed = time(NULL);
  RandomGen::SetSeed(randomSeed);
}

bool ScheduleDAGOptSched::isOptSchedEnabled() const {
  // check scheduler ini file to see if optsched is enabled
  std::string optSchedOption =
      SchedulerOptions::getInstance().GetString("USE_OPT_SCHED");
  if (optSchedOption == "YES") {
    return true;
  } else if (optSchedOption == "HOT_ONLY") {
    // get the name of the function this scheduler was created for
    std::string functionName = context->MF->getFunction().getName();
    // check the list of hot functions for the name of the current function
    return hotFunctions.GetBool(functionName, false);
  } else if (optSchedOption == "NO") {
    return false;
  } else {
    DEBUG(llvm::dbgs() << "Invalid value for USE_OPT_SCHED" << optSchedOption
                       << "Assuming NO.\n");
    return false;
  }
}

LATENCY_PRECISION ScheduleDAGOptSched::fetchLatencyPrecision() const {
  std::string lpName =
      SchedulerOptions::getInstance().GetString("LATENCY_PRECISION");
  if (lpName == "PRECISE") {
    return LTP_PRECISE;
  } else if (lpName == "ROUGH") {
    return LTP_ROUGH;
  } else if (lpName == "UNITY") {
    return LTP_UNITY;
  } else {
    Logger::Error("Unrecognized latency precision. Defaulted to PRECISE.");
    return LTP_PRECISE;
  }
}

LB_ALG ScheduleDAGOptSched::parseLowerBoundAlgorithm() const {
  std::string LBalg = SchedulerOptions::getInstance().GetString("LB_ALG");
  if (LBalg == "RJ") {
    return LBA_RJ;
  } else if (LBalg == "LC") {
    return LBA_LC;
  } else {
    Logger::Error("Unrecognized lower bound technique. Defaulted to Rim-Jain.");
    return LBA_RJ;
  }
}

SchedPriorities
ScheduleDAGOptSched::parseHeuristic(const std::string &str) const {
  SchedPriorities prirts;
  size_t len = str.length();
  char word[HEUR_NAME_MAX_SIZE];
  int wIndx = 0;
  prirts.cnt = 0;
  prirts.isDynmc = false;
  size_t i, j;

  for (i = 0; i <= len; i++) {
    char ch = str.c_str()[i];
    if (ch == '_' || ch == 0) { // end of word
      word[wIndx] = 0;
      for (j = 0; j < sizeof(hurstcNames); j++) {
        if (strcmp(word, hurstcNames[j]) == 0) {
          prirts.vctr[prirts.cnt] = (LISTSCHED_HEURISTIC)j;
          if ((LISTSCHED_HEURISTIC)j == LSH_LUC)
            prirts.isDynmc = true;
          break;
        } // end if
      }   // end for j
      if (j == sizeof(hurstcNames)) {
        Logger::Error("Unrecognized heuristic %s. Defaulted to CP.", word);
        prirts.vctr[prirts.cnt] = LSH_CP;
      }
      prirts.cnt++;
      wIndx = 0;
    } else {
      word[wIndx] = ch;
      wIndx++;
    } // end else
  }   // end for i
  return prirts;
}

SPILL_COST_FUNCTION ScheduleDAGOptSched::parseSpillCostFunc() const {
  std::string name =
      SchedulerOptions::getInstance().GetString("SPILL_COST_FUNCTION");
  // PERP used to be called PEAK.
  if (name == "PERP" || name == "PEAK") {
    return SCF_PERP;
  } else if (name == "PRP") {
    return SCF_PRP;
  } else if (name == "PEAK_PER_TYPE") {
    return SCF_PEAK_PER_TYPE;
  } else if (name == "SUM") {
    return SCF_SUM;
  } else if (name == "PEAK_PLUS_AVG") {
    return SCF_PEAK_PLUS_AVG;
  } else if (name == "SLIL") {
    return SCF_SLIL;
  } else {
    Logger::Error("Unrecognized spill cost function. Defaulted to PERP.");
    return SCF_PERP;
  }
}

bool ScheduleDAGOptSched::shouldPrintSpills() {
  std::string printSpills =
      SchedulerOptions::getInstance().GetString("PRINT_SPILL_COUNTS");
  if (printSpills == "YES") {
    return true;
  } else if (printSpills == "NO") {
    return false;
  } else if (printSpills == "HOT_ONLY") {
    std::string functionName = context->MF->getFunction().getName();
    return hotFunctions.GetBool(functionName, false);
  } else {
    Logger::Error("Unknown value for PRINT_SPILL_COUNTS: %s. Assuming NO.",
                  printSpills.c_str());
    return false;
  }
}

bool ScheduleDAGOptSched::rpMismatch(InstSchedule *sched) {
  SetupLLVMDag();

  // LLVM peak register pressure
  const std::vector<unsigned> &RegionPressure =
      RPTracker.getPressure().MaxSetPressure;
  // OptSched preak registesr pressure
  const unsigned *regPressures = nullptr;
  // auto regTypeCount = sched->GetPeakRegPressures(regPressures);

  for (unsigned i = 0, e = RegionPressure.size(); i < e; ++i) {
    if (RegionPressure[i] != regPressures[i])
      return true;
  }

  return false;
}

void ScheduleDAGOptSched::finalizeSchedule() {
  llvm::ScheduleDAGMILive::finalizeSchedule();

  if (isSimRegAllocEnabled()) {
    llvm::dbgs() << "*************************************\n";
    llvm::dbgs() << "Function: " << MF.getName()
                 << "\nTotal Simulated Spills: " << totalSimulatedSpills
                 << "\n";
    llvm::dbgs() << "*************************************\n";
  }
}

bool ScheduleDAGOptSched::isSimRegAllocEnabled() {
  // This will return false if only the list schedule is allocated.
  return (
      OPTSCHED_gPrintSpills &&
      (SchedulerOptions::getInstance().GetString(
           "SIMULATE_REGISTER_ALLOCATION") == "BEST" ||
       SchedulerOptions::getInstance().GetString(
           "SIMULATE_REGISTER_ALLOCATION") == "BOTH" ||
       SchedulerOptions::getInstance().GetString(
           "SIMULATE_REGISTER_ALLOCATION") == "TAKE_SCHED_WITH_LEAST_SPILLS"));
}

void ScheduleDAGOptSched::getRealCfgPaths() {
  // Find full path to OptSchedCfg directory.
  llvm::SmallString<128> PathCfg;
  llvm::sys::fs::real_path(OptSchedCfg, PathCfg, true);

  // If the path to any of the config files are not specified, use the default
  // values.
  if (OptSchedCfgS.empty())
    (PathCfg + DEFAULT_CFGS_FNAME).toVector(PathCfgS);
  else {
    PathCfgS = OptSchedCfgS;
    llvm::SmallString<128> Tmp = PathCfgS;
    llvm::sys::fs::real_path(Tmp, PathCfgS, true);
  }

  if (OptSchedCfgHF.empty())
    (PathCfg + DEFAULT_CFGHF_FNAME).toVector(PathCfgHF);
  else {
    PathCfgHF = OptSchedCfgHF;
    llvm::SmallString<128> Tmp = PathCfgHF;
    llvm::sys::fs::real_path(Tmp, PathCfgHF, true);
  }

  if (OptSchedCfgMM.empty())
    (PathCfg + DEFAULT_CFGMM_FNAME).toVector(PathCfgMM);
  else {
    PathCfgMM = OptSchedCfgMM;
    llvm::SmallString<128> Tmp = PathCfgHF;
    llvm::sys::fs::real_path(Tmp, PathCfgMM, true);
  }

  // Convert full paths to native fromat.
  llvm::sys::path::native(PathCfgS);
  llvm::sys::path::native(PathCfgHF);
  llvm::sys::path::native(PathCfgMM);
}
