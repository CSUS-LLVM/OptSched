//===- OptimizingScheduler.cpp - The optimizing scheduler -----------------===//
//
// Implements a combinatorial scheduling algorithm.
//
//===----------------------------------------------------------------------===//
#include "OptimizingScheduler.h"
#include "OptSchedDDGWrapperBasic.h"
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedDDGWrapperBase.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/utilities.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <string>

#define DEBUG_TYPE "optsched"

using namespace llvm::opt_sched;

// hack to print spills
bool OPTSCHED_gPrintSpills;

// An array of possible OptSched heuristic names
#define LSHPair std::pair<const char *, LISTSCHED_HEURISTIC>
static LSHPair HeuristicNames[] = {
    LSHPair("CP", LSH_CP),    LSHPair("LUC", LSH_LUC),
    LSHPair("UC", LSH_UC),    LSHPair("NID", LSH_NID),
    LSHPair("CPR", LSH_CPR),  LSHPair("ISO", LSH_ISO),
    LSHPair("SC", LSH_SC),    LSHPair("LS", LSH_LS),
    LSHPair("LLVM", LSH_LLVM)};

// Max size for heuristic name.
static constexpr int HEUR_NAME_MAX_SIZE = 10;

// Default path to the the configuration directory for opt-sched.
static constexpr const char *DEFAULT_CFG_DIR = "~/.optsched-cfg/";

// Default path to the scheduler options configuration file for opt-sched.
static constexpr const char *DEFAULT_CFGS_FNAME = "/sched.ini";

// Default path to the list of hot functions to schedule using opt-sched.
static constexpr const char *DEFAULT_CFGHF_FNAME = "/hotfuncs.ini";

// Default path to the machine model specification file for opt-sched.
static constexpr const char *DEFAULT_CFGMM_FNAME = "/machine_model.cfg";

// Create OptSched ScheduleDAG.
static ScheduleDAGInstrs *createOptSched(MachineSchedContext *C) {
  return new ScheduleDAGOptSched(C, llvm::make_unique<GenericScheduler>(C));
}

// Register the machine scheduler.
static MachineSchedRegistry OptSchedMIRegistry("optsched",
                                               "Use the OptSched scheduler.",
                                               createOptSched);

// Command line options for opt-sched.
static cl::opt<std::string> OptSchedCfg(
    "optsched-cfg", cl::Hidden,
    cl::desc(
        "Path to the directory containing configuration files for opt-sched."),
    cl::init(DEFAULT_CFG_DIR));

static cl::opt<std::string> OptSchedCfgS(
    "optsched-cfg-sched", cl::Hidden,
    cl::desc(
        "Path to the scheduler options configuration file for opt-sched."));

static cl::opt<std::string> OptSchedCfgHF(
    "optsched-cfg-hotfuncs", cl::Hidden,
    cl::desc("Path to the list of hot functions to schedule using opt-sched."));

static cl::opt<std::string> OptSchedCfgMM(
    "optsched-cfg-machine-model", cl::Hidden,
    cl::desc("Path to the machine model specification file for opt-sched."));

static void getRealCfgPathCL(SmallString<128> &Path) {
  SmallString<128> Tmp = Path;
  auto EC = sys::fs::real_path(Tmp, Path, true);
  if (EC)
    llvm::report_fatal_error(EC.message() + ": " + Tmp, false);
}

static void reportCfgDirPathError(std::error_code EC,
                                  llvm::StringRef OptSchedCfg) {
  if (OptSchedCfg == DEFAULT_CFG_DIR)
    llvm::report_fatal_error(EC.message() +
                                 ": Error searching for the OptSched config "
                                 "directory in the default location: " +
                                 DEFAULT_CFG_DIR,
                             false);
  else
    llvm::report_fatal_error(EC.message() + ": " + OptSchedCfg, false);
}

// If this iterator is a debug value, increment until reaching the End or a
// non-debug instruction. static function copied from
// llvm/CodeGen/MachineScheduler.cpp
static MachineBasicBlock::iterator
nextIfDebug(MachineBasicBlock::iterator I,
            MachineBasicBlock::const_iterator End) {
  for (; I != End; ++I) {
    if (!I->isDebugValue())
      break;
  }
  return I;
}

static bool skipRegion(const StringRef RegionName, const Config &SchedIni) {
  const bool ScheduleSpecificRegions =
      SchedIni.GetBool("SCHEDULE_SPECIFIC_REGIONS");

  if (!ScheduleSpecificRegions)
    return false;

  const std::list<std::string> regionList =
      SchedIni.GetStringList("REGIONS_TO_SCHEDULE");
  return std::find(std::begin(regionList), std::end(regionList), RegionName) ==
         std::end(regionList);
}

static BLOCKS_TO_KEEP blocksToKeep(const Config &SchedIni) {
  const auto &Setting = SchedIni.GetString("BLOCKS_TO_KEEP");
  if (Setting == "ZERO_COST")
    return BLOCKS_TO_KEEP::ZERO_COST;
  if (Setting == "OPTIMAL")
    return BLOCKS_TO_KEEP::OPTIMAL;
  if (Setting == "IMPROVED")
    return BLOCKS_TO_KEEP::IMPROVED;
  if (Setting == "IMPROVED_OR_OPTIMAL")
    return BLOCKS_TO_KEEP::IMPROVED_OR_OPTIMAL;

  return BLOCKS_TO_KEEP::ALL;
}

static SchedulerType parseListSchedType() {
  auto SchedTypeString =
      SchedulerOptions::getInstance().GetString("HEUR_SCHED_TYPE", "LIST");
  if (SchedTypeString == "LIST")
    return SCHED_LIST;
  if (SchedTypeString == "SEQ")
    return SCHED_SEQ;
  if (SchedTypeString == "ACO")
    return SCHED_ACO;

  Logger::Info("Unknown heuristic scheduler type selected defaulting to basic "
               "list scheduler.");
  return SCHED_LIST;
}

static std::unique_ptr<GraphTrans>
createStaticNodeSupTrans(DataDepGraph *DataDepGraph, bool IsMultiPass = false) {
  return llvm::make_unique<StaticNodeSupTrans>(DataDepGraph, IsMultiPass);
}

void ScheduleDAGOptSched::addGraphTransformations(
    OptSchedDDGWrapperBasic *BDDG) {
  auto *GraphTransfomations = BDDG->GetGraphTrans();
  if (StaticNodeSup)
    GraphTransfomations->push_back(
        createStaticNodeSupTrans(BDDG, MultiPassStaticNodeSup));
}

ScheduleDAGOptSched::ScheduleDAGOptSched(
    MachineSchedContext *C, std::unique_ptr<MachineSchedStrategy> S)
    : ScheduleDAGMILive(C, std::move(S)), C(C) {
  LLVM_DEBUG(dbgs() << "********** Optimizing Scheduler **********\n");

  secondPass = false;

  // Find the native paths to the scheduler configuration files.
  getRealCfgPaths();

  // Setup config object
  Config &schedIni = SchedulerOptions::getInstance();
  // load OptSched ini file
  schedIni.Load(PathCfgS.c_str());

  // load hot functions ini file
  HotFunctions.Load(PathCfgHF.c_str());

  // Load config files for the OptScheduler
  loadOptSchedConfig();

  StringRef ArchName = TM.getTargetTriple().getArchName();
  auto TargetFactory =
      OptSchedTargetRegistry::Registry.getFactoryWithName(ArchName);

  if (!TargetFactory)
    TargetFactory =
        OptSchedTargetRegistry::Registry.getFactoryWithName("generic");

  OST = TargetFactory();
  MM = OST->createMachineModel(PathCfgMM.c_str());
  MM->convertMachineModel(static_cast<ScheduleDAGInstrs &>(*this),
                          RegClassInfo);
}

void ScheduleDAGOptSched::SetupLLVMDag() {
  // Initialize the register pressure tracker used by buildSchedGraph.
  RPTracker.init(&MF, RegClassInfo, LIS, BB, LiveRegionEnd,
                 /*ShouldTrackLaneMasks=*/true, /*TrackUntiedDefs=*/true);

  // Account for liveness generate by the region boundary. LiveRegionEnd is
  // the end of the MBB.
  if (LiveRegionEnd != RegionEnd)
    RPTracker.recede();

  // Build the DAG, and compute current register pressure.
  buildSchedGraph(AA, &RPTracker, nullptr, LIS, /*ShouldTrackLaneMasks=*/true);

  // Finalize live-in
  RPTracker.closeTop();

  // Apply llvm DAG post processing.
  if (EnableMutations) {
    Topo.InitDAGTopologicalSorting();
    postprocessDAG();
  }
}

// schedule called for each basic block
void ScheduleDAGOptSched::schedule() {
  ShouldTrackPressure = true;
  ShouldTrackLaneMasks = true;
  Config &schedIni = SchedulerOptions::getInstance();

  ++RegionNumber;
  const std::string RegionName = C->MF->getFunction().getName().data() +
                                 std::string(":") +
                                 std::to_string(RegionNumber);
  if (!OptSchedEnabled || skipRegion(RegionName, schedIni)) {
    LLVM_DEBUG(dbgs() << "Skipping region " << RegionName << "\n");
    return;
  }
  Logger::Info("********** Opt Scheduling **********");
  LLVM_DEBUG(dbgs() << "********** Scheduling Region " << RegionName
                    << " **********\n");
  LLVM_DEBUG(const auto *MBB = RegionBegin->getParent();
             dbgs() << MF.getName() << ":" << printMBBReference(*MBB) << " "
                    << MBB->getName() << "\n  From: " << *begin() << "    To: ";
             if (RegionEnd != MBB->end()) dbgs() << *RegionEnd;
             else dbgs() << "End";
             dbgs() << " RegionInstrs: " << NumRegionInstrs << '\n');

#ifdef IS_DEBUG_PEAK_PRESSURE
  SetupLLVMDag();
  Logger::Info("RP before scheduling");
  RPTracker.dump();
#endif

  // Use LLVM's heuristic schedule as input to the B&B scheduler.
  if (UseLLVMScheduler) {
    ScheduleDAGMILive::schedule();

    OriginalDAG = SUnits;
    // The schedule generated by LLVM
    ISOSchedule.resize(SUnits.size());
    // A vector index by NodeNum. The value
    // is the order in LLVM's schedule.
    std::vector<int> nodeNumMap(SUnits.size());

    // Find the schedule generated by LLVM which is stored in MMB.
    int counter = 0;
    for (MachineBasicBlock::instr_iterator I = BB->instr_begin(),
                                           E = BB->instr_end();
         I != E; ++I) {
      MachineInstr &instr = *I;
      SUnit *su = getSUnit(&instr);

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
      SUnit *unit = &SUnits[i];
      // Update successor NodeNums.
      for (SDep &dep : unit->Succs) {
        SUnit *succ = dep.getSUnit();
        // check if the successor is a boundary node
        if (succ->isBoundaryNode())
          continue;
        dep.setSUnit(&SUnits[nodeNumMap[succ->NodeNum]]);
      }

      // Update predecessor NodeNums.
      for (SDep &dep : unit->Preds) {
        SUnit *pred = dep.getSUnit();
        // check if the predecessor is a boundary node
        if (pred->isBoundaryNode())
          continue;
        dep.setSUnit(&SUnits[nodeNumMap[pred->NodeNum]]);
      }
    }
  }

  // Build LLVM DAG
  SetupLLVMDag();
  OST->initRegion(this, MM.get());
  // Convert graph
  auto DDG =
      OST->createDDGWrapper(C, this, MM.get(), LatencyPrecision, RegionName);
  DDG->convertSUnits();
  DDG->convertRegFiles();

  auto *BDDG = static_cast<OptSchedDDGWrapperBasic *>(DDG.get());
  addGraphTransformations(BDDG);

  // create region
  SchedRegion *region = new BBWithSpill(
      OST.get(), static_cast<DataDepGraph *>(DDG.get()), 0, HistTableHashBits,
      LowerBoundAlgorithm, HeuristicPriorities, EnumPriorities, VerifySchedule,
      PruningStrategy, SchedForRPOnly, EnumStalls, SCW, SCF, HeurSchedType);

  bool IsEasy = false;
  InstCount NormBestCost = 0;
  InstCount BestSchedLngth = 0;
  InstCount NormHurstcCost = 0;
  InstCount HurstcSchedLngth = 0;
  InstSchedule *Sched = NULL;
  FUNC_RESULT Rslt;
  bool FilterByPerp = schedIni.GetBool("FILTER_BY_PERP");

  if (IsTimeoutPerInst) {
    // Re-calculate timeout values if timeout setting is per instruction
    // becuase we want a unique value per DAG size
    RegionTimeout = schedIni.GetInt("REGION_TIMEOUT") * SUnits.size();
    LengthTimeout = schedIni.GetInt("LENGTH_TIMEOUT") * SUnits.size();
  }

  // Used for two-pass-optsched to alter upper bound value.
  if (secondPass)
    region->InitSecondPass();

  // Setup time before scheduling
  Utilities::startTime = std::chrono::high_resolution_clock::now();
  // Schedule region.
  Rslt = region->FindOptimalSchedule(RegionTimeout, LengthTimeout, IsEasy,
                                     NormBestCost, BestSchedLngth,
                                     NormHurstcCost, HurstcSchedLngth, Sched,
                                     FilterByPerp, blocksToKeep(schedIni));

  if ((!(Rslt == RES_SUCCESS || Rslt == RES_TIMEOUT) || Sched == NULL)) {
    LLVM_DEBUG(
        Logger::Info("OptSched run failed: rslt=%d, sched=%p. Falling back.",
                     Rslt, (void *)Sched));
    // Scheduling with opt-sched failed.
    // fallbackScheduler();
    return;
  }

  LLVM_DEBUG(Logger::Info("OptSched succeeded."));
  OST->finalizeRegion(Sched);
  if (!OST->shouldKeepSchedule())
    return;

  // Count simulated spills.
  if (isSimRegAllocEnabled()) {
    SimulatedSpills += region->GetSimSpills();
  }

  // Convert back to LLVM.
  // Advance past initial DebugValues.
  CurrentTop = nextIfDebug(RegionBegin, RegionEnd);
  CurrentBottom = RegionEnd;
  InstCount cycle, slot;
  for (InstCount i = Sched->GetFrstInst(cycle, slot); i != INVALID_VALUE;
       i = Sched->GetNxtInst(cycle, slot)) {
    // Skip artificial instrs.
    if (i > static_cast<int>(SUnits.size()) - 1)
      continue;

    if (i == SCHD_STALL)
      ScheduleNode(NULL, cycle);
    else {
      SUnit *unit = &SUnits[i];
      if (unit && unit->isInstr())
        ScheduleNode(unit, cycle);
    }
  }
  placeDebugValues();

#ifdef IS_DEBUG_PEAK_PRESSURE
  Logger::Info("Register pressure after");
  RPTracker.dump();
#endif

  delete region;
}

void ScheduleDAGOptSched::ScheduleNode(SUnit *SU, unsigned CurCycle) {
#ifdef IS_DEBUG_CONVERT_LLVM
  Logger::Info("*** Scheduling [%lu]: ", CurCycle);
#endif
  if (SU) {
    MachineInstr *instr = SU->getInstr();
    // Reset read - undef flags and update them later.
    for (auto &Op : instr->operands())
      if (Op.isReg() && Op.isDef())
        Op.setIsUndef(false);

    if (&*CurrentTop == instr)
      CurrentTop = nextIfDebug(++CurrentTop, CurrentBottom);
    else
      moveInstruction(instr, CurrentTop);

    RegisterOperands RegOpers;
    RegOpers.collect(*instr, *TRI, MRI, true, false);
    // Adjust liveness and add missing dead+read-undef flags.
    auto SlotIdx = LIS->getInstructionIndex(*instr).getRegSlot();
    RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, instr);
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
  if (!UseLLVMScheduler)
    ScheduleDAGMILive::schedule();
  // Restore the original llvm schedule that was found earlier.
  else {
    CurrentTop = nextIfDebug(RegionBegin, RegionEnd);
    CurrentBottom = RegionEnd;

    for (size_t i = 0; i < SUnits.size(); i++) {
      MachineInstr *instr = SUnits[i].getInstr();

      if (CurrentTop == NULL) {
        LLVM_DEBUG(Logger::Error("Currenttop is NULL"));
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
  OptSchedEnabled = isOptSchedEnabled();
  LatencyPrecision = fetchLatencyPrecision();
  TreatOrderAsDataDeps = schedIni.GetBool("TREAT_ORDER_DEPS_AS_DATA_DEPS");

  UseLLVMScheduler = false;
  // should we print spills for the current function
  OPTSCHED_gPrintSpills = shouldPrintSpills();
  StaticNodeSup = schedIni.GetBool("STATIC_NODE_SUPERIORITY", false);
  MultiPassStaticNodeSup =
      schedIni.GetBool("MULTI_PASS_NODE_SUPERIORITY", false);
  // setup pruning
  PruningStrategy.rlxd = schedIni.GetBool("APPLY_RELAXED_PRUNING");
  PruningStrategy.nodeSup = schedIni.GetBool("DYNAMIC_NODE_SUPERIORITY");
  PruningStrategy.histDom = schedIni.GetBool("APPLY_HISTORY_DOMINATION");
  PruningStrategy.spillCost = schedIni.GetBool("APPLY_SPILL_COST_PRUNING");
  PruningStrategy.useSuffixConcatenation =
      schedIni.GetBool("ENABLE_SUFFIX_CONCATENATION");
  MultiPassStaticNodeSup = schedIni.GetBool("MULTI_PASS_NODE_SUPERIORITY");
  SchedForRPOnly = schedIni.GetBool("SCHEDULE_FOR_RP_ONLY");
  HistTableHashBits =
      static_cast<int16_t>(schedIni.GetInt("HIST_TABLE_HASH_BITS"));
  VerifySchedule = schedIni.GetBool("VERIFY_SCHEDULE");
  EnableMutations = schedIni.GetBool("LLVM_MUTATIONS");
  EnumStalls = schedIni.GetBool("ENUMERATE_STALLS");
  SCW = schedIni.GetInt("SPILL_COST_WEIGHT");
  LowerBoundAlgorithm = parseLowerBoundAlgorithm();
  HeuristicPriorities = parseHeuristic(schedIni.GetString("HEURISTIC"));
  EnumPriorities = parseHeuristic(schedIni.GetString("ENUM_HEURISTIC"));
  SCF = parseSpillCostFunc();
  RegionTimeout = schedIni.GetInt("REGION_TIMEOUT");
  LengthTimeout = schedIni.GetInt("LENGTH_TIMEOUT");
  if (schedIni.GetString("TIMEOUT_PER") == "INSTR")
    IsTimeoutPerInst = true;
  else
    IsTimeoutPerInst = false;
  int randomSeed = schedIni.GetInt("RANDOM_SEED", 0);
  if (randomSeed == 0)
    randomSeed = time(NULL);
  RandomGen::SetSeed(randomSeed);
  HeurSchedType = parseListSchedType();
}

bool ScheduleDAGOptSched::isOptSchedEnabled() const {
  // check scheduler ini file to see if optsched is enabled
  auto optSchedOption =
      SchedulerOptions::getInstance().GetString("USE_OPT_SCHED");
  if (optSchedOption == "YES") {
    return true;
  } else if (optSchedOption == "HOT_ONLY") {
    // get the name of the function this scheduler was created for
    std::string functionName = C->MF->getFunction().getName();
    // check the list of hot functions for the name of the current function
    return HotFunctions.GetBool(functionName, false);
  } else if (optSchedOption == "NO") {
    return false;
  } else {
    LLVM_DEBUG(dbgs() << "Invalid value for USE_OPT_SCHED" << optSchedOption
                      << "Assuming NO.\n");
    return false;
  }
}

LATENCY_PRECISION ScheduleDAGOptSched::fetchLatencyPrecision() const {
  std::string lpName =
      SchedulerOptions::getInstance().GetString("LATENCY_PRECISION");
  if (lpName == "FILE" || lpName == "PRECISE") {
    return LTP_PRECISE;
  } else if (lpName == "LLVM" || lpName == "ROUGH") {
    return LTP_ROUGH;
  } else if (lpName == "UNIT" || lpName == "UNITY") {
    return LTP_UNITY;
  } else {
    LLVM_DEBUG(
        Logger::Error("Unrecognized latency precision. Defaulted to PRECISE."));
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
    LLVM_DEBUG(Logger::Error(
        "Unrecognized lower bound technique. Defaulted to Rim-Jain."));
    return LBA_RJ;
  }
}

// Helper function to find the next substring which is a heuristic name in Str
static LISTSCHED_HEURISTIC GetNextHeuristicName(const std::string &Str,
                                                size_t &StartIndex) {
  size_t Walk;
  for (Walk = StartIndex; Walk <= Str.length(); ++Walk) {
    if (Str[Walk] == '_')
      break;
  }

  // Match heuristic name to enum id
  for (const auto &LSH : HeuristicNames)
    if (!Str.compare(StartIndex, Walk - StartIndex, LSH.first)) {
      StartIndex = Walk + 1;
      return LSH.second;
    }
  llvm_unreachable("Unknown heuristic.");
}

SchedPriorities ScheduleDAGOptSched::parseHeuristic(const std::string &Str) {
  SchedPriorities Priorities;
  size_t StartIndex = 0;
  Priorities.cnt = 0;
  Priorities.isDynmc = false;
  do {
    LISTSCHED_HEURISTIC LSH = GetNextHeuristicName(Str, StartIndex);
    Priorities.vctr[Priorities.cnt++] = LSH;
    switch (LSH) {
    // Is LUC still the only dynamic heuristic?
    case LSH_LUC:
      Priorities.isDynmc = true;
      break;
    case LSH_LLVM:
      UseLLVMScheduler = true;
      break;
    default:
      break;
    }
  } while (!(StartIndex > Str.length()));

  return Priorities;
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
  } else if (name == "OCC" || name == "TARGET") {
    return SCF_TARGET;
  } else {
    LLVM_DEBUG(
        Logger::Error("Unrecognized spill cost function. Defaulted to PERP."));
    return SCF_PERP;
  }
}

bool ScheduleDAGOptSched::shouldPrintSpills() const {
  std::string printSpills =
      SchedulerOptions::getInstance().GetString("PRINT_SPILL_COUNTS");
  if (printSpills == "YES") {
    return true;
  } else if (printSpills == "NO") {
    return false;
  } else if (printSpills == "HOT_ONLY") {
    std::string functionName = C->MF->getFunction().getName();
    return HotFunctions.GetBool(functionName, false);
  } else {
    LLVM_DEBUG(
        Logger::Error("Unknown value for PRINT_SPILL_COUNTS: %s. Assuming NO.",
                      printSpills.c_str()));
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
  ScheduleDAGMILive::finalizeSchedule();

  LLVM_DEBUG(if (isSimRegAllocEnabled()) {
    dbgs() << "*************************************\n";
    dbgs() << "Function: " << MF.getName()
           << "\nTotal Simulated Spills: " << SimulatedSpills << "\n";
    dbgs() << "*************************************\n";
  });
}

bool ScheduleDAGOptSched::isSimRegAllocEnabled() const {
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
  SmallString<128> PathCfg;
  auto EC = sys::fs::real_path(OptSchedCfg, PathCfg, true);
  if (EC)
    reportCfgDirPathError(EC, OptSchedCfg);

  // If the path to any of the config files are not specified, use the default
  // values.
  if (OptSchedCfgS.empty())
    (PathCfg + DEFAULT_CFGS_FNAME).toVector(PathCfgS);
  else {
    PathCfgS = OptSchedCfgS;
    getRealCfgPathCL(PathCfgS);
  }

  if (OptSchedCfgHF.empty())
    (PathCfg + DEFAULT_CFGHF_FNAME).toVector(PathCfgHF);
  else {
    PathCfgHF = OptSchedCfgHF;
    getRealCfgPathCL(PathCfgHF);
  }

  if (OptSchedCfgMM.empty())
    (PathCfg + DEFAULT_CFGMM_FNAME).toVector(PathCfgMM);
  else {
    PathCfgMM = OptSchedCfgMM;
    getRealCfgPathCL(PathCfgMM);
  }

  // Convert full paths to native fromat.
  sys::path::native(PathCfgS);
  sys::path::native(PathCfgHF);
  sys::path::native(PathCfgMM);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// Print registers from RegisterMaskPair vector
static Printable
printMaskPairs(const SmallVectorImpl<RegisterMaskPair> &RegPairs,
               const TargetRegisterInfo *TRI, const MachineRegisterInfo &MRI) {
  return Printable([&RegPairs, TRI, &MRI](raw_ostream &OS) {
    for (const auto &P : RegPairs) {
      const TargetRegisterClass *RegClass;

      if (TRI->isPhysicalRegister(P.RegUnit))
        RegClass = TRI->getMinimalPhysRegClass(P.RegUnit);
      else if (TRI->isVirtualRegister(P.RegUnit))
        RegClass = MRI.getRegClass(P.RegUnit);
      else
        RegClass = nullptr;

      OS << printReg(P.RegUnit, TRI, 0) << ':' << PrintLaneMask(P.LaneMask)
         << " (" << (RegClass ? TRI->getRegClassName(RegClass) : "noclass")
         << ") ";
    }
  });
}

LLVM_DUMP_METHOD
void ScheduleDAGOptSched::dumpLLVMRegisters() const {
  dbgs() << "LLVM Registers\n";
  for (const auto &SU : SUnits) {
    assert(SU.isInstr());
    const MachineInstr *MI = SU.getInstr();
    dbgs() << "Instr: ";
    dbgs() << "(" << SU.NodeNum << ") " << TII->getName(MI->getOpcode())
           << '\n';

    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *TRI, MRI, true, false);

    dbgs() << "\t--Defs: " << printMaskPairs(RegOpers.Defs, TRI, MRI);
    dbgs() << '\n';

    dbgs() << "\t--Uses: " << printMaskPairs(RegOpers.Uses, TRI, MRI);
    dbgs() << "\n\n";
  }

  // Print registers used/defined by the region boundary
  const MachineInstr *MI = getRegionEnd();
  // Make sure EndRegion is not a sentinel value
  if (MI) {
    dbgs() << "Instr: ";
    dbgs() << "(ExitSU) " << TII->getName(MI->getOpcode()) << '\n';

    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *TRI, MRI, true, false);

    dbgs() << "\t--Defs: " << printMaskPairs(RegOpers.Defs, TRI, MRI);
    dbgs() << '\n';

    dbgs() << "\t--Uses: " << printMaskPairs(RegOpers.Uses, TRI, MRI);
    dbgs() << "\n\n";
  }

  // Print live-in/live-out register
  dbgs() << "Live-In/Live-Out registers:\n";
  dbgs() << "\t--Live-In: "
         << printMaskPairs(getRegPressure().LiveInRegs, TRI, MRI);
  dbgs() << '\n';

  dbgs() << "\t--Live-Out: "
         << printMaskPairs(getRegPressure().LiveOutRegs, TRI, MRI);
  dbgs() << "\n\n";
}
#endif
