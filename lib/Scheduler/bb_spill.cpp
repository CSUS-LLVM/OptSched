#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/enumerator.h"
#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/reg_alloc.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/relaxed_sched.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/utilities.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

extern bool OPTSCHED_gPrintSpills;

using namespace llvm::opt_sched;

// The denominator used when calculating cost weight.
static const int COST_WGHT_BASE = 100;

// The max number of instructions in a cluster
static const unsigned MAX_INSTR_IN_CLUSTER = 15;

BBWithSpill::BBWithSpill(const OptSchedTarget *OST_, DataDepGraph *dataDepGraph,
                         long rgnNum, int16_t sigHashSize, LB_ALG lbAlg,
                         SchedPriorities hurstcPrirts,
                         SchedPriorities enumPrirts, bool vrfySched,
                         Pruning PruningStrategy, bool SchedForRPOnly,
                         bool enblStallEnum, int SCW,
                         SPILL_COST_FUNCTION spillCostFunc,
                         SchedulerType HeurSchedType)
    : SchedRegion(OST_->MM, dataDepGraph, rgnNum, sigHashSize, lbAlg,
                  hurstcPrirts, enumPrirts, vrfySched, PruningStrategy,
                  HeurSchedType, spillCostFunc),
      OST(OST_) {
  enumrtr_ = NULL;
  optmlSpillCost_ = INVALID_VALUE;

  crntCycleNum_ = INVALID_VALUE;
  crntSlotNum_ = INVALID_VALUE;
  crntSpillCost_ = INVALID_VALUE;

  SchedForRPOnly_ = SchedForRPOnly;

  enblStallEnum_ = enblStallEnum;
  SCW_ = SCW;
  schedCostFactor_ = COST_WGHT_BASE;
  trackLiveRangeLngths_ = true;

  regTypeCnt_ = OST->MM->GetRegTypeCnt();
  regFiles_ = dataDepGraph->getRegFiles();
  liveRegs_ = new WeightedBitVector[regTypeCnt_];
  livePhysRegs_ = new WeightedBitVector[regTypeCnt_];
  spillCosts_ = new InstCount[dataDepGraph_->GetInstCnt()];
  peakRegPressures_ = new InstCount[regTypeCnt_];
  regPressures_.resize(regTypeCnt_);
  sumOfLiveIntervalLengths_.resize(regTypeCnt_, 0);

  entryInstCnt_ = 0;
  exitInstCnt_ = 0;
  schduldEntryInstCnt_ = 0;
  schduldExitInstCnt_ = 0;
  schduldInstCnt_ = 0;
  ClusterGroupCount = dataDepGraph_->getMinClusterCount();
  MinClusterBlocks = 0;
//  if (ClusterMemoryOperations && ClusterGroupCount > 0) {
  if (ClusterGroupCount > 0) {
    ClusterCount.resize(ClusterGroupCount + 1);
    ClusterInstrRemainderCount.resize(ClusterGroupCount + 1);
    MinClusterBlocks = calculateClusterStaticLB();
    initForClustering();
  }
}
/****************************************************************************/

void BBWithSpill::initForClustering() {
  // Memory clustering variables initialization
  SchedInstruction::SetActiveCluster(0);
  CurrentClusterSize = 0;
  ClusterActiveGroup = 0;
  CurrentClusterCost = 0;
  PastClustersList.clear();
  LastCluster.reset();
  InstrList.reset();
  DynamicClusterLowerBound = 0;

  for (int begin = 1; begin <= ClusterGroupCount; begin++) {
    ClusterCount[begin] = 0;
    ClusterInstrRemainderCount[begin] =
        dataDepGraph_->getTotalInstructionsInCluster(begin);
  }
}

BBWithSpill::~BBWithSpill() {
  if (enumrtr_ != NULL) {
    delete enumrtr_;
  }

  delete[] liveRegs_;
  delete[] livePhysRegs_;
  delete[] spillCosts_;
  delete[] peakRegPressures_;
}
/*****************************************************************************/

int BBWithSpill::calculateClusterStaticLB() {
  // No cluster in this scheduling region
  if (ClusterGroupCount == 0)
    return 0;

  // Calculate the minimum cluster blocks that will be needed to cluster all of
  // the instructions. The maximum amount in a cluster block is determined by
  // the constant MAX_INSTR_IN_CLUSTER.
  int ClusterCost = 0;
  for (int begin = 1; begin <= ClusterGroupCount; begin++) {
    int InstructionCount = dataDepGraph_->getTotalInstructionsInCluster(begin);
    int CurrentClusterCost =
        std::ceil(double(InstructionCount) / MAX_INSTR_IN_CLUSTER);
    Logger::Info("Cost for block %d is %d", begin, CurrentClusterCost);
    ClusterCost += CurrentClusterCost;
  }

  return ClusterCost;
}

bool BBWithSpill::EnableEnum_() {
  return true;
  /*
  if (maxSpillCost_ > 0 && hurstcCost_ > maxSpillCost_) {
    Logger::Info("Bypassing enumeration due to a large spill cost of %d",
                 hurstcCost_);
    return false;
  }
  return true;
  */
}
/*****************************************************************************/

ConstrainedScheduler *BBWithSpill::AllocHeuristicScheduler_() {
  switch (GetHeuristicSchedulerType()) {
  case SCHED_LIST:
    return new ListScheduler(dataDepGraph_, machMdl_, abslutSchedUprBound_,
                             GetHeuristicPriorities());
    break;
  case SCHED_SEQ:
    return new SequentialListScheduler(dataDepGraph_, machMdl_,
                                       abslutSchedUprBound_,
                                       GetHeuristicPriorities());
    break;
  }
  llvm_unreachable("Unknown heuristic scheduler type!");
}
/*****************************************************************************/

void BBWithSpill::SetupPhysRegs_() {
  int physRegCnt;
  for (int i = 0; i < regTypeCnt_; i++) {
    physRegCnt = regFiles_[i].FindPhysRegCnt();
    if (physRegCnt > 0)
      livePhysRegs_[i].Construct(physRegCnt);
  }
}
/*****************************************************************************/

void BBWithSpill::CmputAbslutUprBound_() {
  abslutSchedUprBound_ = dataDepGraph_->GetAbslutSchedUprBound();
  dataDepGraph_->SetAbslutSchedUprBound(abslutSchedUprBound_);
}
/*****************************************************************************/

void BBWithSpill::CmputSchedUprBound_() {
  // The maximum increase in sched length that might result in a smaller cost
  // than the known one
  int maxLngthIncrmnt = (GetBestCost() - 1) / schedCostFactor_;

  if (machMdl_->IsSimple() && dataDepGraph_->GetMaxLtncy() <= 1) {
#if defined(IS_DEBUG_DAG) || defined(IS_DEBUG_SIMPLE_DAGS)
    Logger::Info("Simple DAG with max latency of one or less.");
#endif
    maxLngthIncrmnt = 0;
  }

  assert(maxLngthIncrmnt >= 0);

  // Any schedule longer than this will have a cost that is greater than or
  // equal to that of the list schedule
  schedUprBound_ = schedLwrBound_ + maxLngthIncrmnt;

  if (abslutSchedUprBound_ < schedUprBound_) {
    schedUprBound_ = abslutSchedUprBound_;
  }
}
/*****************************************************************************/

static InstCount ComputeSLILStaticLowerBound(int64_t regTypeCnt_,
                                             RegisterFile *regFiles_,
                                             DataDepGraph *dataDepGraph_) {
  // (Chris): To calculate a naive lower bound of the SLIL, count all the defs
  // and uses for each register.
  int naiveLowerBound = 0;
  for (int i = 0; i < regTypeCnt_; ++i) {
    for (int j = 0; j < regFiles_[i].GetRegCnt(); ++j) {
      const auto &reg = regFiles_[i].GetReg(j);
      for (const auto &instruction : reg->GetDefList()) {
        if (reg->AddToInterval(instruction)) {
          ++naiveLowerBound;
        }
      }
      for (const auto &instruction : reg->GetUseList()) {
        if (reg->AddToInterval(instruction)) {
          ++naiveLowerBound;
        }
      }
    }
  }

#if defined(IS_DEBUG_SLIL_COST_LOWER_BOUND)
  Logger::Info("SLIL Naive Static Lower Bound Cost  is %llu for Dag %s",
               naiveLowerBound, dataDepGraph_->GetDagID());
#endif

  // (Chris): Another improvement to the lower bound calculation takes advantage
  // of the transitive closure of the DAG. Suppose instruction X must happen
  // between A and B, where A defines a register that B uses. Then, the live
  // range length of A increases by 1.
  auto closureLowerBound = naiveLowerBound;
  for (int i = 0; i < dataDepGraph_->GetInstCnt(); ++i) {
    const auto &inst = dataDepGraph_->GetInstByIndx(i);
    // For each register this instruction defines, compute the intersection
    // between the recursive successor list of this instruction and the
    // recursive predecessors of the dependent instruction.
    Register **definedRegisters = nullptr;
    auto defRegCount = inst->GetDefs(definedRegisters);
    auto recSuccBV = inst->GetRcrsvNghbrBitVector(DIR_FRWRD);
    for (int j = 0; j < defRegCount; ++j) {
      for (const auto &dependentInst : definedRegisters[j]->GetUseList()) {
        auto recPredBV = const_cast<SchedInstruction *>(dependentInst)
                             ->GetRcrsvNghbrBitVector(DIR_BKWRD);
        assert(recSuccBV->GetSize() == recPredBV->GetSize() &&
               "Successor list size doesn't match predecessor list size!");
        for (int k = 0; k < recSuccBV->GetSize(); ++k) {
          if (recSuccBV->GetBit(k) & recPredBV->GetBit(k)) {
            if (definedRegisters[j]->AddToInterval(
                    dataDepGraph_->GetInstByIndx(k))) {
              ++closureLowerBound;
            }
          }
        }
      }
    }
  }

#if defined(IS_DEBUG_SLIL_COST_LOWER_BOUND)
  Logger::Info("SLIL Closur Static Lower Bound Cost is %llu for Dag %s",
               closureLowerBound, dataDepGraph_->GetDagID());
#endif

  // (Chris): A better lower bound can be computed by adding more to the SLIL
  // based on the instructions that use more than one register (defined by
  // different instructions).
  int commonUseLowerBound = closureLowerBound;
  std::vector<std::pair<const SchedInstruction *, Register *>> usedInsts;
  for (int i = 0; i < dataDepGraph_->GetInstCnt(); ++i) {
    const auto &inst = dataDepGraph_->GetInstByIndx(i);
    Register **usedRegisters = nullptr;
    auto usedRegCount = inst->GetUses(usedRegisters);

    // Get a list of instructions that define the registers, in array form.
    usedInsts.clear();
    for (int j = 0; j < usedRegCount; ++j) {
      Register *reg = usedRegisters[j];
      assert(reg->GetDefList().size() == 1 &&
             "Number of defs for register is not 1!");
      usedInsts.push_back(std::make_pair(*(reg->GetDefList().begin()), reg));
    }

#if defined(IS_DEBUG_SLIL_COMMON_USE_LB)
    Logger::Info("Common Use Lower Bound Instruction %d", inst->GetNum());
    Logger::Info("  Instruction %d uses:", inst->GetNum());
    for (const auto &p : usedInsts) {
      Logger::Info("    Instruction %d register %d:%d", p.first->GetNum(),
                   p.second->GetType(), p.second->GetNum());
    }

    for (const auto &p : usedInsts) {
      Logger::Info("  Live interval of Register %d:%d (defined by Inst %d):",
                   p.second->GetType(), p.second->GetNum(), p.first->GetNum());
      for (const auto &s : p.second->GetLiveInterval()) {
        Logger::Info("    %d", s->GetNum());
      }
    }
#endif

    for (size_t j = 0; j < usedInsts.size(); ++j) {
      for (size_t k = j + 1; k < usedInsts.size(); ++k) {
        const auto &jReg = usedInsts[j].second;
        const auto &kReg = usedInsts[k].second;

        // If k is not in the live interval of j AND ALSO j is not in the live
        // interval of k, add k to the live interval of j, and increment the
        // lower bound by 1.
        bool found = jReg->IsInInterval(usedInsts[k].first) ||
                     kReg->IsInInterval(usedInsts[j].first) ||
                     jReg->IsInPossibleInterval(usedInsts[k].first) ||
                     kReg->IsInPossibleInterval(usedInsts[j].first);

        if (!found && usedInsts[j].first != usedInsts[k].first) {
          jReg->AddToPossibleInterval(usedInsts[k].first);
          kReg->AddToPossibleInterval(usedInsts[j].first);

          commonUseLowerBound++;
#if defined(IS_DEBUG_SLIL_COMMON_USE_LB)
          Logger::Info("  Common Use: Found two instructions %d and %d",
                       usedInsts[j].first->GetNum(),
                       usedInsts[k].first->GetNum());
#endif
        }
      }
    }
  }

#if defined(IS_DEBUG_SLIL_COST_LOWER_BOUND)
  if (commonUseLowerBound > closureLowerBound)
    Logger::Info("SLIL Final  Static Lower Bound Cost is %llu for Dag %s",
                 commonUseLowerBound, dataDepGraph_->GetDagID());
#endif

  return static_cast<InstCount>(commonUseLowerBound);
}
/*****************************************************************************/

InstCount BBWithSpill::CmputCostLwrBound() {
  InstCount spillCostLwrBound = 0;

  if (GetSpillCostFunc() == SCF_SLIL) {
    spillCostLwrBound =
        ComputeSLILStaticLowerBound(regTypeCnt_, regFiles_, dataDepGraph_);
    dynamicSlilLowerBound_ = spillCostLwrBound;
    staticSlilLowerBound_ = spillCostLwrBound;
  }

  // for(InstCount i=0; i< dataDepGraph_->GetInstCnt(); i++) {
  //   inst = dataDepGraph_->GetInstByIndx(i);
  // }

  InstCount staticLowerBound =
      schedLwrBound_ * schedCostFactor_ + spillCostLwrBound * SCW_;

  // Add the minimum of the possible clusters to the lower bound
  if (IsSecondPass() && ClusterMemoryOperations) {
    staticLowerBound += MinClusterBlocks * ClusteringWeight;
  }

#if defined(IS_DEBUG_STATIC_LOWER_BOUND)
  Logger::Info(
      "DAG %s spillCostLB %d scFactor %d lengthLB %d lenFactor %d staticLB %d",
      dataDepGraph_->GetDagID(), spillCostLwrBound, SCW_, schedLwrBound_,
      schedCostFactor_, staticLowerBound);
#endif

  return staticLowerBound;
}
/*****************************************************************************/

void BBWithSpill::InitForSchdulng() {
  InitForCostCmputtn_();

  schduldEntryInstCnt_ = 0;
  schduldExitInstCnt_ = 0;
  schduldInstCnt_ = 0;
}
/*****************************************************************************/

void BBWithSpill::InitForCostCmputtn_() {
  if (IsSecondPass() && ClusterMemoryOperations)
    initForClustering();

  int i;

  crntCycleNum_ = 0;
  crntSlotNum_ = 0;
  crntSpillCost_ = 0;
  crntStepNum_ = -1;
  peakSpillCost_ = 0;
  totSpillCost_ = 0;

  for (i = 0; i < regTypeCnt_; i++) {
    regFiles_[i].ResetCrntUseCnts();
    regFiles_[i].ResetCrntLngths();
  }

  for (i = 0; i < regTypeCnt_; i++) {
    liveRegs_[i].Reset();
    if (regFiles_[i].GetPhysRegCnt() > 0)
      livePhysRegs_[i].Reset();
    //    if (chkCnflcts_)
    //      regFiles_[i].ResetConflicts();
    peakRegPressures_[i] = 0;
    regPressures_[i] = 0;
  }

  for (i = 0; i < dataDepGraph_->GetInstCnt(); i++)
    spillCosts_[i] = 0;

  for (auto &i : sumOfLiveIntervalLengths_)
    i = 0;

  dynamicSlilLowerBound_ = staticSlilLowerBound_;
}
/*****************************************************************************/

InstCount BBWithSpill::CmputNormCost_(InstSchedule *sched,
                                      COST_COMP_MODE compMode,
                                      InstCount &execCost, bool trackCnflcts) {
  InstCount cost = CmputCost_(sched, compMode, execCost, trackCnflcts);

  cost -= GetCostLwrBound();
  execCost -= GetCostLwrBound();

  sched->SetCost(cost);
  sched->SetExecCost(execCost);
  return cost;
}
/*****************************************************************************/

InstCount BBWithSpill::CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                                  InstCount &execCost, bool trackCnflcts) {
  if (compMode == CCM_STTC) {
    if (GetSpillCostFunc() == SCF_SPILLS) {
      LocalRegAlloc regAlloc(sched, dataDepGraph_);
      regAlloc.SetupForRegAlloc();
      regAlloc.AllocRegs();
      crntSpillCost_ = regAlloc.GetCost();
    }
  }

  assert(sched->IsComplete());
  InstCount cost = sched->GetCrntLngth() * schedCostFactor_;
  execCost = cost;
  cost += crntSpillCost_ * SCW_;
  // Add the current clustering cost
  if (IsSecondPass() && ClusterMemoryOperations) {
    cost += CurrentClusterCost * ClusteringWeight;
    assert(calculateClusterDLB() == CurrentClusterCost);
    sched->setClusterSize(CurrentClusterCost);
  }

  sched->SetSpillCosts(spillCosts_);
  sched->SetPeakRegPressures(peakRegPressures_);
  sched->SetSpillCost(crntSpillCost_);
  return cost;
}
/*****************************************************************************/

void BBWithSpill::CmputCrntSpillCost_() {
  switch (GetSpillCostFunc()) {
  case SCF_PERP:
  case SCF_PRP:
  case SCF_PEAK_PER_TYPE:
  case SCF_TARGET:
    crntSpillCost_ = peakSpillCost_;
    break;
  case SCF_SUM:
    crntSpillCost_ = totSpillCost_;
    break;
  case SCF_PEAK_PLUS_AVG:
    crntSpillCost_ =
        peakSpillCost_ + totSpillCost_ / dataDepGraph_->GetInstCnt();
    break;
  case SCF_SLIL:
    crntSpillCost_ = slilSpillCost_;
    break;
  default:
    crntSpillCost_ = peakSpillCost_;
    break;
  }
}
/*****************************************************************************/

void BBWithSpill::computeAndPrintClustering(InstSchedule *Sched) {
  InstCount instNum;
  InstCount cycleNum;
  InstCount slotNum;
  SchedInstruction *inst;
  bool temp = ClusterMemoryOperations;

  ClusterMemoryOperations = true;
  InitForCostCmputtn_();
  for (instNum = Sched->GetFrstInst(cycleNum, slotNum);
       instNum != INVALID_VALUE;
       instNum = Sched->GetNxtInst(cycleNum, slotNum)) {
    inst = dataDepGraph_->GetInstByIndx(instNum);
    SchdulInst(inst, cycleNum, slotNum, false);
  }
  printCurrentClustering();
  ClusterMemoryOperations = temp;
}

void BBWithSpill::saveCluster(SchedInstruction *inst) {
  if (LastCluster)
    // Save previous clusters in a vector except the last cluster
    // that we just exited out of.
    PastClustersList.push_back(std::move(LastCluster));

  // Last cluster that we just exited out of, used for fast accessing
  // to its contents.
  LastCluster = llvm::make_unique<PastClusters>(
      ClusterActiveGroup, CurrentClusterSize, inst->GetNum(), StartCycle);

  LastCluster->InstrList = std::move(InstrList);
}

void BBWithSpill::initCluster(SchedInstruction *inst) {
  ClusterActiveGroup = inst->GetClusterGroup();
  inst->SetActiveCluster(ClusterActiveGroup);
  CurrentClusterSize = 1;
  ClusterInstrRemainderCount[ClusterActiveGroup]--;
  InstrList = llvm::make_unique<llvm::SmallVector<llvm::StringRef, 4>>();
  InstrList->push_back(inst->GetName());
  ClusterCount[ClusterActiveGroup]++;
  CurrentClusterCost++;
}

void BBWithSpill::resetActiveCluster(SchedInstruction *inst) {
  ClusterActiveGroup = 0;
  inst->SetActiveCluster(0);
  CurrentClusterSize = 0;
}

void BBWithSpill::restorePreviousCluster(SchedInstruction *inst) {
  CurrentClusterSize = LastCluster->ClusterSize;
  ClusterActiveGroup = LastCluster->ClusterGroup;
  StartCycle = LastCluster->Start;
  inst->SetActiveCluster(ClusterActiveGroup);
  InstrList = std::move(LastCluster->InstrList);
  LastCluster.reset(); // Release current cluster pointer

  // Get previous cluster from vector list
  if (!PastClustersList.empty()) {
    LastCluster = std::move(PastClustersList.back());
    PastClustersList.pop_back();
  }
}

bool BBWithSpill::isClusterFinished() {
  assert(ClusterActiveGroup != 0);
  if (ClusterInstrRemainderCount[ClusterActiveGroup] == 0 ||
      CurrentClusterSize == MAX_INSTR_IN_CLUSTER) {
    return true;
  }
  return false;
}

int BBWithSpill::calculateClusterDLB() {
  int OptimisticLowerBound = 0;

  for (int begin = 1; begin <= ClusterGroupCount; begin++) {
    if (begin != ClusterActiveGroup)
      OptimisticLowerBound += std::ceil(
          double(ClusterInstrRemainderCount[begin]) / MAX_INSTR_IN_CLUSTER);
    else {
      // The amount of instructions remaining that the current open cluster can
      // add
      int AbsorbCount = MAX_INSTR_IN_CLUSTER - CurrentClusterSize;
      // Assume the current open cluster can add the max amount of instructions
      // that a cluster can contain.
      int Remainder = ClusterInstrRemainderCount[begin] - AbsorbCount;
      // If the remainder is negative then that indicates the open cluster can
      // absorb all of the remaining instructions.
      if (Remainder < 0)
        Remainder = 0;
      // Estimate the optimistic dynamic lower bound for the current cluster
      OptimisticLowerBound +=
          std::ceil(double(Remainder) / MAX_INSTR_IN_CLUSTER);
    }
  }
  return CurrentClusterCost + OptimisticLowerBound;
}

void BBWithSpill::UpdateSpillInfoForSchdul_(SchedInstruction *inst,
                                            bool trackCnflcts, int Start) {
  int16_t regType;
  int defCnt, useCnt, regNum, physRegNum;
  Register **defs, **uses;
  Register *def, *use;
  int liveRegs;
  InstCount newSpillCost;

  // Conditions for creating a cluster:
  // 1.) If a block is ended before it reaches 15 && there are remaining
  // instructions

  // Conditions for removing a cluster:
  // 1.) If the block is not 15 && there are remaining instructions

  // Scheduling cases for clustering project:
  // 1.) Same Cluster -> Same Cluster
  // If size == MAX_INSTR_IN_CLUSTER
  // Save cluster to restore
  // Set active to 0
  // 2.) Cluster -> Different Cluster
  // 3.) Non-Cluster -> Cluster
  // 4.) Cluster -> Non-Cluster

  // Possibly keep track of the current memory clustering size here
  // and in UpdateSpillInfoForUnSchdul_()
  if (IsSecondPass() && ClusterMemoryOperations) {
    // Check if the current instruction is part of a cluster
    if (inst->GetMayCluster()) {
      // Check if there is a current active cluster
      // A ClusterActiveGroup == 0 indicates that there is no currently active
      // clustering While ClusterActiveGroup != 0 indicates that there is active
      // clustering
      if (ClusterActiveGroup != 0) {
        // Check if the instruction is in the same cluster group as the active
        // cluster
        if (ClusterActiveGroup == inst->GetClusterGroup()) {
          // Case 1: Simple case where the current instruction is part of an
          // already active cluster.
          CurrentClusterSize++;
          ClusterInstrRemainderCount[ClusterActiveGroup]--;
          InstrList->push_back(inst->GetName());

          // If we reach the max amount for this cluster then save the cluster
          // and reset.
          if (isClusterFinished()) {
            saveCluster(inst);
            resetActiveCluster(inst);
          }
        } else {
          // Case 2: Else the instruction is part of different cluster that
          // is not currently active. Store information of the old cluster
          // group and start clustering for the new cluster.
          saveCluster(inst);

          // Finish setting up the new cluster
          initCluster(inst);
          StartCycle = Start;
        }
      } else {
        // Case 3: Not currently clustering. Initialize clustering
        initCluster(inst);
        StartCycle = Start;
      }
    } else if (ClusterActiveGroup != 0) {
      // Case 4: Exiting out of an active cluster
      // Save the cluster to restore when backtracking.
      saveCluster(inst);

      // Reset active cluster
      resetActiveCluster(inst);
    }
  }

  defCnt = inst->GetDefs(defs);
  useCnt = inst->GetUses(uses);

#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Updating reg pressure after scheduling Inst %d",
               inst->GetNum());
#endif

  // Update Live regs after uses
  for (int i = 0; i < useCnt; i++) {
    use = uses[i];
    regType = use->GetType();
    regNum = use->GetNum();
    physRegNum = use->GetPhysicalNumber();

    if (use->IsLive() == false)
      Logger::Fatal("Reg %d of type %d is used without being defined", regNum,
                    regType);

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Inst %d uses reg %d of type %d and %d uses", inst->GetNum(),
                 regNum, regType, use->GetUseCnt());
#endif

    use->AddCrntUse();

    if (use->IsLive() == false) {
      // (Chris): The SLIL calculation below the def and use for-loops doesn't
      // consider the last use of a register. Thus, an additional increment must
      // happen here.
      if (GetSpillCostFunc() == SCF_SLIL) {
        sumOfLiveIntervalLengths_[regType]++;
        if (!use->IsInInterval(inst) && !use->IsInPossibleInterval(inst)) {
          ++dynamicSlilLowerBound_;
        }
      }

      liveRegs_[regType].SetBit(regNum, false, use->GetWght());

#ifdef IS_DEBUG_REG_PRESSURE
      Logger::Info("Reg type %d now has %d live regs", regType,
                   liveRegs_[regType].GetOneCnt());
#endif

      if (regFiles_[regType].GetPhysRegCnt() > 0 && physRegNum >= 0)
        livePhysRegs_[regType].SetBit(physRegNum, false, use->GetWght());
    }
  }

  // Update Live regs after defs
  for (int i = 0; i < defCnt; i++) {
    def = defs[i];
    regType = def->GetType();
    regNum = def->GetNum();
    physRegNum = def->GetPhysicalNumber();

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Inst %d defines reg %d of type %d and %d uses",
                 inst->GetNum(), regNum, regType, def->GetUseCnt());
#endif

    // if (def->GetUseCnt() > 0) {

    if (trackCnflcts && liveRegs_[regType].GetOneCnt() > 0)
      regFiles_[regType].AddConflictsWithLiveRegs(
          regNum, liveRegs_[regType].GetOneCnt());

    liveRegs_[regType].SetBit(regNum, true, def->GetWght());

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Reg type %d now has %d live regs", regType,
                 liveRegs_[regType].GetOneCnt());
#endif

    if (regFiles_[regType].GetPhysRegCnt() > 0 && physRegNum >= 0)
      livePhysRegs_[regType].SetBit(physRegNum, true, def->GetWght());
    def->ResetCrntUseCnt();
    //}
  }

  newSpillCost = 0;

#ifdef IS_DEBUG_SLIL_CORRECT
  if (OPTSCHED_gPrintSpills) {
    Logger::Info(
        "Printing live range lengths for instruction BEFORE calculation.");
    for (int j = 0; j < sumOfLiveIntervalLengths_.size(); j++) {
      Logger::Info("SLIL for regType %d %s is currently %d", j,
                   sumOfLiveIntervalLengths_[j]);
    }
    Logger::Info("Now computing spill cost for instruction.");
  }
#endif

  for (int16_t i = 0; i < regTypeCnt_; i++) {
    liveRegs = liveRegs_[i].GetWghtedCnt();
    // Set current RP for register type "i"
    regPressures_[i] = liveRegs;
    // Update peak RP for register type "i"
    if (liveRegs > peakRegPressures_[i])
      peakRegPressures_[i] = liveRegs;

    // (Chris): Compute sum of live range lengths at this point
    if (GetSpillCostFunc() == SCF_SLIL) {
      sumOfLiveIntervalLengths_[i] += liveRegs_[i].GetOneCnt();
      for (int j = 0; j < liveRegs_[i].GetSize(); ++j) {
        if (liveRegs_[i].GetBit(j)) {
          const Register *reg = regFiles_[i].GetReg(j);
          if (!reg->IsInInterval(inst) && !reg->IsInPossibleInterval(inst)) {
            ++dynamicSlilLowerBound_;
          }
        }
      }
    }

    // FIXME: Can this be taken out of this loop?
    if (GetSpillCostFunc() == SCF_SLIL) {
      slilSpillCost_ = std::accumulate(sumOfLiveIntervalLengths_.begin(),
                                       sumOfLiveIntervalLengths_.end(), 0);
    }
  }

  if (GetSpillCostFunc() == SCF_TARGET) {
    newSpillCost = OST->getCost(regPressures_);

  } else if (GetSpillCostFunc() == SCF_SLIL) {
    slilSpillCost_ = std::accumulate(sumOfLiveIntervalLengths_.begin(),
                                     sumOfLiveIntervalLengths_.end(), 0);

  } else if (GetSpillCostFunc() == SCF_PRP) {
    newSpillCost =
        std::accumulate(regPressures_.begin(), regPressures_.end(), 0);

  } else if (GetSpillCostFunc() == SCF_PEAK_PER_TYPE) {
    for (int i = 0; i < regTypeCnt_; i++)
      newSpillCost +=
          std::max(0, peakRegPressures_[i] - machMdl_->GetPhysRegCnt(i));

  } else {
    // Default is PERP (Some SCF like SUM rely on PERP being the default here)
    int i = 0;
    std::for_each(
        regPressures_.begin(), regPressures_.end(), [&](InstCount RP) {
          newSpillCost += std::max(0, RP - machMdl_->GetPhysRegCnt(i++));
        });
  }

#ifdef IS_DEBUG_SLIL_CORRECT
  if (OPTSCHED_gPrintSpills) {
    Logger::Info(
        "Printing live range lengths for instruction AFTER calculation.");
    for (int j = 0; j < sumOfLiveIntervalLengths_.size(); j++) {
      Logger::Info("SLIL for regType %d is currently %d", j,
                   sumOfLiveIntervalLengths_[j]);
    }
  }
#endif

  crntStepNum_++;
  spillCosts_[crntStepNum_] = newSpillCost;

#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Spill cost at step  %d = %d", crntStepNum_, newSpillCost);
#endif

  totSpillCost_ += newSpillCost;

  peakSpillCost_ = std::max(peakSpillCost_, newSpillCost);

  CmputCrntSpillCost_();

  schduldInstCnt_++;
  if (inst->MustBeInBBEntry())
    schduldEntryInstCnt_++;
  if (inst->MustBeInBBExit())
    schduldExitInstCnt_++;
}
/*****************************************************************************/

void BBWithSpill::UpdateSpillInfoForUnSchdul_(SchedInstruction *inst) {
  int16_t regType;
  int i, defCnt, useCnt, regNum, physRegNum;
  Register **defs, **uses;
  Register *def, *use;
  bool isLive;

#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Updating reg pressure after unscheduling Inst %d",
               inst->GetNum());
#endif

  // Backtracking cases for clustering project:
  // 1.) Same Cluster <- Same Cluster
  // 2.) Non-Cluster <- Cluster
  // 3.) Different Cluster <- Cluster
  // 4.) Cluster <- Non-cluster
  if (IsSecondPass() && ClusterMemoryOperations) {
    // If the instruction we are backtracking from is part of a cluster
    if (inst->GetMayCluster()) {
      if (CurrentClusterSize != 0) {
        // Case 1, 2, and 3
        // Reduce the cluster size
        CurrentClusterSize--;
        ClusterInstrRemainderCount[ClusterActiveGroup]++;
        // Remove instruction's name from the list
        InstrList->pop_back();

        // Case 2: If there are no more instructions in the currently active
        // cluster then it indicates that we backtracked out of a cluster.
        if (CurrentClusterSize == 0) {
          ClusterCount[ClusterActiveGroup]--;
          assert(ClusterCount[ClusterActiveGroup] >= 0);
          CurrentClusterCost--;
          // Set active cluster to none.
          resetActiveCluster(inst);

          // Case 3: Check If this instruction ended another cluster
          if (LastCluster && LastCluster->InstNum == inst->GetNum()) {
            // If so, then we need to restore the state of the previous cluster
            restorePreviousCluster(inst);
          }
        }
      }
      // A cluster size of 0 while an instruction may cluster indicates that
      // the current instruction is at the end of a finished cluster
      else if (CurrentClusterSize == 0) {
        assert(inst->GetNum() == LastCluster->InstNum);
        restorePreviousCluster(inst);

        CurrentClusterSize--;
        ClusterInstrRemainderCount[ClusterActiveGroup]++;
        // Remove instruction's name from the list
        InstrList->pop_back();
      }
    } else if (LastCluster && LastCluster->InstNum == inst->GetNum()) {
      // Case 4: If there was a previous cluster and this instruction
      // ended the cluster then restore the previous cluster's state
      restorePreviousCluster(inst);
    }
  }

  defCnt = inst->GetDefs(defs);
  useCnt = inst->GetUses(uses);

  // (Chris): Update the SLIL for all live regs at this point.
  if (GetSpillCostFunc() == SCF_SLIL) {
    for (int i = 0; i < regTypeCnt_; ++i) {
      for (int j = 0; j < liveRegs_[i].GetSize(); ++j) {
        if (liveRegs_[i].GetBit(j)) {
          const Register *reg = regFiles_[i].GetReg(j);
          sumOfLiveIntervalLengths_[i]--;
          if (!reg->IsInInterval(inst) && !reg->IsInPossibleInterval(inst)) {
            --dynamicSlilLowerBound_;
          }
        }
      }
      assert(sumOfLiveIntervalLengths_[i] >= 0 &&
             "UpdateSpillInfoForUnSchdul_: SLIL negative!");
    }
  }

  // Update Live regs
  for (i = 0; i < defCnt; i++) {
    def = defs[i];
    regType = def->GetType();
    regNum = def->GetNum();
    physRegNum = def->GetPhysicalNumber();

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Inst %d defines reg %d of type %d and %d uses",
                 inst->GetNum(), regNum, regType, def->GetUseCnt());
#endif

    // if (def->GetUseCnt() > 0) {
    assert(liveRegs_[regType].GetBit(regNum));
    liveRegs_[regType].SetBit(regNum, false, def->GetWght());

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Reg type %d now has %d live regs", regType,
                 liveRegs_[regType].GetOneCnt());
#endif

    if (regFiles_[regType].GetPhysRegCnt() > 0 && physRegNum >= 0)
      livePhysRegs_[regType].SetBit(physRegNum, false, def->GetWght());
    def->ResetCrntUseCnt();
    //}
  }

  for (i = 0; i < useCnt; i++) {
    use = uses[i];
    regType = use->GetType();
    regNum = use->GetNum();
    physRegNum = use->GetPhysicalNumber();

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Inst %d uses reg %d of type %d and %d uses", inst->GetNum(),
                 regNum, regType, use->GetUseCnt());
#endif

    isLive = use->IsLive();
    use->DelCrntUse();
    assert(use->IsLive());

    if (isLive == false) {
      // (Chris): Since this was the last use, the above SLIL calculation didn't
      // take this instruction into account.
      if (GetSpillCostFunc() == SCF_SLIL) {
        sumOfLiveIntervalLengths_[regType]--;
        if (!use->IsInInterval(inst) && !use->IsInPossibleInterval(inst)) {
          --dynamicSlilLowerBound_;
        }
        assert(sumOfLiveIntervalLengths_[regType] >= 0 &&
               "UpdateSpillInfoForUnSchdul_: SLIL negative!");
      }
      liveRegs_[regType].SetBit(regNum, true, use->GetWght());

#ifdef IS_DEBUG_REG_PRESSURE
      Logger::Info("Reg type %d now has %d live regs", regType,
                   liveRegs_[regType].GetOneCnt());
#endif

      if (regFiles_[regType].GetPhysRegCnt() > 0 && physRegNum >= 0)
        livePhysRegs_[regType].SetBit(physRegNum, true, use->GetWght());
    }
  }

  schduldInstCnt_--;
  if (inst->MustBeInBBEntry())
    schduldEntryInstCnt_--;
  if (inst->MustBeInBBExit())
    schduldExitInstCnt_--;

  totSpillCost_ -= spillCosts_[crntStepNum_];
  crntStepNum_--;

#ifdef IS_DEBUG_REG_PRESSURE
// Logger::Info("Spill cost at step  %d = %d", crntStepNum_, newSpillCost);
#endif
}
/*****************************************************************************/

void BBWithSpill::SchdulInst(SchedInstruction *inst, InstCount cycleNum,
                             InstCount slotNum, bool trackCnflcts) {
  crntCycleNum_ = cycleNum;
  crntSlotNum_ = slotNum;
  if (inst == NULL)
    return;
  assert(inst != NULL);
  UpdateSpillInfoForSchdul_(inst, trackCnflcts, crntCycleNum_);
}
/*****************************************************************************/

void BBWithSpill::UnschdulInst(SchedInstruction *inst, InstCount cycleNum,
                               InstCount slotNum, EnumTreeNode *trgtNode) {
  if (slotNum == 0) {
    crntCycleNum_ = cycleNum - 1;
    crntSlotNum_ = machMdl_->GetIssueRate() - 1;
  } else {
    crntCycleNum_ = cycleNum;
    crntSlotNum_ = slotNum - 1;
  }

  if (inst == NULL) {
    return;
  }

  UpdateSpillInfoForUnSchdul_(inst);
  peakSpillCost_ = trgtNode->GetPeakSpillCost();
  CmputCrntSpillCost_();
}
/*****************************************************************************/

void BBWithSpill::FinishHurstc_() {

#ifdef IS_DEBUG_BBSPILL_COST
  stats::traceCostLowerBound.Record(costLwrBound_);
  stats::traceHeuristicCost.Record(hurstcCost_);
  stats::traceHeuristicScheduleLength.Record(hurstcSchedLngth_);
#endif
}
/*****************************************************************************/

void BBWithSpill::FinishOptml_() {
#ifdef IS_DEBUG_BBSPILL_COST
  stats::traceOptimalCost.Record(GetBestCost());
  stats::traceOptimalScheduleLength.Record(bestSchedLngth_);
#endif
}
/*****************************************************************************/

Enumerator *BBWithSpill::AllocEnumrtr_(Milliseconds timeout) {
  bool enblStallEnum = enblStallEnum_;
  bool ClusteringEnabled = IsSecondPass() && ClusterMemoryOperations;
  /*  if (!dataDepGraph_->IncludesUnpipelined()) {
      enblStallEnum = false;
    }*/

  enumrtr_ = new LengthCostEnumerator(
      dataDepGraph_, machMdl_, schedUprBound_, GetSigHashSize(),
      GetEnumPriorities(), GetPruningStrategy(), SchedForRPOnly_, enblStallEnum,
      timeout, GetSpillCostFunc(), ClusteringEnabled, 0, NULL);

  return enumrtr_;
}
/*****************************************************************************/

FUNC_RESULT BBWithSpill::Enumerate_(Milliseconds startTime,
                                    Milliseconds rgnTimeout,
                                    Milliseconds lngthTimeout) {
  InstCount trgtLngth;
  FUNC_RESULT rslt = RES_SUCCESS;
  int iterCnt = 0;
  int costLwrBound = 0;
  bool timeout = false;

  Milliseconds rgnDeadline, lngthDeadline;
  rgnDeadline =
      (rgnTimeout == INVALID_VALUE) ? INVALID_VALUE : startTime + rgnTimeout;
  lngthDeadline =
      (rgnTimeout == INVALID_VALUE) ? INVALID_VALUE : startTime + lngthTimeout;
  assert(lngthDeadline <= rgnDeadline);

  for (trgtLngth = schedLwrBound_; trgtLngth <= schedUprBound_; trgtLngth++) {
    InitForSchdulng();
    //#ifdef IS_DEBUG_ENUM_ITERS
    Logger::Info("Enumerating at target length %d", trgtLngth);
    //#endif
    rslt = enumrtr_->FindFeasibleSchedule(enumCrntSched_, trgtLngth, this,
                                          costLwrBound, lngthDeadline);
    if (rslt == RES_TIMEOUT)
      timeout = true;
    HandlEnumrtrRslt_(rslt, trgtLngth);

    if (GetBestCost() == 0 || rslt == RES_ERROR ||
        (lngthDeadline == rgnDeadline && rslt == RES_TIMEOUT)) {
      break;
    }

    enumrtr_->Reset();
    enumCrntSched_->Reset();

    CmputSchedUprBound_();

    iterCnt++;
    costLwrBound += 1;
    lngthDeadline = Utilities::GetProcessorTime() + lngthTimeout;
    if (lngthDeadline > rgnDeadline)
      lngthDeadline = rgnDeadline;
  }

#ifdef IS_DEBUG_ITERS
  stats::iterations.Record(iterCnt);
  stats::enumerations.Record(enumrtr_->GetSearchCnt());
  stats::lengths.Record(iterCnt);
#endif

  // Failure to find a feasible sched. in the last iteration is still
  // considered an overall success
  if (rslt == RES_SUCCESS || rslt == RES_FAIL) {
    rslt = RES_SUCCESS;
  }
  if (timeout)
    rslt = RES_TIMEOUT;

  return rslt;
}
/*****************************************************************************/

InstCount BBWithSpill::UpdtOptmlSched(InstSchedule *crntSched,
                                      LengthCostEnumerator *) {
  InstCount crntCost;
  InstCount crntExecCost;

  //  crntCost = CmputNormCost_(crntSched, CCM_DYNMC, crntExecCost, false);
  crntCost = CmputNormCost_(crntSched, CCM_STTC, crntExecCost, false);

  //#ifdef IS_DEBUG_SOLN_DETAILS_2
  Logger::Info(
      "Found a feasible sched. of length %d, spill cost %d and tot cost %d",
      crntSched->GetCrntLngth(), crntSched->GetSpillCost(), crntCost);
  //  crntSched->Print(Logger::GetLogStream(), "New Feasible Schedule");
  //#endif

  if (crntCost < GetBestCost()) {

    if (crntSched->GetCrntLngth() > schedLwrBound_)
      Logger::Info("$$$ GOOD_HIT: Better spill cost for a longer schedule");

    SetBestCost(crntCost);
    if (IsSecondPass() && ClusterMemoryOperations)
      setBestClusterCost(CurrentClusterCost);
    optmlSpillCost_ = crntSpillCost_;
    SetBestSchedLength(crntSched->GetCrntLngth());
    enumBestSched_->Copy(crntSched);
    bestSched_ = enumBestSched_;
  }

  return GetBestCost();
}

void BBWithSpill::printCurrentClustering() {
  // Print the instructions in the clusters after finding a schedule.
  if (IsSecondPass() && ClusterMemoryOperations) {
    dbgs() << "Printing clustered instructions:\n";
    int i = 1;
    for (const auto &clusters : PastClustersList) {
      dbgs() << "Printing cluster " << i << ", start cycle (" << clusters->Start
             << "): ";
      for (const auto &instr : *clusters->InstrList) {
        dbgs() << instr << " ";
      }
      i++;
      dbgs() << '\n';
    }

    if (LastCluster) {
      dbgs() << "Printing cluster " << i << ", start cycle ("
             << LastCluster->Start << "): ";
      for (const auto &instr : *(LastCluster->InstrList)) {
        dbgs() << instr << " ";
      }
      i++;
      dbgs() << '\n';
    }

    if (InstrList && InstrList->size() > 0) {
      dbgs() << "Printing cluster " << i << ", start cycle (" << StartCycle
             << "): ";
      for (const auto &instr : *InstrList) {
        dbgs() << instr << " ";
      }
      dbgs() << '\n';
    }
  }
}
/*****************************************************************************/

void BBWithSpill::SetupForSchdulng_() {
  for (int i = 0; i < regTypeCnt_; i++) {
    liveRegs_[i].Construct(regFiles_[i].GetRegCnt());
  }

  SetupPhysRegs_();

  entryInstCnt_ = dataDepGraph_->GetEntryInstCnt();
  exitInstCnt_ = dataDepGraph_->GetExitInstCnt();
  schduldEntryInstCnt_ = 0;
  schduldExitInstCnt_ = 0;

  /*
  if (chkCnflcts_)
    for (int i = 0; i < regTypeCnt_; i++) {
      regFiles_[i].SetupConflicts();
    }
 */
}
/*****************************************************************************/

bool BBWithSpill::ChkCostFsblty(InstCount trgtLngth, EnumTreeNode *node) {
  bool fsbl = true;
  InstCount crntCost, dynmcCostLwrBound;
  int ClusterDynamicLowerBound;
  if (GetSpillCostFunc() == SCF_SLIL) {
    crntCost = dynamicSlilLowerBound_ * SCW_ + trgtLngth * schedCostFactor_;
  } else {
    crntCost = crntSpillCost_ * SCW_ + trgtLngth * schedCostFactor_;
  }
  // Add the cost of clustering
  if (IsSecondPass() && ClusterMemoryOperations) {
    ClusterDynamicLowerBound = calculateClusterDLB();
    crntCost += ClusterDynamicLowerBound * ClusteringWeight;
  }

  crntCost -= GetCostLwrBound();
  dynmcCostLwrBound = crntCost;

  // assert(cost >= 0);
  assert(dynmcCostLwrBound >= 0);

  /*
    if (IsSecondPass() && ClusterMemoryOperations) {
      dbgs() << "Current cycle: " << node->GetTime() <<", current cost is: " <<
    dynmcCostLwrBound << ". Current best is: " << GetBestCost() << '\n';
      printCurrentClustering();
    }
  */

  fsbl = dynmcCostLwrBound < GetBestCost();

  // FIXME: RP tracking should be limited to the current SCF. We need RP
  // tracking interface.
  if (fsbl) {
    node->SetCost(crntCost);
    node->SetCostLwrBound(dynmcCostLwrBound);
    node->SetPeakSpillCost(peakSpillCost_);
    node->SetSpillCostSum(totSpillCost_);
    if (IsSecondPass() && ClusterMemoryOperations) {
      node->setClusteringCost(CurrentClusterCost);
      node->setCurClusteringGroup(ClusterActiveGroup);
      node->setClusterLwrBound(ClusterDynamicLowerBound);
      if (ClusterActiveGroup != 0) {
        node->setClusterAbsorbCount(15 - CurrentClusterSize);
      } else {
        node->setClusterAbsorbCount(0);
      }
    }
  }
  return fsbl;
}
/*****************************************************************************/

void BBWithSpill::SetSttcLwrBounds(EnumTreeNode *) {
  // Nothing.
}

/*****************************************************************************/

bool BBWithSpill::ChkInstLglty(SchedInstruction *inst) {
  return true;
  /*
  int16_t regType;
  int defCnt, physRegNum;
  Register **defs;
  Register *def, *liveDef;

#ifdef IS_DEBUG_CHECK
  Logger::Info("Checking inst %d %s", inst->GetNum(), inst->GetOpCode());
#endif

  if (fixLivein_) {
    if (inst->MustBeInBBEntry() == false &&
        schduldEntryInstCnt_ < entryInstCnt_)
      return false;
  }

  if (fixLiveout_) {
    if (inst->MustBeInBBExit() == true &&
        schduldInstCnt_ < (dataDepGraph_->GetInstCnt() - exitInstCnt_))
      return false;
  }

  defCnt = inst->GetDefs(defs);

  // Update Live regs
  for (int i = 0; i < defCnt; i++) {
    def = defs[i];
    regType = def->GetType();
    physRegNum = def->GetPhysicalNumber();

    // If this is a physical register definition and another
    // definition of the same physical register is live, then
    // scheduling this instruction is illegal unless this
    // instruction is the last use of that physical reg definition.
    if (regFiles_[regType].GetPhysRegCnt() > 0 && physRegNum >= 0 &&
        livePhysRegs_[regType].GetBit(physRegNum) == true) {

      liveDef = regFiles_[regType].FindLiveReg(physRegNum);
      assert(liveDef != NULL);

      // If this instruction is the last use of the current live def
      if (liveDef->GetCrntUseCnt() + 1 == liveDef->GetUseCnt() &&
          inst->FindUse(liveDef) == true)
        return true;
      else
        return false;
    } // end if
  }   // end for
  return true;
  */
}

bool BBWithSpill::ChkSchedule_(InstSchedule *bestSched,
                               InstSchedule *lstSched) {
  return true;
  /*
  if (bestSched == NULL || bestSched == lstSched)
    return true;
  if (chkSpillCostSum_) {

    InstCount i, heurLarger = 0, bestLarger = 0;
    for (i = 0; i < dataDepGraph_->GetInstCnt(); i++) {
      if (lstSched->GetSpillCost(i) > bestSched->GetSpillCost(i))
        heurLarger++;
      if (bestSched->GetSpillCost(i) > lstSched->GetSpillCost(i))
        bestLarger++;
    }
    Logger::Info("Heuristic spill cost is larger at %d points, while best "
                 "spill cost is larger at %d points",
                 heurLarger, bestLarger);
    if (bestSched->GetTotSpillCost() > lstSched->GetTotSpillCost()) {
      // Enumerator's best schedule has a greater spill cost sum than the
      // heuristic
      // This can happen if we are using a cost function other than the spill
      // cost sum function
      Logger::Info("??? Heuristic sched has a smaller spill cost sum than best "
                   "sched, heur : %d, best : %d. ",
                   lstSched->GetTotSpillCost(), bestSched->GetTotSpillCost());
      if (lstSched->GetCrntLngth() <= bestSched->GetCrntLngth()) {
        Logger::Info("Taking heuristic schedule");
        bestSched->Copy(lstSched);
        return false;
      }
    }
  }
  if (chkCnflcts_) {
    CmputCnflcts_(lstSched);
    CmputCnflcts_(bestSched);

#ifdef IS_DEBUG_CONFLICTS
    Logger::Info("Heuristic conflicts : %d, best conflicts : %d. ",
                 lstSched->GetConflictCount(), bestSched->GetConflictCount());
#endif

    if (bestSched->GetConflictCount() > lstSched->GetConflictCount()) {
      // Enumerator's best schedule causes more conflicst than the heuristic
      // schedule.
      Logger::Info("??? Heuristic sched causes fewer conflicts than best "
                   "sched, heur : %d, best : %d. ",
                   lstSched->GetConflictCount(), bestSched->GetConflictCount());
      if (lstSched->GetCrntLngth() <= bestSched->GetCrntLngth()) {
        Logger::Info("Taking heuristic schedule");
        bestSched->Copy(lstSched);
        return false;
      }
    }
  }
  return true;
  */
}

void BBWithSpill::CmputCnflcts_(InstSchedule *sched) {
  int cnflctCnt = 0;
  InstCount execCost;

  CmputNormCost_(sched, CCM_STTC, execCost, true);
  for (int i = 0; i < regTypeCnt_; i++) {
    cnflctCnt += regFiles_[i].GetConflictCnt();
  }
  sched->SetConflictCount(cnflctCnt);
}
