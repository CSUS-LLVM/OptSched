#include "opt-sched/Scheduler/bb_spill.h"
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
#include "opt-sched/Scheduler/dev_defines.h"
#include "Wrapper/AMDGPU/OptSchedGCNTarget.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>
#include <hip/hip_runtime.h>

extern bool OPTSCHED_gPrintSpills; 

using namespace llvm::opt_sched;

// The denominator used when calculating cost weight.
static const int COST_WGHT_BASE = 100;

BBWithSpill::BBWithSpill(const OptSchedTarget *OST_, DataDepGraph *dataDepGraph,
                         long rgnNum, int16_t sigHashSize, LB_ALG lbAlg,
                         SchedPriorities hurstcPrirts,
                         SchedPriorities enumPrirts, bool vrfySched,
                         Pruning PruningStrategy, bool SchedForRPOnly,
                         bool enblStallEnum, int SCW,
                         SPILL_COST_FUNCTION spillCostFunc,
                         SchedulerType HeurSchedType, MachineModel *dev_machMdl,
                         SchedPriorities acoPrirts1, SchedPriorities acoPrirts2)
    : SchedRegion(OST_->MM, dev_machMdl, dataDepGraph, rgnNum, sigHashSize, lbAlg,
                  hurstcPrirts, enumPrirts, vrfySched, PruningStrategy,
                  HeurSchedType, acoPrirts1, acoPrirts2, spillCostFunc),
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
  NeedsComputeSLIL = (spillCostFunc == SCF_SLIL);
  needsComputeTarget = (spillCostFunc == SCF_TARGET);

  regTypeCnt_ = OST->MM->GetRegTypeCnt();
  MaxOccLDS_ = ((OptSchedGCNTarget *) OST)->getMaxOccLDS();
  TargetOccupancy_ = ((OptSchedGCNTarget *) OST)->getTargetOccupancy();
  regFiles_ = dataDepGraph->getRegFiles(); 
  liveRegs_ = new WeightedBitVector[regTypeCnt_];
  livePhysRegs_ = new WeightedBitVector[regTypeCnt_];
  spillCosts_ = new InstCount[dataDepGraph_->GetInstCnt()];
  peakRegPressures_ = new InstCount[regTypeCnt_];
  regPressures_.resize(regTypeCnt_);
  sumOfLiveIntervalLengths_ = new int[regTypeCnt_];

  //initialize all values to 0
  for (int i = 0; i < regTypeCnt_; i++)
    sumOfLiveIntervalLengths_[i] = 0;

  entryInstCnt_ = 0;
  exitInstCnt_ = 0;
  schduldEntryInstCnt_ = 0;
  schduldExitInstCnt_ = 0;
  schduldInstCnt_ = 0;
}
/****************************************************************************/

BBWithSpill::~BBWithSpill() {
  if (enumrtr_ != NULL) {
    delete enumrtr_;
  }
 
  delete[] sumOfLiveIntervalLengths_;
  delete[] liveRegs_;
  delete[] livePhysRegs_;
  delete[] spillCosts_;
  delete[] peakRegPressures_;
}

/*****************************************************************************/

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
      for (const auto &instNum : reg->GetDefList()) {
        if (reg->AddToInterval(dataDepGraph_->GetInstByIndx(instNum))) {
          ++naiveLowerBound;
        }
      }
      for (const auto &instNum : reg->GetUseList()) {
        if (reg->AddToInterval(dataDepGraph_->GetInstByIndx(instNum))) {
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
    RegIndxTuple *definedRegisters = nullptr;
    auto defRegCount = inst->GetDefs(definedRegisters);
    auto recSuccBV = inst->GetRcrsvNghbrBitVector(DIR_FRWRD);
    for (int j = 0; j < defRegCount; ++j) {
      for (const auto &dependentInstNum : 
           dataDepGraph_->getRegByTuple(&definedRegisters[j])->GetUseList()) {
        auto recPredBV = const_cast<SchedInstruction *>(
			 dataDepGraph_->GetInstByIndx(dependentInstNum))
                             ->GetRcrsvNghbrBitVector(DIR_BKWRD);
        assert(recSuccBV->GetSize() == recPredBV->GetSize() &&
               "Successor list size doesn't match predecessor list size!");
        for (int k = 0; k < recSuccBV->GetSize(); ++k) {
          if (recSuccBV->GetBit(k) & recPredBV->GetBit(k)) {
            if (dataDepGraph_->getRegByTuple(&definedRegisters[j])->
		AddToInterval(dataDepGraph_->GetInstByIndx(k))) {
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
  std::vector<std::pair<const SchedInstruction *, llvm::opt_sched::Register *>> usedInsts;
  for (int i = 0; i < dataDepGraph_->GetInstCnt(); ++i) {
    const auto &inst = dataDepGraph_->GetInstByIndx(i);
    RegIndxTuple *usedRegisters = nullptr;
    auto usedRegCount = inst->GetUses(usedRegisters);

    // Get a list of instructions that define the registers, in array form.
    usedInsts.clear();
    for (int j = 0; j < usedRegCount; ++j) {
      llvm::opt_sched::Register *reg = dataDepGraph_->getRegByTuple(&usedRegisters[j]);
      assert(reg->GetDefList().size() == 1 &&
             "Number of defs for register is not 1!");
      usedInsts.push_back(std::make_pair(
	  dataDepGraph_->GetInstByIndx(*(reg->GetDefList().begin())), reg));
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
  InstCount staticLowerBound = CmputExecCostLwrBound() + CmputRPCostLwrBound();

#if defined(IS_DEBUG_STATIC_LOWER_BOUND)
  Logger::Info(
      "DAG %s spillCostLB %d scFactor %d lengthLB %d lenFactor %d staticLB %d",
      dataDepGraph_->GetDagID(), spillCostLwrBound, SCW_, schedLwrBound_,
      schedCostFactor_, staticLowerBound);
#endif

  return staticLowerBound;
}

InstCount BBWithSpill::CmputExecCostLwrBound() {
  ExecCostLwrBound_ = schedLwrBound_ * schedCostFactor_;
  return ExecCostLwrBound_;
}

InstCount BBWithSpill::CmputRPCostLwrBound() {
  InstCount spillCostLwrBound = 0;

  if (GetSpillCostFunc() == SCF_SLIL) {
    spillCostLwrBound =
        ComputeSLILStaticLowerBound(regTypeCnt_, regFiles_, dataDepGraph_);
    dynamicSlilLowerBound_ = spillCostLwrBound;
    staticSlilLowerBound_ = spillCostLwrBound;
  }

  RpCostLwrBound_ = spillCostLwrBound * SCW_;
  return RpCostLwrBound_;
}

/*****************************************************************************/

__host__ __device__
void BBWithSpill::InitForSchdulng() {
  InitForCostCmputtn_();

  schduldEntryInstCnt_ = 0;
  schduldExitInstCnt_ = 0;
  schduldInstCnt_ = 0;
}

//device version of InitForSchdulng(), bypasses needles polymorphism that
//prevents it from being called on device
__device__
void BBWithSpill::Dev_InitForSchdulng() {
  InitForCostCmputtn_();

  dev_schduldInstCnt_[GLOBALTID] = 0;
}

/*****************************************************************************/

__host__ __device__
void BBWithSpill::InitForCostCmputtn_() {
  int i;
#ifdef __HIP_DEVICE_COMPILE__ // Device version
  dev_crntCycleNum_[GLOBALTID] = 0;
  dev_crntSlotNum_[GLOBALTID] = 0;
  dev_crntSpillCost_[GLOBALTID] = 0;
  dev_crntStepNum_[GLOBALTID] = -1;
  dev_peakSpillCost_[GLOBALTID] = 0;
  dev_totSpillCost_[GLOBALTID] = 0;

  for (i = 0; i < regTypeCnt_; i++) {
    regFiles_[i].ResetCrntUseCnts();
    regFiles_[i].ResetCrntLngths();
  }

  for (i = 0; i < regTypeCnt_; i++) {
    dev_liveRegs_[i][GLOBALTID].Dev_Reset();
    
    dev_peakRegPressures_[i*numThreads_+GLOBALTID] = 0;
    dev_regPressures_[i*numThreads_+GLOBALTID] = 0;
  }
  
  int _instCnt = dataDepGraph_->GetInstCnt();
  for (i = 0; i < _instCnt; i++)
    dev_spillCosts_[i*numThreads_+GLOBALTID] = 0;
  if (needsSLIL()) {
    for (int i = 0; i < regTypeCnt_; i++)
      dev_sumOfLiveIntervalLengths_[i*numThreads_+GLOBALTID] = 0;

    dev_dynamicSlilLowerBound_[GLOBALTID] = staticSlilLowerBound_;
  }

#else // Host version
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

    if (regFiles_[i].GetPhysRegCnt() > 0) {
      livePhysRegs_[i].Reset();
    }

    peakRegPressures_[i] = 0;
    regPressures_[i] = 0;
  }

  for (i = 0; i < dataDepGraph_->GetInstCnt(); i++)
    spillCosts_[i] = 0;

  for (int i = 0; i < regTypeCnt_; i++)
    sumOfLiveIntervalLengths_[i] = 0;

  dynamicSlilLowerBound_ = staticSlilLowerBound_;
#endif
}
/*****************************************************************************/

InstCount BBWithSpill::CmputNormCost_(InstSchedule *sched,
                                      COST_COMP_MODE compMode,
                                      InstCount &execCost, bool trackCnflcts) {
  InstCount cost = CmputCost_(sched, compMode, execCost, trackCnflcts);

  cost -= GetCostLwrBound();
  execCost -= GetExecCostLwrBound();

  sched->SetCost(cost);
  sched->SetExecCost(execCost);
  sched->SetNormSpillCost(sched->GetSpillCost() * SCW_ - GetRPCostLwrBound());
  return cost;
}

__device__
InstCount BBWithSpill::Dev_CmputNormCost_(InstSchedule *sched, 
		                          COST_COMP_MODE compMode,
                                          InstCount &execCost, 
					  bool trackCnflcts) {
  InstCount cost = Dev_CmputCost_(sched, compMode, execCost, trackCnflcts);

  cost -= GetCostLwrBound();
  execCost -= GetExecCostLwrBound();

  sched->SetCost(cost);
  sched->SetExecCost(execCost);
  sched->SetNormSpillCost(sched->GetSpillCost() * SCW_ - GetRPCostLwrBound());
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
  sched->SetSpillCosts(spillCosts_);
  sched->SetPeakRegPressures(peakRegPressures_);
  sched->SetSpillCost(crntSpillCost_);
  return cost;
}

__device__
InstCount BBWithSpill::Dev_CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                                  InstCount &execCost, bool trackCnflcts) {
  if (compMode == CCM_STTC) {
    //printf("***** Device called with CCM_STTC *****\n");
/*  TODO:port to device if possible
    // LocalRegAlloc has not been ported to device
    if (GetSpillCostFunc() == SCF_SPILLS) {
      LocalRegAlloc regAlloc(sched, dataDepGraph_);
      regAlloc.SetupForRegAlloc();
      regAlloc.AllocRegs();
      crntSpillCost_ = regAlloc.GetCost();
    }
*/
  }

  assert(sched->IsComplete());
  InstCount cost = sched->GetCrntLngth() * schedCostFactor_;
  execCost = cost;
  cost += dev_crntSpillCost_[GLOBALTID] * SCW_;
  sched->Dev_SetSpillCosts(dev_spillCosts_);
  sched->Dev_SetPeakRegPressures(dev_peakRegPressures_);
  sched->SetSpillCost(dev_crntSpillCost_[GLOBALTID]);
  return cost;
}
/*****************************************************************************/

__host__ __device__
void BBWithSpill::CmputCrntSpillCost_() {
#ifdef __HIP_DEVICE_COMPILE__ //Device version of function
  switch (GetSpillCostFunc()) {
  case SCF_PERP:
  case SCF_PRP:
  case SCF_PEAK_PER_TYPE:
  case SCF_TARGET:
    dev_crntSpillCost_[GLOBALTID] = dev_peakSpillCost_[GLOBALTID];
    break;
  case SCF_SUM:
    dev_crntSpillCost_[GLOBALTID] = dev_totSpillCost_[GLOBALTID];
    break;
  case SCF_PEAK_PLUS_AVG:
    dev_crntSpillCost_[GLOBALTID] =
        dev_peakSpillCost_[GLOBALTID] + 
	dev_totSpillCost_[GLOBALTID] / dataDepGraph_->GetInstCnt();
    break;
  case SCF_SLIL:
    dev_crntSpillCost_[GLOBALTID] = dev_slilSpillCost_[GLOBALTID];
    break;
  default:
    dev_crntSpillCost_[GLOBALTID] = dev_peakSpillCost_[GLOBALTID];
    break;
  }

#else // Host version of function
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
#endif
}
/******************************i***********************************************/

//#define IS_DEBUG_REG_PRESSURE
//note: Logger::info/fatal cannot be called on device. using __HIP_DEVICE_COMPILE__ 
//macro to call printf on device instead
__host__ __device__
void BBWithSpill::UpdateSpillInfoForSchdul_(SchedInstruction *inst,
                                            bool trackCnflcts) {
  int16_t regType;
  int defCnt, useCnt, regNum, physRegNum;
  RegIndxTuple *defs, *uses;
  Register *def, *use;
  int liveRegs;
  InstCount newSpillCost;
  InstCount perpValueForSlil;

#ifdef __HIP_DEVICE_COMPILE__ // Device Version of function
#ifdef IS_DEBUG_REG_PRESSURE
  printf("Updating reg pressure after scheduling Inst %d\n",
               inst->GetNum());
#endif

  defCnt = inst->GetDefCnt();
  useCnt = inst->GetUseCnt();

  // Update Live regs after uses
  // TODO(bruce): convert to dev uses
  int useStart = inst->ddgUseIndex;
  for (int i = 0; i < useCnt; i++) {
    use = dataDepGraph_->getRegByTuple(dataDepGraph_->getUseByIndex(useStart + i));
    regType = use->GetType();
    regNum = use->GetNum();
    physRegNum = use->GetPhysicalNumber();

    if (use->IsLive() == false) {
      printf("Reg %d of type %d is used without being defined\n", regNum,
             regType);
    }

#ifdef IS_DEBUG_REG_PRESSURE
    printf("Inst %d uses reg %d of type %d and %d uses\n", inst->GetNum(), 
		                        regNum, regType, use->GetUseCnt());
#endif

    use->AddCrntUse();

    if (use->IsLive() == false) {
      // (Chris): The SLIL calculation below the def and use for-loops doesn't
      // consider the last use of a register. Thus, an additional increment must
      // happen here.
      if (needsSLIL()) {
        dev_sumOfLiveIntervalLengths_[regType*numThreads_+GLOBALTID]++;
      }

      dev_liveRegs_[regType][GLOBALTID].SetBit(regNum, false, use->GetWght());

#ifdef IS_DEBUG_REG_PRESSURE
      printf("Reg type %d now has %d live regs\n", regType,
             dev_liveRegs_[regType][GLOBALTID].GetOneCnt());
#endif
    }
  }

  // Update Live regs after defs
  // TODO(bruce): convert to dev defs
  int defStart = inst->ddgDefIndex;
  for (int i = 0; i < defCnt; i++) {
    def = dataDepGraph_->getRegByTuple(dataDepGraph_->getDefByIndex(defStart + i));
    regType = def->GetType();
    regNum = def->GetNum();
    physRegNum = def->GetPhysicalNumber();

#ifdef IS_DEBUG_REG_PRESSURE
    printf("Inst %d defines reg %d of type %d and %d uses\n",inst->GetNum(),
           regNum, regType, def->GetUseCnt());
#endif

    if (trackCnflcts && dev_liveRegs_[regType][GLOBALTID].GetOneCnt() > 0)
      regFiles_[regType].AddConflictsWithLiveRegs(
          regNum, dev_liveRegs_[regType][GLOBALTID].GetOneCnt());

    dev_liveRegs_[regType][GLOBALTID].SetBit(regNum, true, def->GetWght());

#ifdef IS_DEBUG_REG_PRESSURE
    printf("Reg type %d now has %d live regs\n", regType,
           dev_liveRegs_[regType][GLOBALTID].GetOneCnt());
#endif

    def->ResetCrntUseCnt();
  }

  newSpillCost = 0;

#ifdef IS_DEBUG_SLIL_CORRECT
  if (OPTSCHED_gPrintSpills) {
    printf("Printing live range lengths for instruction BEFORE calculation.\n");
    for (int j = 0; j < regTypeCnt_; j++) {
      printf("SLIL for regType %d %s is currently %d\n", j,
             dev_sumOfLiveIntervalLengths_[j*numThreads_+GLOBALTID]);
    }
    printf("Now computing spill cost for instruction.\n");
  }
#endif

  for (int16_t i = 0; i < regTypeCnt_; i++) {
    liveRegs = dev_liveRegs_[i][GLOBALTID].GetWghtedCnt();
    // Set current RP for register type "i"
    dev_regPressures_[i*numThreads_+GLOBALTID] = liveRegs;
    // Update peak RP for register type "i"
    if (liveRegs > dev_peakRegPressures_[i*numThreads_+GLOBALTID])
      dev_peakRegPressures_[i*numThreads_+GLOBALTID] = liveRegs;

    // (Chris): Compute sum of live range lengths at this point
    if (needsSLIL()) {
      dev_sumOfLiveIntervalLengths_[i*numThreads_+GLOBALTID] += 
	               dev_liveRegs_[i][GLOBALTID].GetOneCnt();
    }
  }

  if (GetSpillCostFunc() == SCF_SLIL) {
    dev_slilSpillCost_[GLOBALTID] = 
                   Dev_CmputCostForFunction(GetSpillCostFunc());
    // calculate PERP with SLIL to consider schedules with PERP of 0
    // even if SLIL is higher
    perpValueForSlil = Dev_CmputCostForFunction(SCF_PERP);
    if (dev_peakSpillCost_[GLOBALTID] < perpValueForSlil)
      dev_peakSpillCost_[GLOBALTID] = perpValueForSlil;
  }
  else
    newSpillCost = Dev_CmputCostForFunction(GetSpillCostFunc());

#ifdef IS_DEBUG_SLIL_CORRECT
  if (OPTSCHED_gPrintSpills) {
    printf("Printing live range lengths for instruction AFTER calculation.\n");
    for (int j = 0; j < regTypeCnt_; j++) {
      printf("SLIL for regType %d is currently %d\n", j,
             dev_sumOfLiveIntervalLengths_[j*numThreads_+GLOBALTID]);
    }
  }
#endif

  dev_crntStepNum_[GLOBALTID]++;
  dev_spillCosts_[dev_crntStepNum_[GLOBALTID]*numThreads_+GLOBALTID] = newSpillCost;

#ifdef IS_DEBUG_REG_PRESSURE
  printf("Spill cost at step  %d = %d\n", dev_crntStepNum_[GLOBALTID], newSpillCost);
#endif

  dev_totSpillCost_[GLOBALTID] += newSpillCost;

  if (dev_peakSpillCost_[GLOBALTID] < newSpillCost)
    dev_peakSpillCost_[GLOBALTID] = newSpillCost;

  CmputCrntSpillCost_();

  dev_schduldInstCnt_[GLOBALTID]++;

#else // Host Version of function
#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Updating reg pressure after scheduling Inst %d",
               inst->GetNum());
#endif

  defCnt = inst->GetDefs(defs);
  useCnt = inst->GetUses(uses);

  // Update Live regs after uses
  for (int i = 0; i < useCnt; i++) {
    use = dataDepGraph_->getRegByTuple(&uses[i]);
    regType = use->GetType();
    regNum = use->GetNum();
    physRegNum = use->GetPhysicalNumber();

    if (use->IsLive() == false) {
      Logger::Fatal("Reg %d of type %d is used without being defined", regNum,
                    regType);
    }

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Inst %d uses reg %d of type %d and %d uses", inst->GetNum(),
                 regNum, regType, use->GetUseCnt());
#endif

    use->AddCrntUse();

    if (use->IsLive() == false) {
      // (Chris): The SLIL calculation below the def and use for-loops doesn't
      // consider the last use of a register. Thus, an additional increment must
      // happen here.
      if (needsSLIL()) {
        sumOfLiveIntervalLengths_[regType]++;
        // if (!use->IsInInterval(inst) && !use->IsInPossibleInterval(inst)) {
        //   ++dynamicSlilLowerBound_;
        // }
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
    def = dataDepGraph_->getRegByTuple(&defs[i]);
    regType = def->GetType();
    regNum = def->GetNum();
    physRegNum = def->GetPhysicalNumber();

#ifdef IS_DEBUG_REG_PRESSURE
    Logger::Info("Inst %d defines reg %d of type %d and %d uses",
                 inst->GetNum(), regNum, regType, def->GetUseCnt());
#endif

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
  }

  newSpillCost = 0;

#ifdef IS_DEBUG_SLIL_CORRECT
  if (OPTSCHED_gPrintSpills) {
    Logger::Info(
        "Printing live range lengths for instruction BEFORE calculation.");
    for (int j = 0; j < regTypeCnt_; j++) {
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
    if (needsSLIL()) {
      sumOfLiveIntervalLengths_[i] += liveRegs_[i].GetOneCnt();
      // for (int j = 0; j < liveRegs_[i].GetSize(); ++j) {
      //   if (liveRegs_[i].GetBit(j)) {
      //     const Register *reg = regFiles_[i].GetReg(j);
      //     if (!reg->IsInInterval(inst) && !reg->IsInPossibleInterval(inst)) {
      //       ++dynamicSlilLowerBound_;
      //     }
      //   }
      // }
    }
  }
  
  if (GetSpillCostFunc() == SCF_SLIL) {
    slilSpillCost_ = CmputCostForFunction(GetSpillCostFunc());
    // calculate PERP with SLIL to consider schedules with PERP of 0
    // even if SLIL is higher
    perpValueForSlil = CmputCostForFunction(SCF_PERP);
    if (peakSpillCost_ < perpValueForSlil)
      peakSpillCost_ = perpValueForSlil;
  }
  else
    newSpillCost = CmputCostForFunction(GetSpillCostFunc());

#ifdef IS_DEBUG_SLIL_CORRECT
  if (OPTSCHED_gPrintSpills) {
    Logger::Info(
        "Printing live range lengths for instruction AFTER calculation.");
    for (int j = 0; j < regTypeCnt_; j++) {
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
#endif
}
/*****************************************************************************/

void BBWithSpill::UpdateSpillInfoForUnSchdul_(SchedInstruction *inst) {
  int16_t regType;
  int i, defCnt, useCnt, regNum, physRegNum;
  RegIndxTuple *defs, *uses;
  Register *def, *use;
  bool isLive;

#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Updating reg pressure after unscheduling Inst %d",
               inst->GetNum());
#endif

  defCnt = inst->GetDefs(defs);
  useCnt = inst->GetUses(uses);

  // (Chris): Update the SLIL for all live regs at this point.
  if (GetSpillCostFunc() == SCF_SLIL) {
    for (int i = 0; i < regTypeCnt_; ++i) {
      for (int j = 0; j < liveRegs_[i].GetSize(); ++j) {
        if (liveRegs_[i].GetBit(j)) {
          // const Register *reg = regFiles_[i].GetReg(j);
          sumOfLiveIntervalLengths_[i]--;
          // if (!reg->IsInInterval(inst) && !reg->IsInPossibleInterval(inst)) {
          //   --dynamicSlilLowerBound_;
          // }
        }
      }
      assert(sumOfLiveIntervalLengths_[i] >= 0 &&
             "UpdateSpillInfoForUnSchdul_: SLIL negative!");
    }
  }

  // Update Live regs
  for (i = 0; i < defCnt; i++) {
    def = dataDepGraph_->getRegByTuple(&defs[i]);
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
    use = dataDepGraph_->getRegByTuple(&uses[i]);
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
        // if (!use->IsInInterval(inst) && !use->IsInPossibleInterval(inst)) {
        //   --dynamicSlilLowerBound_;
        // }
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
  UpdateSpillInfoForSchdul_(inst, trackCnflcts);
}

__device__
void BBWithSpill::Dev_SchdulInst(SchedInstruction *inst, InstCount cycleNum,
                             InstCount slotNum, bool trackCnflcts) {
  dev_crntCycleNum_[GLOBALTID] = cycleNum;
  dev_crntSlotNum_[GLOBALTID] = slotNum;
  if (inst == NULL)
    return;
  assert(inst != NULL);
  UpdateSpillInfoForSchdul_(inst, trackCnflcts);
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
  stats::traceOptimalCost.Record(bestCost_);
  stats::traceOptimalScheduleLength.Record(bestSchedLngth_);
#endif
}
/*****************************************************************************/

Enumerator *BBWithSpill::AllocEnumrtr_(Milliseconds timeout) {
  bool enblStallEnum = enblStallEnum_;
  /*  if (!dataDepGraph_->IncludesUnpipelined()) {
      enblStallEnum = false;
    }*/

  enumrtr_ = new LengthCostEnumerator(
      dataDepGraph_, machMdl_, schedUprBound_, GetSigHashSize(),
      GetEnumPriorities(), GetPruningStrategy(), SchedForRPOnly_, enblStallEnum,
      timeout, GetSpillCostFunc(), 0, NULL);

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
        (lngthDeadline == rgnDeadline && rslt == RES_TIMEOUT) ||
        (rslt == RES_SUCCESS && IsSecondPass())) {

      // If doing two pass optsched and on the second pass then terminate if a
      // schedule is found with the same min-RP found in first pass.
      if (rslt == RES_SUCCESS && IsSecondPass()) {
        Logger::Info("Schedule found in second pass, terminating BB loop.");

        if (trgtLngth < schedUprBound_)
          Logger::Info("Schedule found with length %d is shorter than current "
                       "schedule with length %d.",
                       trgtLngth, schedUprBound_);
      }

      break;
    }

    enumrtr_->Reset();
    enumCrntSched_->Reset();

    if (!IsSecondPass())
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

InstCount BBWithSpill::CmputCostForFunction(SPILL_COST_FUNCTION SpillCF) {
  // return the requested cost
  switch (SpillCF) {
  case SCF_TARGET: {
    return OST->getCost(regPressures_);
  }
  case SCF_SLIL: {
    InstCount SLILCost = 0;
    for (int i = 0; i < regTypeCnt_; i ++)
      SLILCost += sumOfLiveIntervalLengths_[i];
    return SLILCost;
  }
  case SCF_PRP: {
    InstCount PRPCost = 0;
    for (int i = 0; i < regTypeCnt_; i ++)
      PRPCost += regPressures_[i];
    return PRPCost;
  }
  case SCF_PEAK_PER_TYPE: {
    InstCount SC = 0;
    InstCount inc;
    for (int i = 0; i < regTypeCnt_; i++) {
      inc = peakRegPressures_[i] - machMdl_->GetPhysRegCnt(i);
      if (inc > 0)
        SC += inc;
    }
    return SC;
  }
  default: {
    // Default is PERP (Some SCF like SUM rely on PERP being the default here)
    InstCount inc;
    InstCount SC = 0;
    for (int i = 0; i < regTypeCnt_; i ++) {
      inc = regPressures_[i] - machMdl_->GetPhysRegCnt(i);
      if (inc > 0)
        SC += inc;
    }
    return SC;
  }
  }
}

__device__
static unsigned getOccupancyWithNumVGPRs(unsigned VGPRs) {
  // approximation from llvm/lib/Target/AMDGPUSubtarget.cpp
  // from this llvm commit fd08dcb9db0df6dc1aaf329f790cc4a7af9e0a91
  if (VGPRs <= 24)
    return 10;
  if (VGPRs <= 28)
    return 9;
  if (VGPRs <= 32)
    return 8;
  if (VGPRs <= 36)
    return 7;
  if (VGPRs <= 40)
    return 6;
  if (VGPRs <= 48)
    return 5;
  if (VGPRs <= 64)
    return 4;
  if (VGPRs <= 84)
    return 3;
  if (VGPRs <= 128)
    return 2;
  return 1;
}

__device__
static unsigned getOccupancyWithNumSGPRs(unsigned SGPRs) {
  // copied from llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
  if (SGPRs <= 80)
    return 10;
  if (SGPRs <= 88)
    return 9;
  if (SGPRs <= 100)
    return 8;
  return 7;
}

__device__
static unsigned getAdjustedOccupancy(unsigned VGPRCount, unsigned SGPRCount,
                                     unsigned MaxOccLDS) {
  unsigned MaxOccVGPR = getOccupancyWithNumVGPRs(VGPRCount);
  unsigned MaxOccSGPR = getOccupancyWithNumSGPRs(SGPRCount);

  #ifdef DEBUG_CLOSE_TO_OCCUPANCY
  #ifdef __HIP_DEVICE_COMPILE__
    if (GLOBALTID==0) {
      printf("Actual MaxOccVGPR Occ: %u, MaxOccSGPR Occ: %u, MaxOccLDS: %u\n", MaxOccVGPR, MaxOccSGPR, MaxOccLDS);
    }
  #endif
  #endif

  if (MaxOccLDS <= MaxOccVGPR && MaxOccLDS <= MaxOccSGPR)
    return MaxOccLDS;
  else if (MaxOccVGPR <= MaxOccSGPR)
    return MaxOccVGPR;
  else
    return MaxOccSGPR;
}

__device__
InstCount BBWithSpill::getAMDGPUCost(unsigned * PRP, unsigned TargetOccupancy,
                               unsigned MaxOccLDS, int16_t regTypeCnt) {
  auto Occ =
      getAdjustedOccupancy(PRP[OptSchedDDGWrapperGCN::VGPR32*numThreads_+GLOBALTID],
                           PRP[OptSchedDDGWrapperGCN::SGPR32*numThreads_+GLOBALTID], MaxOccLDS);
  // RP cost is the difference between the minimum allowed occupancy for the
  // function, and the current occupancy.
  return Occ >= TargetOccupancy ? 0 : TargetOccupancy - Occ;
}

__host__ __device__
unsigned BBWithSpill::closeOccupancyWithNumVGPRs(unsigned VGPRs) {
  // approximation from llvm/lib/Target/AMDGPUSubtarget.cpp
  // from this llvm commit fd08dcb9db0df6dc1aaf329f790cc4a7af9e0a91
  if (VGPRs <= 22)
    return 11;
  if (VGPRs <= 25)
    return 10;
  if (VGPRs <= 29)
    return 9;
  if (VGPRs <= 33)
    return 8;
  if (VGPRs <= 37)
    return 7;
  if (VGPRs <= 45)
    return 6;
  if (VGPRs <= 59)
    return 5;
  if (VGPRs <= 78)
    return 4;
  if (VGPRs <= 118)
    return 3;
  if (VGPRs <= 128)
    return 2;
  return 1;
}

__host__ __device__
unsigned BBWithSpill::closeOccupancyWithNumSGPRs(unsigned SGPRs) {
  // copied from llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
  if (SGPRs <= 76)
    return 11;
  if (SGPRs <= 84)
    return 10;
  if (SGPRs <= 94)
    return 9;
  if (SGPRs <= 100)
    return 8;
  return 7;
}

__host__ __device__
unsigned BBWithSpill::getCloseToOccupancy(unsigned VGPRCount, unsigned SGPRCount,
                                     unsigned MaxOccLDS) {
  unsigned MaxOccVGPR = closeOccupancyWithNumVGPRs(VGPRCount);
  unsigned MaxOccSGPR = closeOccupancyWithNumSGPRs(SGPRCount);

  #ifdef DEBUG_CLOSE_TO_OCCUPANCY
  #ifdef __HIP_DEVICE_COMPILE__
    if (GLOBALTID==0) {
      printf("Close check: MaxOccVGPR Occ: %u, MaxOccSGPR Occ: %u, MaxOccLDS: %u\n", MaxOccVGPR, MaxOccSGPR, MaxOccLDS + 1);
    }
  #endif
  #endif

  // + 1 to account for being close
  if (MaxOccLDS + 1 <= MaxOccVGPR && MaxOccLDS + 1 <= MaxOccSGPR)
    return MaxOccLDS + 1;
  else if (MaxOccVGPR <= MaxOccSGPR)
    return MaxOccVGPR;
  else
    return MaxOccSGPR;
}

__host__ __device__
bool BBWithSpill::closeToRPConstraint() {
  #ifdef __HIP_DEVICE_COMPILE__
  auto Occ =
      getCloseToOccupancy(dev_regPressures_[OptSchedDDGWrapperGCN::VGPR32*numThreads_+GLOBALTID],
                           dev_regPressures_[OptSchedDDGWrapperGCN::SGPR32*numThreads_+GLOBALTID], MaxOccLDS_);
  #else
  auto Occ =
      getCloseToOccupancy(regPressures_[OptSchedDDGWrapperGCN::VGPR32],
                           regPressures_[OptSchedDDGWrapperGCN::SGPR32], MaxOccLDS_);
  #endif
  return Occ <= TargetOccupancy_;
}

__device__
InstCount BBWithSpill::Dev_CmputCostForFunction(SPILL_COST_FUNCTION SpillCF) {
  // return the requested cost
  switch (SpillCF) {
  case SCF_TARGET: {
    return getAMDGPUCost(dev_regPressures_, TargetOccupancy_, MaxOccLDS_, regTypeCnt_);
  }
  case SCF_SLIL: {
    InstCount SLILCost = 0; 
    for (int i = 0; i < regTypeCnt_; i++)
      SLILCost += dev_sumOfLiveIntervalLengths_[i*numThreads_+GLOBALTID];
    return SLILCost;
  }
  case SCF_PRP: {
    InstCount PRPCost = 0; 
    for (int i = 0; i < regTypeCnt_; i ++)
      PRPCost += dev_regPressures_[i*numThreads_+GLOBALTID];
    return PRPCost;
  }
  case SCF_PEAK_PER_TYPE: {
    InstCount SC = 0; 
    InstCount inc;
    for (int i = 0; i < regTypeCnt_; i++) {
      inc = dev_peakRegPressures_[i*numThreads_+GLOBALTID] - machMdl_->GetPhysRegCnt(i);
      if (inc > 0) 
        SC += inc; 
    }    
    return SC;
  }
  default: {
    // Default is PERP (Some SCF like SUM rely on PERP being the default here)
    InstCount inc;
    InstCount SC = 0;
    for (int i = 0; i < regTypeCnt_; i ++) {
      inc = dev_regPressures_[i*numThreads_+GLOBALTID] - machMdl_->GetPhysRegCnt(i);
      if (inc > 0) 
        SC += inc;
    }
    return SC;
  }
  }
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
    optmlSpillCost_ = crntSpillCost_;
    SetBestSchedLength(crntSched->GetCrntLngth());
    enumBestSched_->Copy(crntSched);
    bestSched_ = enumBestSched_;
  }

  return GetBestCost();
}
/*****************************************************************************/

__host__ __device__
bool BBWithSpill::needsSLIL() { return NeedsComputeSLIL; }

__host__ __device__
bool BBWithSpill::needsTarget() { return needsComputeTarget; }

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
  if (GetSpillCostFunc() == SCF_SLIL) {
    crntCost = dynamicSlilLowerBound_ * SCW_ + trgtLngth * schedCostFactor_;
  } else {
    crntCost = crntSpillCost_ * SCW_ + trgtLngth * schedCostFactor_;
  }
  crntCost -= GetCostLwrBound();
  dynmcCostLwrBound = crntCost;

  // assert(cost >= 0);
  assert(dynmcCostLwrBound >= 0);

  fsbl = dynmcCostLwrBound < GetBestCost();

  // FIXME: RP tracking should be limited to the current SCF. We need RP
  // tracking interface.
  if (fsbl) {
    node->SetCost(crntCost);
    node->SetCostLwrBound(dynmcCostLwrBound);
    node->SetPeakSpillCost(peakSpillCost_);
    node->SetSpillCostSum(totSpillCost_);
  }
  return fsbl;
}
/*****************************************************************************/

void BBWithSpill::SetSttcLwrBounds(EnumTreeNode *) {
  // Nothing.
}

/*****************************************************************************/

__host__ __device__
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

__host__ __device__
InstCount BBWithSpill::GetCrntSpillCost() {
#ifdef __HIP_DEVICE_COMPILE__ // Device version of function
  return dev_crntSpillCost_[GLOBALTID];
#else
  return crntSpillCost_;
#endif
}

__host__ __device__
InstCount BBWithSpill::ReturnPeakSpillCost() {
#ifdef __CUDA_ARCH__ // Device version of function
  return dev_peakSpillCost_[GLOBALTID];
#else
  return peakSpillCost_;
#endif
}

void BBWithSpill::AllocDevArraysForParallelACO(int numThreads) {
  // Temporarily holds large hipMalloc arrays as they are divided
  InstCount *temp;
  unsigned *u_temp;
  size_t memSize = sizeof(InstCount) * numThreads;
  hipMalloc(&dev_crntCycleNum_, memSize);
  hipMalloc(&dev_crntSlotNum_, memSize);
  hipMalloc(&dev_crntSpillCost_, memSize);
  hipMalloc(&dev_crntStepNum_, memSize);
  hipMalloc(&dev_peakSpillCost_, memSize);
  hipMalloc(&dev_totSpillCost_, memSize);
  if (needsSLIL()) {
    hipMalloc(&dev_slilSpillCost_, memSize);
    hipMalloc(&dev_dynamicSlilLowerBound_, memSize);
  }
  memSize = sizeof(int) * numThreads;
  hipMalloc(&dev_schduldInstCnt_, memSize);
  memSize = sizeof(WeightedBitVector *) * regTypeCnt_;
  hipMallocManaged(&dev_liveRegs_, memSize);
  memSize = sizeof(InstCount) * regTypeCnt_ * numThreads;
  hipMalloc(&dev_peakRegPressures_, memSize);
  memSize = sizeof(unsigned) * regTypeCnt_ * numThreads;
  hipMalloc(&dev_regPressures_, memSize);
  memSize = sizeof(InstCount) * dataDepGraph_->GetInstCnt() * numThreads;
  hipMalloc(&dev_spillCosts_, memSize);
  if (needsSLIL()) {
    memSize = sizeof(int) * regTypeCnt_ * numThreads;
    hipMalloc(&dev_sumOfLiveIntervalLengths_, memSize);
  }
}

void BBWithSpill::CopyPointersToDevice(SchedRegion* dev_rgn, int numThreads) {
  size_t memSize;
  // copy liveRegs to device
  // this will hold the array of all liveRegs for all threads on the device
  // temporarily before the pointers are assigned to dev_liveRegs_
  WeightedBitVector *dev_temp_liveRegs;
  WeightedBitVector *temp_bv;
  unsigned int *dev_vctr = NULL;
  int unitCnt, indx = 0;
  // Find totUnitCnt to determine size of vctr for all liveRegs_
  int totUnitCnt = 0;
  for (int i = 0; i < regTypeCnt_; i++) {
    totUnitCnt += liveRegs_[i].GetUnitCnt();
    // set one bit, so oneCnt_ is nonzero to force reset of vctrs on device
    if (liveRegs_[i].GetUnitCnt() > 0)
      liveRegs_[i].SetBit(0,true,1);
  }
  // Allocate vctr for all dev_liveRegs
  memSize = totUnitCnt * sizeof(unsigned int) * numThreads;
  gpuErrchk(hipMalloc((void**)&dev_vctr, memSize));
  // prepare temp host array to copy all dev_liveRegs in one call
  memSize = regTypeCnt_ * sizeof(WeightedBitVector) * numThreads;
  gpuErrchk(hipMallocManaged((void**)&dev_temp_liveRegs, memSize));
  temp_bv = (WeightedBitVector *)malloc(memSize);
  // temp array laid out in the format temp_bv[liveRegIndx][TID]
  // so that all of the threads have their copy of liveRegs
  // next to each other in memory
  memSize = sizeof(WeightedBitVector);
  for (int i = 0; i < regTypeCnt_; i++)
    for (int j = 0; j < numThreads; j++)
      memcpy(&temp_bv[(i * numThreads) + j], &liveRegs_[i], memSize);
  // copy formatted host array to device
  memSize = regTypeCnt_ * sizeof(WeightedBitVector) * numThreads;
  gpuErrchk(hipMemcpy(dev_temp_liveRegs, temp_bv, memSize,
                       hipMemcpyHostToDevice));
  // free the temp host array
  free(temp_bv);
  // make sure host also have copy of the device pointers
  gpuErrchk(hipMemPrefetchAsync(dev_temp_liveRegs, memSize, hipCpuDeviceId));
  // assign each dev_temp_liveReg a portion of the dev_vctr allocation
  // and then set the dev_liveRegs_[index] pointers to each group of
  // dev_temp_liveRegs
  indx = 0;
  for (int i = 0; i < regTypeCnt_; i++) {
    unitCnt = liveRegs_[i].GetUnitCnt();
    for (int j = 0; j < numThreads; j++) {
      if (unitCnt > 0) {
        dev_temp_liveRegs[(i * numThreads) + j].vctr_ = &dev_vctr[indx];
        indx += unitCnt;
      }
    }
    //update device pointer
    ((BBWithSpill *)
    dev_rgn)->dev_liveRegs_[i] = &dev_temp_liveRegs[i * numThreads];
  }
  // make sure managed mem is updated on device before kernel start
  memSize = regTypeCnt_ * sizeof(WeightedBitVector) * numThreads;
  gpuErrchk(hipMemPrefetchAsync(dev_temp_liveRegs, memSize, 0));

  // make sure managed mem is updated on device before kernel start
  memSize = sizeof(WeightedBitVector *) * regTypeCnt_;
  gpuErrchk(hipMemPrefetchAsync(dev_liveRegs_, memSize, 0));
}

void BBWithSpill::FreeDevicePointers(int numThreads) {
  hipFree(dev_liveRegs_[0][0].vctr_);
  hipFree(dev_liveRegs_[0]);
  hipFree(dev_liveRegs_);
  hipFree(dev_crntCycleNum_);
  hipFree(dev_crntSlotNum_);
  hipFree(dev_crntSpillCost_);
  hipFree(dev_crntStepNum_);
  hipFree(dev_peakSpillCost_);
  hipFree(dev_totSpillCost_);
  if (needsSLIL()) {
    hipFree(dev_slilSpillCost_);
    hipFree(dev_dynamicSlilLowerBound_);
    hipFree(dev_sumOfLiveIntervalLengths_);
  }
  hipFree(dev_schduldInstCnt_);
  hipFree(dev_peakRegPressures_);
  hipFree(dev_regPressures_);
  hipFree(dev_spillCosts_);
}
