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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>

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
  NeedsComputeSLIL = (spillCostFunc == SCF_SLIL);

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
}
/****************************************************************************/

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
  case SCHED_STALLING_LIST:
    return new StallSchedulingListScheduler(dataDepGraph_, machMdl_,
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
    auto recSuccBV = inst->GetRcrsvNghbrBitVector(DIR_FRWRD);
    for (Register *def : inst->GetDefs()) {
      for (const auto &dependentInst : def->GetUseList()) {
        auto recPredBV = const_cast<SchedInstruction *>(dependentInst)
                             ->GetRcrsvNghbrBitVector(DIR_BKWRD);
        assert(recSuccBV->GetSize() == recPredBV->GetSize() &&
               "Successor list size doesn't match predecessor list size!");
        for (int k = 0; k < recSuccBV->GetSize(); ++k) {
          if (recSuccBV->GetBit(k) & recPredBV->GetBit(k)) {
            if (def->AddToInterval(dataDepGraph_->GetInstByIndx(k))) {
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

    // Get a list of instructions that define the registers, in array form.
    usedInsts.clear();
    llvm::transform(inst->GetUses(), std::back_inserter(usedInsts),
                    [&](Register *reg) {
                      assert(reg->GetDefList().size() == 1 &&
                             "Number of defs for register is not 1!");
                      return std::make_pair(*(reg->GetDefList().begin()), reg);
                    });

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
  // for(InstCount i=0; i< dataDepGraph_->GetInstCnt(); i++) {
  //   inst = dataDepGraph_->GetInstByIndx(i);
  // }

  InstCount staticLowerBound = CmputExecCostLwrBound() + CmputRPCostLwrBound();

#if defined(IS_DEBUG_STATIC_LOWER_BOUND)
  Logger::Event("StaticLowerBoundDebugInfo", "name", dataDepGraph_->GetDagID(),
                "spill_cost_lb", spillCostLwrBound, "sc_factor", SCW_,       //
                "length_lb", schedLwrBound_, "len_factor", schedCostFactor_, //
                "static_lb", staticLowerBound);
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

void BBWithSpill::addRecordedCost(SPILL_COST_FUNCTION Scf) {
  NeedsComputeSLIL |= (Scf == SCF_SLIL);
  if (!llvm::is_contained(recordedCostFunctions, Scf))
    recordedCostFunctions.push_back(Scf);
}
/*****************************************************************************/

void BBWithSpill::storeExtraCost(InstSchedule *sched, SPILL_COST_FUNCTION Scf) {
  sched->SetExtraSpillCost(Scf, CmputCostForFunction(Scf));
}

/*****************************************************************************/

InstCount BBWithSpill::getUnnormalizedIncrementalRPCost() {return crntSpillCost_;}
/*****************************************************************************/

void BBWithSpill::InitForSchdulng() {
  InitForCostCmputtn_();

  schduldEntryInstCnt_ = 0;
  schduldExitInstCnt_ = 0;
  schduldInstCnt_ = 0;
}
/*****************************************************************************/

void BBWithSpill::InitForCostCmputtn_() {
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

  for (SPILL_COST_FUNCTION CostFunction : recordedCostFunctions)
    storeExtraCost(sched, CostFunction);

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

void BBWithSpill::UpdateSpillInfoForSchdul_(SchedInstruction *inst,
                                            bool trackCnflcts) {
  int16_t regType;
  int regNum, physRegNum;
  int liveRegs;
  InstCount newSpillCost;

#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Updating reg pressure after scheduling Inst %d",
               inst->GetNum());
#endif

  // Update Live regs after uses
  for (Register *use : inst->GetUses()) {
    regType = use->GetType();
    regNum = use->GetNum();
    physRegNum = use->GetPhysicalNumber();

    if (use->IsLive() == false)
      llvm::report_fatal_error("Reg " + std::to_string(regNum) + " of type " +
                                   std::to_string(regType) +
                                   " is used without being defined",
                               false);

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
  for (Register *def : inst->GetDefs()) {
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
    if (needsSLIL()) {
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
  }

  if (GetSpillCostFunc() == SCF_SLIL)
    slilSpillCost_ = CmputCostForFunction(GetSpillCostFunc());
  else
    newSpillCost = CmputCostForFunction(GetSpillCostFunc());

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
  int regNum, physRegNum;
  bool isLive;

#ifdef IS_DEBUG_REG_PRESSURE
  Logger::Info("Updating reg pressure after unscheduling Inst %d",
               inst->GetNum());
#endif

  // (Chris): Update the SLIL for all live regs at this point.
  if (needsSLIL()) {
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
  for (Register *def : inst->GetDefs()) {
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

  for (Register *use : inst->GetUses()) {
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
      if (needsSLIL()) {
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
    Logger::Event("Enumerating", "target_length", trgtLngth);

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
  case SCF_TARGET:
    return OST->getCost(regPressures_);

  case SCF_SLIL:
    return std::accumulate(sumOfLiveIntervalLengths_.begin(),
                           sumOfLiveIntervalLengths_.end(), 0);

  case SCF_PRP:
    return std::accumulate(regPressures_.begin(), regPressures_.end(), 0);

  case SCF_PEAK_PER_TYPE: {
    InstCount SC = 0;
    for (int i = 0; i < regTypeCnt_; i++)
      SC += std::max(0, peakRegPressures_[i] - machMdl_->GetPhysRegCnt(i));
    return SC;
  }
  default: {
    // Default is PERP (Some SCF like SUM rely on PERP being the default here)
    int i = 0;
    InstCount SC = 0;
    std::for_each(regPressures_.begin(), regPressures_.end(),
                  [&](InstCount RP) {
                    SC += std::max(0, RP - machMdl_->GetPhysRegCnt(i++));
                  });
    return SC;
  }
  }
}

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

bool BBWithSpill::needsSLIL() { return NeedsComputeSLIL; }

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

  // Update Live regs
  for (Register *def : inst->GetDefs()) {
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
