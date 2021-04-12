/*******************************************************************************
Description:  Defines a scheduling region for basic blocks whose scheduler takes
              into account the cost of spilled registers.
Author:       Ghassan Shobaki
Created:      Unknown
Last Update:  Apr. 2011
*******************************************************************************/

#ifndef OPTSCHED_SPILL_BB_SPILL_H
#define OPTSCHED_SPILL_BB_SPILL_H

#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <set>
#include <vector>

namespace llvm {
namespace opt_sched {

class LengthCostEnumerator;
class EnumTreeNode;
class Register;
class RegisterFile;
class BitVector;

class BBWithSpill : public SchedRegion {
private:
  LengthCostEnumerator *enumrtr_;

  InstCount crntSpillCost_;
  InstCount optmlSpillCost_;

  // The target machine
  const OptSchedTarget *OST;

  bool enblStallEnum_;
  int SCW_;
  int schedCostFactor_;

  bool SchedForRPOnly_;

  int16_t regTypeCnt_;
  RegisterFile *regFiles_;

  // A bit vector indexed by register number indicating whether that
  // register is live
  WeightedBitVector *liveRegs_;

  // A bit vector indexed by physical register number indicating whether
  // that physical register is live
  WeightedBitVector *livePhysRegs_;

  // Sum of lengths of live ranges. This vector is indexed by register type,
  // and each type will have its sum of live interval lengths computed.
  std::vector<int> sumOfLiveIntervalLengths_;

  InstCount staticSlilLowerBound_ = 0;

  // (Chris): The dynamic lower bound for SLIL is calculated differently from
  // the other cost functions. It is first set when the static lower bound is
  // calculated.
  InstCount dynamicSlilLowerBound_ = 0;

  int entryInstCnt_;
  int exitInstCnt_;
  int schduldEntryInstCnt_;
  int schduldExitInstCnt_;
  int schduldInstCnt_;

  InstCount *spillCosts_;
  // Current register pressure for each register type.
  SmallVector<unsigned, 8> regPressures_;
  SmallVector<SPILL_COST_FUNCTION, 8> recordedCostFunctions;
  InstCount *peakRegPressures_;
  InstCount crntStepNum_;
  InstCount peakSpillCost_;
  InstCount totSpillCost_;
  InstCount slilSpillCost_;
  bool trackLiveRangeLngths_;
  bool NeedsComputeSLIL;

  // Virtual Functions:
  // Given a schedule, compute the cost function value
  InstCount CmputNormCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                           InstCount &execCost, bool trackCnflcts);
  InstCount CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                       InstCount &execCost, bool trackCnflcts);
  void CmputSchedUprBound_();
  Enumerator *AllocEnumrtr_(Milliseconds timeout);
  FUNC_RESULT Enumerate_(Milliseconds startTime, Milliseconds rgnDeadline,
                         Milliseconds lngthDeadline);
  void SetupForSchdulng_();
  void FinishHurstc_();
  void FinishOptml_();
  void CmputAbslutUprBound_();
  ConstrainedScheduler *AllocHeuristicScheduler_();
  bool EnableEnum_();

  // BBWithSpill-specific Functions:
  InstCount CmputCostLwrBound_(InstCount schedLngth);
  InstCount CmputCostLwrBound_();
  void InitForCostCmputtn_();
  InstCount CmputDynmcCost_();

  void UpdateSpillInfoForSchdul_(SchedInstruction *inst, bool trackCnflcts);
  void UpdateSpillInfoForUnSchdul_(SchedInstruction *inst);
  void SetupPhysRegs_();
  // can only compute SLIL if SLIL was the spillCostFunc
  // This function must only be called after the regPressures_ is computed
  InstCount CmputCostForFunction(SPILL_COST_FUNCTION SpillCF);
  void CmputCrntSpillCost_();
  bool ChkSchedule_(InstSchedule *bestSched, InstSchedule *lstSched);
  void CmputCnflcts_(InstSchedule *sched);

public:
  BBWithSpill(const OptSchedTarget *OST_, DataDepGraph *dataDepGraph,
              long rgnNum, int16_t sigHashSize, LB_ALG lbAlg,
              SchedPriorities hurstcPrirts, SchedPriorities enumPrirts,
              bool vrfySched, Pruning PruningStrategy, bool SchedForRPOnly,
              bool enblStallEnum, int SCW, SPILL_COST_FUNCTION spillCostFunc,
              SchedulerType HeurSchedType);
  ~BBWithSpill();

  InstCount CmputCostLwrBound();
  InstCount CmputExecCostLwrBound();
  InstCount CmputRPCostLwrBound();

  // calling addRecordedCost will cause this region to record the current spill
  // cost of the schedule using Scf whenever the spill cost updates
  void addRecordedCost(SPILL_COST_FUNCTION Scf);
  void storeExtraCost(InstSchedule *sched, SPILL_COST_FUNCTION Scf);
  InstCount getUnnormalizedIncrementalRPCost();

  InstCount UpdtOptmlSched(InstSchedule *crntSched,
                           LengthCostEnumerator *enumrtr);
  bool ChkCostFsblty(InstCount trgtLngth, EnumTreeNode *treeNode);
  void SchdulInst(SchedInstruction *inst, InstCount cycleNum, InstCount slotNum,
                  bool trackCnflcts);
  void UnschdulInst(SchedInstruction *inst, InstCount cycleNum,
                    InstCount slotNum, EnumTreeNode *trgtNode);
  void SetSttcLwrBounds(EnumTreeNode *node);
  bool ChkInstLglty(SchedInstruction *inst);
  bool needsSLIL();
  void InitForSchdulng();

protected:
  // (Chris)
  inline virtual const std::vector<int> &GetSLIL_() const {
    return sumOfLiveIntervalLengths_;
  }
};

} // namespace opt_sched
} // namespace llvm

#endif
