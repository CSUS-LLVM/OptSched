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
#include <cuda_runtime.h>

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
  // pointer to a device array used to store crntSpillCost_ for
  // each thread by parallel ACO
  InstCount *dev_crntSpillCost_;
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
  // pointer to a device array used to store liveRegs_ for
  // each thread by parallel ACO
  WeightedBitVector **dev_liveRegs_;

  // A bit vector indexed by physical register number indicating whether
  // that physical register is live
  WeightedBitVector *livePhysRegs_;
  // pointer to a device array used to store livePhysRegs_ for
  // each thread by parallel ACO
  WeightedBitVector **dev_livePhysRegs_;

  // Sum of lengths of live ranges. This array is indexed by register type,
  // and each type will have its sum of live interval lengths computed.
  int *sumOfLiveIntervalLengths_;
  // pointer to a device array used to store sumOfLiveIntervalLengths_ for
  // each thread by parallel ACO
  int **dev_sumOfLiveIntervalLengths_;

  InstCount staticSlilLowerBound_ = 0;

  // (Chris): The dynamic lower bound for SLIL is calculated differently from
  // the other cost functions. It is first set when the static lower bound is
  // calculated.
  InstCount dynamicSlilLowerBound_ = 0;
  // pointer to a device array used to store dynamicSlilLowerBound_ for
  // each thread by parallel ACO
  InstCount *dev_dynamicSlilLowerBound_;

  int entryInstCnt_;
  int exitInstCnt_;
  int schduldEntryInstCnt_;
  // pointer to a device array used to store schduldEntryInstCnt_ for
  // each thread by parallel ACO
  int *dev_schduldEntryInstCnt_;
  int schduldExitInstCnt_;
  // pointer to a device array used to store schduldExitInstCnt_ for
  // each thread by parallel ACO
  int *dev_schduldExitInstCnt_;
  int schduldInstCnt_;
  // pointer to a device array used to store schduldInstCnt_ for
  // each thread by parallel ACO
  int *dev_schduldInstCnt_;

  InstCount *spillCosts_;
  // pointer to a device array used to store spillCosts_ for
  // each thread by parallel ACO
  InstCount **dev_spillCosts_;
  // Current register pressure for each register type.
  unsigned *regPressures_;
  // pointer to a device array used to store regPressures_ for
  // each thread by parallel ACO
  unsigned **dev_regPressures_;
  InstCount *peakRegPressures_;
  // pointer to a device array used to store peakRegPressures_ for
  // each thread by parallel ACO
  InstCount **dev_peakRegPressures_;

  InstCount crntStepNum_;
  // pointer to a device array used to store crntStepNum_ for
  // each thread by parallel ACO
  InstCount *dev_crntStepNum_;
  InstCount peakSpillCost_;
  // pointer to a device array used to store peakSpillCost_ for
  // each thread by parallel ACO
  InstCount *dev_peakSpillCost_;
  InstCount totSpillCost_;
  // pointer to a device array used to store totSpillCost_ for
  // each thread by parallel ACO
  InstCount *dev_totSpillCost_;
  InstCount slilSpillCost_;
  // pointer to a device array used to store slilSpillCost_ for
  // each thread by parallel ACO
  InstCount *dev_slilSpillCost_;
  bool trackLiveRangeLngths_;
  bool NeedsComputeSLIL;

  // Virtual Functions:
  // Given a schedule, compute the cost function value
  InstCount CmputNormCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                           InstCount &execCost, bool trackCnflcts);
  InstCount CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                       InstCount &execCost, bool trackCnflcts);
  //non virtual versions of function to be invoked on device
  __device__
  InstCount Dev_CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
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
  __host__ __device__
  void InitForCostCmputtn_();
  InstCount CmputDynmcCost_();

  __host__ __device__
  void UpdateSpillInfoForSchdul_(SchedInstruction *inst, bool trackCnflcts);
  void UpdateSpillInfoForUnSchdul_(SchedInstruction *inst);
  void SetupPhysRegs_();
  // can only compute SLIL if SLIL was the spillCostFunc
  // This function must only be called after the regPressures_ is computed
  InstCount CmputCostForFunction(SPILL_COST_FUNCTION SpillCF);
  // Device version of above function
  __device__
  InstCount Dev_CmputCostForFunction(SPILL_COST_FUNCTION SpillCF);
  __host__ __device__
  void CmputCrntSpillCost_();
  bool ChkSchedule_(InstSchedule *bestSched, InstSchedule *lstSched);
  void CmputCnflcts_(InstSchedule *sched);

public:
  BBWithSpill(const OptSchedTarget *OST_, DataDepGraph *dataDepGraph,
              long rgnNum, int16_t sigHashSize, LB_ALG lbAlg,
              SchedPriorities hurstcPrirts, SchedPriorities enumPrirts,
              bool vrfySched, Pruning PruningStrategy, bool SchedForRPOnly,
              bool enblStallEnum, int SCW, SPILL_COST_FUNCTION spillCostFunc,
              SchedulerType HeurSchedType,
	      MachineModel *dev_machMdl);
  ~BBWithSpill();

  InstCount CmputCostLwrBound();
  InstCount CmputExecCostLwrBound();
  InstCount CmputRPCostLwrBound();

  InstCount UpdtOptmlSched(InstSchedule *crntSched,
                           LengthCostEnumerator *enumrtr);
  bool ChkCostFsblty(InstCount trgtLngth, EnumTreeNode *treeNode);
  void SchdulInst(SchedInstruction *inst, InstCount cycleNum, InstCount slotNum,
                  bool trackCnflcts);
  //SchdulInst cannot be called directly on device due to overriding a 
  //pure virtual function, so a copied BBWithSpill cannot invoke it
  __device__
  void Dev_SchdulInst(SchedInstruction *inst, InstCount cycleNum, 
		      InstCount slotNum, bool trackCnflcts);
  void UnschdulInst(SchedInstruction *inst, InstCount cycleNum,
                    InstCount slotNum, EnumTreeNode *trgtNode);
  void SetSttcLwrBounds(EnumTreeNode *node);
  __host__ __device__
  bool ChkInstLglty(SchedInstruction *inst);
  __host__ __device__
  void InitForSchdulng();
  __device__
  void Dev_InitForSchdulng();
  __device__
  void SetRegFiles(RegisterFile *regFiles) {regFiles_ = regFiles; }
  void AllocDevArraysForParallelACO(int numThreads);
  void CopyPointersToDevice(SchedRegion *dev_rgn, int numThreads);
  void FreeDevicePointers(int numThreads);
  //non virtual versions of function to be invoked on device
  __device__
  InstCount Dev_CmputNormCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                           InstCount &execCost, bool trackCnflcts);
  __host__ __device__
  bool needsSLIL();
  __host__ __device__
  InstCount GetCrntSpillCost();
  __host__ __device__
  bool IsRPHigh(int regType) const {
    #ifdef __CUDA_ARCH__
    return dev_regPressures_[regType][GLOBALTID] > (unsigned int) machMdl_->GetPhysRegCnt(regType);
    #else
      return regPressures_[regType] > (unsigned int) machMdl_->GetPhysRegCnt(regType);
    #endif
  }
protected:
  // (Chris)
  inline virtual const int *GetSLIL_() const {
    return sumOfLiveIntervalLengths_;
  }

  inline virtual const int GetSLIL_size_() const {
    return regTypeCnt_;
  }
};

} // namespace opt_sched
} // namespace llvm

#endif
