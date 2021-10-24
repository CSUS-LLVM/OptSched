/*******************************************************************************
Description:  Implements an abstract base class for representing scheduling
              regions.
Author:       Ghassan Shobaki
Created:      Apr. 2005
Updated By:   Ciprian Elies and Vang Thao
Last Update:  Jan. 2020
*******************************************************************************/

#ifndef OPTSCHED_SCHED_REGION_SCHED_REGION_H
#define OPTSCHED_SCHED_REGION_SCHED_REGION_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
// For DataDepGraph, LB_ALG.
#include "opt-sched/Scheduler/data_dep.h"
// For Enumerator, LengthCostEnumerator, EnumTreeNode and Pruning.
#include "opt-sched/Scheduler/enumerator.h"

namespace llvm {
namespace opt_sched {

// How to compare cost.
// TODO(max): Elaborate.
enum COST_COMP_MODE {
  // Dynamically.
  CCM_DYNMC,
  // Statically.
  CCM_STTC
};

// (Chris)
enum class BLOCKS_TO_KEEP {
  ZERO_COST,
  IMPROVED,
  OPTIMAL,
  IMPROVED_OR_OPTIMAL,
  ALL
};

// Where to perform graph transformations; flag enum
enum class GT_POSITION : uint32_t {
  NONE = 0x0,
  // Run on all blocks before the heuristic
  BEFORE_HEURISTIC = 0x1,
  // Run only if the heuristic scheduler doesn't prove the schedule optimal
  AFTER_HEURISTIC = 0x2,
};

inline GT_POSITION operator|(GT_POSITION lhs, GT_POSITION rhs) {
  return (GT_POSITION)((uint32_t)lhs | (uint32_t)rhs);
}

inline GT_POSITION operator&(GT_POSITION lhs, GT_POSITION rhs) {
  return (GT_POSITION)((uint32_t)lhs & (uint32_t)rhs);
}

inline GT_POSITION &operator|=(GT_POSITION &lhs, GT_POSITION rhs) {
  return lhs = lhs | rhs;
}

inline GT_POSITION &operator&=(GT_POSITION &lhs, GT_POSITION rhs) {
  return lhs = lhs & rhs;
}

class ListScheduler;

class SchedRegion {
public:
  // TODO(max): Document.
  SchedRegion(MachineModel *machMdl, DataDepGraph *dataDepGraph, long rgnNum,
              int16_t sigHashSize, LB_ALG lbAlg, SchedPriorities hurstcPrirts,
              SchedPriorities enumPrirts, bool vrfySched,
              Pruning PruningStrategy, SchedulerType HeurSchedType,
              SPILL_COST_FUNCTION spillCostFunc,
              GT_POSITION GraphTransPosition);
  // Destroys the region. Must be overriden by child classes.
  virtual ~SchedRegion() {}

  // Returns the dependence graph of this region.
  inline DataDepGraph *GetDepGraph() { return dataDepGraph_; }
  // Returns the lower bound on the cost of this region.
  inline int GetCostLwrBound() { return costLwrBound_; }
  // Returns the static lower bound on RP
  inline int getSpillCostLwrBound() { return SpillCostLwrBound_; }
  inline InstCount GetExecCostLwrBound() { return ExecCostLwrBound_; }
  inline InstCount GetRPCostLwrBound() { return RpCostLwrBound_; }
  // Returns the best cost found so far for this region.
  inline InstCount GetBestCost() { return bestCost_; }
  // Returns the heuristic cost for this region.
  inline InstCount GetHeuristicCost() { return hurstcCost_; }
  // Return the spill cost for first pass of this region
  inline InstCount getSpillCostConstraint() const { return BestSpillCost_; }
  // Returns the best spill cost found so far for this region
  inline InstCount getBestSpillCost() { return BestSpillCost_; }
  // Returns a pointer to the list scheduler heurisitcs.
  inline SchedPriorities GetHeuristicPriorities() { return hurstcPrirts_; }
  // Get the number of simulated spills code added for this block.
  inline int GetSimSpills() { return totalSimSpills_; }

  // Gets the un-normalized incremental RP cost for the region(used by ACO)
  virtual InstCount getUnnormalizedIncrementalRPCost() const = 0;
  // Get schedLength for best-so-far sched
  inline InstCount getBestSchedLength() { return bestSchedLngth_; }

  // TODO(max): Document.
  virtual FUNC_RESULT
  FindOptimalSchedule(Milliseconds rgnTimeout, Milliseconds lngthTimeout,
                      bool &isHurstcOptml, InstCount &bestCost,
                      InstCount &bestSchedLngth, InstCount &hurstcCost,
                      InstCount &hurstcSchedLngth, InstSchedule *&bestSched,
                      bool filterByPerp, const BLOCKS_TO_KEEP blocksToKeep);

  // External abstract functions.

  // TODO(max): Document.
  virtual InstCount CmputExecCostLwrBound() = 0;
  virtual InstCount CmputRPCostLwrBound() = 0;
  virtual void CmputAndSetCostLwrBound() = 0;

  virtual int cmputSpillCostLwrBound() = 0;

  // TODO(max): Document.
  virtual void UpdtOptmlSched(InstSchedule *crntSched) = 0;

  virtual void UpdtOptmlSchedFrstPss(InstSchedule *crntSched,
                                     InstCount crntCost) = 0;

  virtual void UpdtOptmlSchedScndPss(InstSchedule *crntSched,
                                     InstCount crntCost) = 0;

  virtual void UpdtOptmlSchedWghtd(InstSchedule *crntSched,
                                   InstCount crntCost) = 0;

  // TODO(max): Document.
  virtual bool ChkCostFsblty(InstCount trgtLngth, EnumTreeNode *treeNode,
                             InstCount &RPCost) = 0;
  // TODO(max): Document.
  virtual void SchdulInst(SchedInstruction *inst, InstCount cycleNum,
                          InstCount slotNum, bool trackCnflcts) = 0;
  // TODO(max): Document.
  virtual void UnschdulInst(SchedInstruction *inst, InstCount cycleNum,
                            InstCount slotNum, EnumTreeNode *trgtNode) = 0;
  // TODO(max): Document.
  virtual void SetSttcLwrBounds(EnumTreeNode *node) = 0;

  // Do region-specific checking for the legality of scheduling the
  // given instruction in the current issue slot
  virtual bool ChkInstLglty(SchedInstruction *inst) = 0;

  virtual void InitForSchdulng() = 0;

  virtual bool ChkSchedule_(InstSchedule *bestSched,
                            InstSchedule *lstSched) = 0;

  // TODO(max): Document.
  InstSchedule *AllocNewSched_();

  void UpdateScheduleCost(InstSchedule *sched);
  SPILL_COST_FUNCTION GetSpillCostFunc();

  // Initialize variables for the second pass of the two-pass-optsched
  void InitSecondPass(bool EnableMutations);

  // Initialize variables to reflect that we are using two-pass version of
  // algorithm
  void initTwoPassAlg();

  bool isTwoPassEnabled() const { return TwoPassEnabled_; }

  bool IsSecondPass() const { return isSecondPass_; }

  bool enumFoundSchedule() { return EnumFoundSchedule; }
  void setEnumFoundSchedule() { EnumFoundSchedule = true; }

private:
  // The algorithm to use for calculated lower bounds.
  LB_ALG lbAlg_;
  // The number of this region.
  long rgnNum_;
  // Whether to verify the schedule after calculating it.
  bool vrfySched_;

  // Whether to dump the DDGs for the blocks we schedule
  bool DumpDDGs_;
  // Where to dump the DDGs
  std::string DDGDumpPath_;

  // The normal heuristic scheduling results.
  InstCount hurstcCost_;

  // Spill cost for heuristic schedule
  InstCount HurstcSpillCost_;

  // total simulated spills.
  int totalSimSpills_;

  // What list scheduler should be used to find an initial feasible schedule.
  SchedulerType HeurSchedType_;

  // Used for two-pass-optsched to enable second pass functionalies.
  bool isSecondPass_;

  /// If mutations are enabled then the sequential list scheduler must ignore
  /// artificial edges when scheduling then add them back in after scheduling.
  bool EnableMutations;

  /// Indicate whether the B&B enumerator found any schedule.
  bool EnumFoundSchedule;

  // The absolute cost lower bound to be used as a ref for normalized costs.
  InstCount costLwrBound_ = 0;

  // The static lower bound for RP - used as reference for normalized RP
  InstCount SpillCostLwrBound_ = 0;

  // The best results found so far.
  InstCount bestCost_;
  InstCount bestSchedLngth_;
  InstCount BestSpillCost_;

  // (Chris): The cost function. Defaults to PERP.
  SPILL_COST_FUNCTION spillCostFunc_ = SCF_PERP;

  // list scheduling heuristics
  SchedPriorities hurstcPrirts_;
  // Scheduling heuristics to use when enumerating
  SchedPriorities enumPrirts_;

  // TODO(max): Document.
  int16_t sigHashSize_;

  // Where to apply graph transformations
  GT_POSITION GraphTransPosition_;

  // The pruning technique to use for this region.
  Pruning prune_;

  // Whether or not we are using two-pass version of algorithm
  bool TwoPassEnabled_;

protected:
  // The dependence graph of this region.
  DataDepGraph *dataDepGraph_;
  // The machine model used by this region.
  MachineModel *machMdl_;

  // The schedule currently used by the enumerator
  InstSchedule *enumCrntSched_;
  // The best schedule found by the enumerator so far
  InstSchedule *enumBestSched_;
  // The best schedule found so far (may be heuristic or enumerator generated)
  InstSchedule *bestSched_;

  void CalculateUpperBounds(bool BbSchedulerEnabled);
  void CalculateLowerBounds(bool BbSchedulerEnabled);

  bool IsLowerBoundSet_ = false;
  bool IsUpperBoundSet_ = false;
  // TODO(max): Document.
  InstCount schedLwrBound_;
  // TODO(max): Document.
  InstCount schedUprBound_;
  // TODO(max): Document.
  InstCount abslutSchedUprBound_;

  // The absolute lower bound on the ILP/execution cost for normalized costs
  InstCount ExecCostLwrBound_ = 0;

  // The STATIC absolute lower bound for register pressure/spill cost
  // THIS LOWER BOUND IS NOT DYNAMIC.  It is used to get the static normalized
  // cost of a schedule
  InstCount RpCostLwrBound_ = 0;

  // TODO(max): Document.
  InstCount crntCycleNum_;
  // TODO(max): Document.
  InstCount crntSlotNum_;

  bool needsTransitiveClosure(Milliseconds rgnTimeout) const;

  // protected accessors:
  SchedulerType GetHeuristicSchedulerType() const { return HeurSchedType_; }

  void SetBestCost(InstCount bestCost) { bestCost_ = bestCost; }

  void SetBestSchedLength(InstCount bestSchedLngth) {
    bestSchedLngth_ = bestSchedLngth;
  }

  void setBestSpillCost(InstCount BestSpillCost) {
    BestSpillCost_ = BestSpillCost;
  }

  void setCostLwrBound(InstCount CostLwrBound) { costLwrBound_ = CostLwrBound; }

  void setSpillCostLwrBound(InstCount SpillCostLwrBound) {
    SpillCostLwrBound_ = SpillCostLwrBound;
    BestSpillCost_ = SpillCostLwrBound;
  }

  const SchedPriorities &GetEnumPriorities() const { return enumPrirts_; }

  int16_t GetSigHashSize() const { return sigHashSize_; }

  Pruning GetPruningStrategy() const { return prune_; }

  // TODO(max): Document.
  void UseFileBounds_();

  // Top-level function for enumerative scheduling
  FUNC_RESULT Optimize_(Milliseconds startTime, Milliseconds rgnTimeout,
                        Milliseconds lngthTimeout);
  // TODO(max): Document.
  void CmputLwrBounds_(bool useFileBounds);
  // TODO(max): Document.
  bool CmputUprBounds_(InstSchedule *schedule, bool useFileBounds);
  // Handle the enumerator's result
  void HandlEnumrtrRslt_(FUNC_RESULT rslt, InstCount trgtLngth);

  // Simulate local register allocation.
  void RegAlloc_(InstSchedule *&bestSched, InstSchedule *&lstSched);

  // TODO(max): Document.
  virtual void CmputAbslutUprBound_();

  // Internal abstract functions.

  // Compute the normalized cost.
  virtual InstCount CmputNormCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                                   InstCount &execCost, bool trackCnflcts) = 0;
  // TODO(max): Document.
  virtual InstCount CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                               InstCount &execCost, bool trackCnflcts) = 0;
  // TODO(max): Document.
  virtual void CmputSchedUprBound_() = 0;
  // TODO(max): Document.
  virtual Enumerator *AllocEnumrtr_(Milliseconds timeout) = 0;
  // Wrapper for the enumerator
  virtual FUNC_RESULT Enumerate_(Milliseconds startTime,
                                 Milliseconds rgnTimeout,
                                 Milliseconds lngthTimeout) = 0;
  // TODO(max): Document.
  virtual void FinishHurstc_() = 0;
  // TODO(max): Document.
  virtual void FinishOptml_() = 0;
  // TODO(max): Document.
  virtual ConstrainedScheduler *AllocHeuristicScheduler_() = 0;

  virtual bool EnableEnum_() = 0;

  virtual bool needsSLIL() const = 0;

  // Prepares the region for being scheduled.
  virtual void SetupForSchdulng_() = 0;

  // (Chris) Get the SLIL for each set
  virtual const std::vector<int> &GetSLIL_() const = 0;

  FUNC_RESULT runACO(InstSchedule *ReturnSched, InstSchedule *InitSched,
                     bool IsPostBB);

  FUNC_RESULT applyGraphTransformations(bool BbScheduleEnabled,
                                        InstSchedule *heuristicSched,
                                        bool &isLstOptml,
                                        InstSchedule *&bestSched);
  FUNC_RESULT applyGraphTransformation(GraphTrans *GT);
  void updateBoundsAfterGraphTransformations(bool BbSchedulerEnabled);
};

} // namespace opt_sched
} // namespace llvm

#endif
