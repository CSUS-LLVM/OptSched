/*******************************************************************************
Description:  Defines a relaxed scheduler.
Author:       Ghassan Shobaki
Created:      Unknown
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_RELAXED_RELAXED_SCHED_H
#define OPTSCHED_RELAXED_RELAXED_SCHED_H

#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/gen_sched.h"
#include "opt-sched/Scheduler/hash_table.h"

namespace llvm {
namespace opt_sched {

enum RLXD_SCHED_TYPE { RST_STTC, RST_DYNMC, RST_SUBDYNMC };

// A relaxed scheduler class, which relaxes the NP-hard scheduling problem by
// removing the latency constraints and replacing them with release times and
// deadlines. This yields a polynomially solvable problem (relaxed problem)
// whose optimal solution gives a lower bound on the optimal solution to the
// original NP-hard problem.
class RelaxedScheduler : public InstScheduler {
protected:
  DataDepStruct *dataDepGraph_; // the data dependence graph or subgraph

  // A list of instructions sorted by scheduling order
  PriorityList<SchedInstruction> *instLst_;

  // An two-dimensional array holding the next available cycle number for each
  // issue type for each release time
  InstCount *nxtAvlblCycles_[MAX_ISSUTYPE_CNT];

  // A two-dimensional array indexed by issue type and cycle number, where
  // each entry indicates the number of slots of that type that are available
  // in that cycle
  int16_t *avlblSlots_[MAX_ISSUTYPE_CNT];

  int16_t *prevAvlblSlots_[MAX_ISSUTYPE_CNT];

  bool *isFxd_;
  bool useFxng_;

  InstCount fxdInstCnt_;

  // the main direction
  DIRECTION mainDir_;

  // the scheduling direction
  DIRECTION schedDir_;

  bool writeBack_;

  int16_t instCntBits_;

  RLXD_SCHED_TYPE schedType_;

  InstCount *frwrdLwrBounds_;
  InstCount *bkwrdLwrBounds_;

  InstCount maxInstCnt_;

#ifdef IS_DEBUG
  bool *wasLwrBoundCmputd_;
#endif

  InstCount SchdulInst_(SchedInstruction *inst, InstCount minCycle,
                        InstCount maxCycle);
  inline InstCount FindNxtAvlblCycle_(IssueType issuType, InstCount strtCycle);

  inline InstCount CmputDelay_(InstCount schedCycle, InstCount lastCycle,
                               InstCount distFrmLeaf);

  inline unsigned long CmputTplgclPrirty_(SchedInstruction *inst,
                                          DIRECTION dir);

  virtual InstCount CmputReleaseTime_(SchedInstruction *inst);

  virtual InstCount GetCrntLwrBound_(SchedInstruction *inst, DIRECTION dir);

  void SetCrntLwrBound_(SchedInstruction *inst, DIRECTION dir,
                        InstCount newBound);

  InstCount PropagateLwrBound_(SchedInstruction *inst, DIRECTION dir);

  void WriteBoundsBack_();

  inline void SetFix_(SchedInstruction *inst, bool val);
  inline void SetFix_(InstCount indx, bool val);
  inline bool GetFix_(SchedInstruction *inst);

  void Reset_(InstCount startIndx);

public:
  RelaxedScheduler(DataDepStruct *dataDepGraph, MachineModel *machMdl,
                   InstCount uprBound, DIRECTION schedDir,
                   RLXD_SCHED_TYPE schedType, InstCount maxInstCnt);
  virtual ~RelaxedScheduler();

  // Find a feasible relaxed schedule and return its length
  virtual InstCount FindSchedule() = 0;

  inline bool IsInstFxd(InstCount indx);
  inline void SetInstFxng(InstCount indx);
  void ClearFxng();
};
/*****************************************************************************/

class RJ_RelaxedScheduler : public RelaxedScheduler {
private:
  InstCount chkdInstCnt_;

  void Initialize_(bool setPrirtyLst);
  void InitChkng_(InstCount crntCycle);
  void EndChkng_(InstCount crntCycle);
  bool FixInsts_(LinkedList<SchedInstruction> *fxdLst);

public:
  RJ_RelaxedScheduler(DataDepStruct *dataDepGraph, MachineModel *machMdl,
                      InstCount uprBound, DIRECTION schedDir,
                      RLXD_SCHED_TYPE schedType,
                      InstCount maxInstCnt = INVALID_VALUE);
  ~RJ_RelaxedScheduler();

  inline void Initialize(bool setPrirtyLst);

  // Find a feasible relaxed schedule of all instructions and return its length
  InstCount FindSchedule();

  // Try to schedule all unscheduled instructions between the current cycle and
  // the given last cycle. Return true if this is feasible and false if not
  bool SchdulAndChkFsblty(InstCount crntCycle, InstCount lastCycle);

  bool CmputDynmcLwrBound(InstCount trgtLastCycle, InstCount trgtLwrBound,
                          InstCount &schedLwrBound);

  void SetupPrirtyLst();

  // Fix an instruction in a given cycle. Return true if this is feasible,
  // otherwise return false
  bool FixInst(SchedInstruction *inst, InstCount cycle);

  // Undo the fixing of an inst.
  void UnFixInst(SchedInstruction *inst, InstCount cycle);
};
/*****************************************************************************/

// A relaxed scheduler based on the L&C method (uses Rim & Jain recursively)
class LC_RelaxedScheduler : public RelaxedScheduler {
private:
  // A list of instructions in a subgraph sorted by lower bound measured
  // relative to the sub-graph's leaf.
  PriorityList<SchedInstruction> *subGraphInstLst_;

  void Initialize_();
  void InitSubGraph_();

  // Find a relaxed schedule for the sub-graph rooted at the given instruction
  // in the specified direction.
  InstCount SchdulSubGraph_(SchedInstruction *inst, DIRECTION dir);

  InstCount CmputReleaseTime_(SchedInstruction *inst);

public:
  LC_RelaxedScheduler(DataDepStruct *dataDepGraph, MachineModel *machMdl,
                      InstCount uprBound, DIRECTION schedDir);
  ~LC_RelaxedScheduler();

  // Find a relaxed schedule of all instructions and return its length
  InstCount FindSchedule();
};
/*****************************************************************************/

// A relaxed scheduler based on the LPP method
class LPP_RelaxedScheduler : public RelaxedScheduler {
private:
  // A list of instructions in a subgraph sorted by deadlines
  PriorityList<SchedInstruction> *subGraphInstLst_;

  // An array of current forward lower bounds (w.r.t the schedDir_)
  InstCount *crntFrwrdLwrBounds_;

  // An array of current backward lower bounds (w.r.t the schedDir_)
  InstCount *crntBkwrdLwrBounds_;

  // A list of instructions whose current forward lower bounds are tighter
  // than their permanent forward lower bounds
  LinkedList<SchedInstruction> *tightndFrwrdLwrBounds_;

  LinkedList<SchedInstruction> *tightndBkwrdLwrBounds_;

  KeyedEntry<SchedInstruction> **lstEntries_;

  // Array holding the number of insructions that have not been scheduled
  // for each issue type
  InstCount neededSlots_[MAX_ISSUTYPE_CNT];

  void Initialize_(bool opstFsbl);
  bool InitLength_(InstCount lngth);
  bool InitSubGraph_(SchedInstruction *newInst, InstCount lastCycle);
  void InitCycleProbe_(SchedInstruction *newInst);
  bool CheckSlots_(InstCount firstCycle, InstCount lastCycle);

  InstCount CmputReleaseTime_(SchedInstruction *inst);

  InstCount GetCrntLwrBound_(SchedInstruction *inst, DIRECTION dir);

  void SetCrntLwrBound_(SchedInstruction *inst, DIRECTION dir,
                        InstCount newBound);

  void RestoreLwrBounds_(DIRECTION dir);

  bool ProbeLength_(InstCount trgtLngth);

  // Find a relaxed schedule for the sub-graph rooted at the given instruction
  InstCount SchdulSubGraph_(SchedInstruction *newInst, InstCount trgtLastCycle);
  bool ProbeCycle_(SchedInstruction *newInst, InstCount trgtLastCycle,
                   InstCount trgtCycle);

public:
  LPP_RelaxedScheduler(DataDepStruct *dataDepGraph, MachineModel *machMdl,
                       InstCount uprBound, DIRECTION schedDir);
  ~LPP_RelaxedScheduler();

  // Find a minimum-length relaxed schedule of all instructions and return its
  // length
  InstCount FindSchedule();

  // Find a relaxed schedule of the given length
  // The Boolean determines whether the opposite direction has been probed and
  // found feasible
  bool FindSchedule(InstCount trgtLngth, bool opstFsbl);
};
/*****************************************************************************/

/*****************************************************************************
In line Functions
******************************************************************************/

InstCount RelaxedScheduler::FindNxtAvlblCycle_(IssueType issuType,
                                               InstCount strtCycle) {
  for (InstCount cycleNum = strtCycle; cycleNum < schedUprBound_; cycleNum++) {
    assert(issuType < issuTypeCnt_);

    if (avlblSlots_[issuType][cycleNum] > 0) {
      return cycleNum;
    }
  }

  assert(false);
  return INVALID_VALUE;
}
/*****************************************************************************/

InstCount RelaxedScheduler::CmputDelay_(InstCount schedCycle,
                                        InstCount lastCycle,
                                        InstCount distFrmLeaf) {
  InstCount deadline = lastCycle - distFrmLeaf;
  return (schedCycle <= deadline) ? 0 : (schedCycle - deadline);
}
/*****************************************************************************/

unsigned long RelaxedScheduler::CmputTplgclPrirty_(SchedInstruction *inst,
                                                   DIRECTION dir) {
  assert(dataDepGraph_->GetType() == DGT_FULL);
  // This implementation is valid only for a full graph, since it is based on
  // the total # of neighbors whether they are in the subgraph or not

  unsigned long prirty = GetCrntLwrBound_(inst, dir);
  prirty <<= instCntBits_;
  InstCount rcrsvPredCnt =
      dir == DIR_FRWRD ? inst->GetRcrsvPrdcsrCnt() : inst->GetRcrsvScsrCnt();
  prirty += rcrsvPredCnt;
  return prirty;
}
/*****************************************************************************/

void RelaxedScheduler::SetFix_(SchedInstruction *inst, bool val) {
  assert(useFxng_);
  InstCount indx = dataDepGraph_->GetInstIndx(inst);

  assert(indx < totInstCnt_);
  assert(dataDepGraph_->GetType() == DGT_SUB || indx == inst->GetNum());
  isFxd_[indx] = val;
}
/*****************************************************************************/

void RelaxedScheduler::SetFix_(InstCount indx, bool val) {
  assert(useFxng_);
  assert(indx < totInstCnt_);
  isFxd_[indx] = val;
}
/*****************************************************************************/

bool RelaxedScheduler::GetFix_(SchedInstruction *inst) {
  assert(useFxng_);
  InstCount indx = dataDepGraph_->GetInstIndx(inst);
  assert(indx < totInstCnt_);
  assert(dataDepGraph_->GetType() == DGT_SUB || indx == inst->GetNum());
  return isFxd_[indx];
}
/*****************************************************************************/

bool RelaxedScheduler::IsInstFxd(InstCount indx) {
  assert(useFxng_);
  assert(0 <= indx && indx < totInstCnt_);
  assert(dataDepGraph_->GetType() == DGT_SUB);
  return isFxd_[indx];
}
/*****************************************************************************/

void RelaxedScheduler::SetInstFxng(InstCount indx) {
  assert(useFxng_);
  assert(0 <= indx && indx < totInstCnt_);
  isFxd_[indx] = true;
}
/*****************************************************************************/

void RJ_RelaxedScheduler::Initialize(bool setPrirtyLst) {
  Initialize_(setPrirtyLst);
}
/*****************************************************************************/

} // namespace opt_sched
} // namespace llvm

#endif
