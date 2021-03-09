#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/stats.h"

using namespace llvm::opt_sched;

ListScheduler::ListScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                             InstCount schedUprBound, SchedPriorities prirts)
    : ConstrainedScheduler(dataDepGraph, machMdl, schedUprBound) {
  crntSched_ = NULL;

  prirts_ = prirts;
  rdyLst_ = new ReadyList(dataDepGraph_, prirts);
}

ListScheduler::~ListScheduler() { delete rdyLst_; }

SchedInstruction *ListScheduler::PickInst() const {
  SchedInstruction *inst = NULL;
  bool legalInst = false;
  while (!legalInst) {
    inst = rdyLst_->GetNextPriorityInst();
    legalInst = ChkInstLglty_(inst);
  }
  return inst;
}

FUNC_RESULT ListScheduler::FindSchedule(InstSchedule *sched, SchedRegion *rgn) {
  InstCount rdyLstSize, maxRdyLstSize = 0, avgRdyLstSize = 0, iterCnt = 0;
  bool isEmptyCycle = true;

  crntSched_ = sched;
  rgn_ = rgn;

  Initialize_();

  while (!IsSchedComplete_()) {
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
    rdyLst_->ResetIterator();

    iterCnt++;
    rdyLstSize = rdyLst_->GetInstCnt();
    if (rdyLstSize > maxRdyLstSize)
      maxRdyLstSize = rdyLstSize;
    avgRdyLstSize += rdyLstSize;

    SchedInstruction *inst = PickInst();

    InstCount instNum;
    // If the ready list is empty.
    if (inst == NULL) {
      instNum = SCHD_STALL;
    } else {
      isEmptyCycle = false;
      instNum = inst->GetNum();
      SchdulInst_(inst, crntCycleNum_);
      inst->Schedule(crntCycleNum_, crntSlotNum_);
      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
      DoRsrvSlots_(inst);
      rdyLst_->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);
    }

    crntSched_->AppendInst(instNum);
    bool cycleAdvanced = MovToNxtSlot_(inst);
    if (cycleAdvanced) {
      bool schedIsLegal = ChkSchedLglty_(isEmptyCycle);
      if (!schedIsLegal)
        return RES_ERROR;

      InitNewCycle_();
      isEmptyCycle = true;
    }
  }

#ifdef IS_DEBUG_SCHED
  crntSched_->Print(Logger::GetLogStream(), " ");
#endif

  return RES_SUCCESS;
}

SequentialListScheduler::SequentialListScheduler(DataDepGraph *dataDepGraph,
                                                 MachineModel *machMdl,
                                                 InstCount schedUprBound,
                                                 SchedPriorities prirts)
    : ListScheduler(dataDepGraph, machMdl, schedUprBound, prirts) {}

bool SequentialListScheduler::ChkInstLglty_(SchedInstruction *inst) const {
  if (IsTriviallyLegal_(inst))
    return true;

  if (!IsSequentialInstruction(inst))
    return false;

  // Do region-specific legality check
  if (rgn_->ChkInstLglty(inst) == false)
    return false;

  // Account for instructions that block the whole cycle.
  if (isCrntCycleBlkd_)
    return false;

  if (inst->BlocksCycle() && crntSlotNum_ != 0)
    return false;

  if (includesUnpipelined_ && rsrvSlots_ &&
      rsrvSlots_[crntSlotNum_].strtCycle != INVALID_VALUE &&
      crntCycleNum_ <= rsrvSlots_[crntSlotNum_].endCycle) {
    return false;
  }

  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);
  assert(avlblSlotsInCrntCycle_[issuType] >= 0);
  return (avlblSlotsInCrntCycle_[issuType] > 0);
}

bool SequentialListScheduler::IsSequentialInstruction(
    const SchedInstruction *Inst) const {
  // Instr with number 0 is always sequential and legal to schedule
  // since its predecessor is OptSched artificial root which is always
  // trivially legal
  if (Inst->GetNum() == 0)
    return true;
  // Instr with number N-1 must already be scheduled.
  return crntSched_->GetSchedCycle(Inst->GetNum() - 1) != SCHD_UNSCHDULD;
}

void ListScheduler::UpdtRdyLst_(InstCount cycleNum, int slotNum) {
  InstCount prevCycleNum = cycleNum - 1;
  LinkedList<SchedInstruction> *lst1 = NULL;
  LinkedList<SchedInstruction> *lst2 = frstRdyLstPerCycle_[cycleNum];

  if (prirts_.isDynmc)
    rdyLst_->UpdatePriorities();

  if (slotNum == 0 && prevCycleNum >= 0) {
    // If at the begining of a new cycle other than the very first cycle,
    // then we also have to include the instructions that might have become
    // ready in the previous cycle due to a zero latency of the instruction
    // scheduled in the very last slot of that cycle [GOS 9.8.02].
    lst1 = frstRdyLstPerCycle_[prevCycleNum];

    if (lst1 != NULL) {
      rdyLst_->AddList(lst1);
      lst1->Reset();
      CleanupCycle_(prevCycleNum);
    }
  }

  if (lst2 != NULL) {
    rdyLst_->AddList(lst2);
    lst2->Reset();
  }
}

StallSchedulingListScheduler::StallSchedulingListScheduler(
    DataDepGraph *dataDepGraph, MachineModel *machMdl, InstCount schedUprBound,
    SchedPriorities prirts)
    : ListScheduler(dataDepGraph, machMdl, schedUprBound, prirts) {}

SchedInstruction *StallSchedulingListScheduler::PickInst() const {
  unsigned long CurrentHeuristic;
  SchedInstruction *inst = rdyLst_->GetNextPriorityInst(CurrentHeuristic);

  // now inst stores the latency ready instruction w/ the best heuristic
  for (InstCount fCycle = 1; fCycle < dataDepGraph_->GetMaxLtncy() &&
                             crntCycleNum_ + fCycle < schedUprBound_;
       ++fCycle) {
    LinkedList<SchedInstruction> *futureReady =
        frstRdyLstPerCycle_[crntCycleNum_ + fCycle];
    if (!futureReady)
      continue;

    for (SchedInstruction *fIns = futureReady->GetFrstElmnt(); fIns;
         fIns = futureReady->GetNxtElmnt()) {
      bool Changed;
      unsigned long Heuristic = rdyLst_->CmputKey_(fIns, false, Changed);
      if (Heuristic > CurrentHeuristic) {
        futureReady->ResetIterator();
        return NULL;
      }
    }
    futureReady->ResetIterator();
  }

  return ChkInstLglty_(inst) ? inst : NULL;
}
