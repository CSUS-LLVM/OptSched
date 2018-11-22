#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/sched_region.h"

using namespace llvm::opt_sched;

ListScheduler::ListScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                             InstCount schedUprBound, SchedPriorities prirts)
    : ConstrainedScheduler(dataDepGraph, machMdl, schedUprBound) {
  crntSched_ = NULL;

  prirts_ = prirts;
  rdyLst_ = new ReadyList(dataDepGraph_, prirts);
  if (rdyLst_ == NULL)
    Logger::Fatal("Out of memory.");
}

ListScheduler::~ListScheduler() { delete rdyLst_; }

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
    // if(dataDepGraph_->GetInstCnt() > 1000)
    // Logger::Info("ready list size = %d", rdyLstSize);

    SchedInstruction *inst = NULL;
    bool legalInst = false;
    int lgltyChkCnt = 0;
    while (!legalInst) {
      lgltyChkCnt++;
      inst = rdyLst_->GetNextPriorityInst();
      legalInst = ChkInstLglty_(inst);
    }

#ifdef IS_DEBUG_MODEL
    Logger::Info("Legality checks made: %d", lgltyChkCnt);
    stats::legalListSchedulerInstructionHits++;
    stats::illegalListSchedulerInstructionHits += (lgltyChkCnt - 1);
#endif

    InstCount instNum;
    // If the ready list is empty.
    if (inst == NULL) {
      instNum = SCHD_STALL;
    } else {
      isEmptyCycle = false;
      instNum = inst->GetNum();
      //      Logger::Info("Scheduling inst %d", instNum);
      SchdulInst_(inst, crntCycleNum_);
      inst->Schedule(crntCycleNum_, crntSlotNum_);
      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
      DoRsrvSlots_(inst);
      rdyLst_->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);
    }

    // if (inst && machMdl_->IsRealInst(inst->GetInstType())) {
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
