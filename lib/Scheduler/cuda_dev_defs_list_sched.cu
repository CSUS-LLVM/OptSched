//This file contains all of the implementations for device functions
//for cuda_list_sched.cu, in order to work around the lack of seperable
//compilation and linking of host and device code.

#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/stats.h"

using namespace llvm::opt_sched;

__device__
void ListScheduler::Dev_UpdtRdyLst_(InstCount cycleNum, int slotNum) {
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

__device__
void ListScheduler::Call_Dev_UpdtRdyLst_(){
  Dev_UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
}

