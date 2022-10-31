#include "opt-sched/Scheduler/gen_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/dev_defines.h"

using namespace llvm::opt_sched;

__host__ __device__
InstScheduler::InstScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                             InstCount schedUprBound) {	
  assert(dataDepGraph != NULL);
  assert(machMdl != NULL);
  machMdl_ = machMdl;

  schedUprBound_ = schedUprBound;

  // PATCH.
  schedUprBound_ += 1;

  totInstCnt_ = dataDepGraph->GetInstCnt();
  rootInst_ = dataDepGraph->GetRootInst();
  leafInst_ = dataDepGraph->GetLeafInst();

  issuRate_ = machMdl_->GetIssueRate();
  issuTypeCnt_ = machMdl_->GetIssueTypeCnt();

  schduldInstCnt_ = 0;

  slotsPerTypePerCycle_ = new int[issuTypeCnt_];
  instCntPerIssuType_ = new InstCount[issuTypeCnt_];

  issuTypeCnt_ = machMdl_->GetSlotsPerCycle(slotsPerTypePerCycle_);
  
  dataDepGraph->GetInstCntPerIssuType(instCntPerIssuType_);

  includesUnpipelined_ = dataDepGraph->IncludesUnpipelined();
}

__host__
InstScheduler::~InstScheduler() {
  delete[] slotsPerTypePerCycle_;
  delete[] instCntPerIssuType_;
}

__host__ __device__
void ConstrainedScheduler::AllocRsrvSlots_() {
  rsrvSlots_ = new ReserveSlot[issuRate_];
  ResetRsrvSlots_();
}

__host__ __device__
void ConstrainedScheduler::ResetRsrvSlots_() {
  assert(includesUnpipelined_);
#ifdef __HIP_DEVICE_COMPILE__
  assert(dev_rsrvSlots_[GLOBALTID] != NULL);

  for (int i = 0; i < issuRate_; i++) {
    dev_rsrvSlots_[GLOBALTID][i].strtCycle = INVALID_VALUE;
    dev_rsrvSlots_[GLOBALTID][i].endCycle = INVALID_VALUE;
  }

  dev_rsrvSlotCnt_[GLOBALTID] = 0;
#else
  assert(rsrvSlots_ != NULL);

  for (int i = 0; i < issuRate_; i++) {
    rsrvSlots_[i].strtCycle = INVALID_VALUE;
    rsrvSlots_[i].endCycle = INVALID_VALUE;
  }

  rsrvSlotCnt_ = 0;
#endif
}

__host__
ConstrainedScheduler::ConstrainedScheduler(DataDepGraph *dataDepGraph,
                                           MachineModel *machMdl,
                                           InstCount schedUprBound,
                                           bool ACOEn)
    : InstScheduler(dataDepGraph, machMdl, schedUprBound) { 
  dataDepGraph_ = dataDepGraph;
  IsACO = ACOEn;

  // Allocate the array of first-ready lists - one list per cycle.
  assert(schedUprBound_ > 0);
  frstRdyLstPerCycle_ = new ArrayList<InstCount> *[schedUprBound_];
  frstRdyLstPerCycle_[0] = new ArrayList<InstCount>(dataDepGraph_->GetInstCnt());

  for (InstCount i = 1; i < schedUprBound_; i++) {
    frstRdyLstPerCycle_[i] = NULL;
  }

  rdyLst_ = NULL;
  crntSched_ = NULL;
  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  crntRealSlotNum_ = 0;
  crntCycleNum_ = 0;
  isCrntCycleBlkd_ = false;
  consecEmptyCycles_ = 0;

  avlblSlotsInCrntCycle_ = new int16_t[issuTypeCnt_];

  rsrvSlots_ = NULL;
  rsrvSlotCnt_ = 0;
}

__host__
ConstrainedScheduler::~ConstrainedScheduler() {
  if (crntCycleNum_ < schedUprBound_ &&
      frstRdyLstPerCycle_[crntCycleNum_] != NULL) {
    delete frstRdyLstPerCycle_[crntCycleNum_];
  }
  delete[] frstRdyLstPerCycle_;
  delete[] avlblSlotsInCrntCycle_;
  if (rsrvSlots_)
    delete[] rsrvSlots_;
}

__host__ __device__
bool ConstrainedScheduler::Initialize_(InstCount trgtSchedLngth,
                                       LinkedList<SchedInstruction> *fxdLst) {
  for (int i = 0; i < totInstCnt_; i++) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(i);

    if (!inst->InitForSchdulng(trgtSchedLngth, fxdLst))
      return false;
  }

#ifdef __HIP_DEVICE_COMPILE__

  ResetRsrvSlots_();
  dev_rsrvSlotCnt_[GLOBALTID] = 0;
  dev_schduldInstCnt_[GLOBALTID] = 0;
  dev_crntSlotNum_[GLOBALTID] = 0;
  dev_crntRealSlotNum_[GLOBALTID] = 0;
  dev_crntCycleNum_[GLOBALTID] = 0;
  dev_isCrntCycleBlkd_[GLOBALTID] = false;
#else
  //wipe the ready list per cycle
  for (InstCount i = 0; i<schedUprBound_; ++i) {
    if (frstRdyLstPerCycle_[i])
      frstRdyLstPerCycle_[i]->Reset();
  }

  if (!frstRdyLstPerCycle_[0])
    frstRdyLstPerCycle_[0] = new ArrayList<InstCount>(dataDepGraph_->GetInstCnt());
  frstRdyLstPerCycle_[0]->InsrtElmnt(rootInst_->GetNum());

  if (rsrvSlots_ != NULL) {
    delete[] rsrvSlots_;
    rsrvSlots_ = NULL;
  }

  rsrvSlotCnt_ = 0;
  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  crntRealSlotNum_ = 0;
  crntCycleNum_ = 0;
  isCrntCycleBlkd_ = false;
  consecEmptyCycles_ = 0;
#endif

  InitNewCycle_();

#ifdef __HIP_DEVICE_COMPILE__
  ((BBWithSpill *)dev_rgn_)->Dev_InitForSchdulng();
#else
  rgn_->InitForSchdulng();
#endif

  return true;
}

__host__ __device__
void ConstrainedScheduler::SchdulInst_(SchedInstruction *inst, InstCount) {
#ifdef __HIP_DEVICE_COMPILE__  // Device version
  InstCount prdcsrNum, scsrRdyCycle;//, scsrRdyListNum;

  // Notify each successor of this instruction that it has been scheduled.
  if(!IsACO) {
    int i = 0;
    for (SchedInstruction *crntScsr = inst->GetScsr(i++, &prdcsrNum);
          crntScsr != NULL; crntScsr = inst->GetScsr(i++, &prdcsrNum)) {
      crntScsr->PrdcsrSchduld(prdcsrNum, dev_crntCycleNum_[GLOBALTID], scsrRdyCycle);
    }
  }
  if (inst->BlocksCycle()) {
    dev_isCrntCycleBlkd_[GLOBALTID] = true;
  }
  dev_schduldInstCnt_[GLOBALTID]++;
#else  // Host version
  InstCount prdcsrNum, scsrRdyCycle;

  // Notify each successor of this instruction that it has been scheduled.
  if(!IsACO) {
    for (SchedInstruction *crntScsr = inst->GetFrstScsr(&prdcsrNum);
        crntScsr != NULL; crntScsr = inst->GetNxtScsr(&prdcsrNum)) {
      bool wasLastPrdcsr =
          crntScsr->PrdcsrSchduld(prdcsrNum, crntCycleNum_, scsrRdyCycle);

      if (wasLastPrdcsr) {
        // If all other predecessors of this successor have been scheduled then
        // we now know in which cycle this successor will become ready.
        assert(scsrRdyCycle < schedUprBound_);
        // If the first-ready list of that cycle has not been created yet.
        if (frstRdyLstPerCycle_[scsrRdyCycle] == NULL) {
          frstRdyLstPerCycle_[scsrRdyCycle] =
                  new ArrayList<InstCount>(dataDepGraph_->GetInstCnt());
        }
        // Add this succesor to the first-ready list of the future cycle
        // in which we now know it will become ready
        frstRdyLstPerCycle_[scsrRdyCycle]->InsrtElmnt(crntScsr->GetNum());
      }
    }
  }

  if (inst->BlocksCycle()) {
    isCrntCycleBlkd_ = true;
  }
  
  schduldInstCnt_++;
#endif
}

__host__ __device__
void ConstrainedScheduler::UnSchdulInst_(SchedInstruction *inst) {
  InstCount prdcsrNum, scsrRdyCycle;

  assert(inst->IsSchduld());

  // Notify each successor of this instruction that it has been unscheduled.
  // The successors are visted in the reverse order so that each one will be
  // at the bottom of its first-ready list (if the scheduling of this
  // instruction has caused it to go there).
  #ifdef __HIP_DEVICE_COMPILE__
  int lastScsrNum = inst->GetScsrCnt_();
    for (SchedInstruction *crntScsr = inst->GetScsr(lastScsrNum--, &prdcsrNum);
          crntScsr != NULL && lastScsrNum > -1; crntScsr = inst->GetScsr(lastScsrNum--, &prdcsrNum)) {
    bool wasLastPrdcsr = crntScsr->PrdcsrUnSchduld(prdcsrNum, scsrRdyCycle);

    if (wasLastPrdcsr) {
      // If this predecessor was the last to schedule and thus resolved the
      // cycle in which this successor will become ready, then this successor
      // must now be taken out of the first ready list for the cycle in which
      // the scheduling of this instruction has made it ready.
      assert(scsrRdyCycle < schedUprBound_);
      assert(frstRdyLstPerCycle_[scsrRdyCycle] != NULL);
      frstRdyLstPerCycle_[scsrRdyCycle]->RmvElmnt(crntScsr->GetNum());
    }
  }
  #else
    for (SchedInstruction *crntScsr = inst->GetLastScsr(&prdcsrNum);
          crntScsr != NULL; crntScsr = inst->GetPrevScsr(&prdcsrNum)) {
    bool wasLastPrdcsr = crntScsr->PrdcsrUnSchduld(prdcsrNum, scsrRdyCycle);

    if (wasLastPrdcsr) {
      // If this predecessor was the last to schedule and thus resolved the
      // cycle in which this successor will become ready, then this successor
      // must now be taken out of the first ready list for the cycle in which
      // the scheduling of this instruction has made it ready.
      assert(scsrRdyCycle < schedUprBound_);
      assert(frstRdyLstPerCycle_[scsrRdyCycle] != NULL);
      frstRdyLstPerCycle_[scsrRdyCycle]->RmvElmnt(crntScsr->GetNum());
    }
  }
  #endif

  schduldInstCnt_--;
}

__host__ __device__
void ConstrainedScheduler::DoRsrvSlots_(SchedInstruction *inst) {
  if (inst == NULL)
    return;

  if (!inst->IsPipelined()) {
#ifdef __HIP_DEVICE_COMPILE__
    rsrvSlots_[dev_crntSlotNum_[GLOBALTID]].strtCycle = 
	    dev_crntCycleNum_[GLOBALTID];
    rsrvSlots_[dev_crntSlotNum_[GLOBALTID]].endCycle = 
	    dev_crntCycleNum_[GLOBALTID] + inst->GetMaxLtncy() - 1;
    dev_rsrvSlotCnt_[GLOBALTID]++;
#else
    if (rsrvSlots_ == NULL)
      AllocRsrvSlots_();
    rsrvSlots_[crntSlotNum_].strtCycle = crntCycleNum_;
    rsrvSlots_[crntSlotNum_].endCycle = crntCycleNum_ + inst->GetMaxLtncy() - 1;
    rsrvSlotCnt_++;
#endif
  }
}

__host__ __device__
void ConstrainedScheduler::UndoRsrvSlots_(SchedInstruction *inst) {
  if (inst == NULL)
    return;

  if (!inst->IsPipelined()) {
    assert(rsrvSlots_ != NULL);
    rsrvSlots_[inst->GetSchedSlot()].strtCycle = INVALID_VALUE;
    rsrvSlots_[inst->GetSchedSlot()].endCycle = INVALID_VALUE;
    rsrvSlotCnt_--;
  }
}

__host__ __device__
bool InstScheduler::IsSchedComplete_() {
#ifdef __HIP_DEVICE_COMPILE__
  return dev_schduldInstCnt_[GLOBALTID] == totInstCnt_;
#else
  return schduldInstCnt_ == totInstCnt_;
#endif
}

__host__ __device__
void ConstrainedScheduler::InitNewCycle_() {
#ifdef __HIP_DEVICE_COMPILE__
  assert(dev_crntSlotNum_[GLOBALTID] == 0 && 
         dev_crntRealSlotNum_[GLOBALTID] == 0);
  for (int i = 0; i < issuTypeCnt_; i++) {
    dev_avlblSlotsInCrntCycle_[GLOBALTID][i] = slotsPerTypePerCycle_[i];
  }
  dev_isCrntCycleBlkd_[GLOBALTID] = false;
#else
  assert(crntSlotNum_ == 0 && crntRealSlotNum_ == 0);
  for (int i = 0; i < issuTypeCnt_; i++) {
    avlblSlotsInCrntCycle_[i] = slotsPerTypePerCycle_[i];
  }
  isCrntCycleBlkd_ = false;
#endif
}

__host__ __device__
bool ConstrainedScheduler::MovToNxtSlot_(SchedInstruction *inst) {
#ifdef __HIP_DEVICE_COMPILE__
  // If we are currently in the last slot of the current cycle.
  if (dev_crntSlotNum_[GLOBALTID] == (issuRate_ - 1)) {
    dev_crntCycleNum_[GLOBALTID]++;
    dev_crntSlotNum_[GLOBALTID] = 0;
    dev_crntRealSlotNum_[GLOBALTID] = 0;
    return true;
  } else {
    dev_crntSlotNum_[GLOBALTID]++;
    if (inst && machMdl_->IsRealInst(inst->GetInstType()))
      dev_crntRealSlotNum_[GLOBALTID]++;
    return false;
  }
#else
  // If we are currently in the last slot of the current cycle.
  if (crntSlotNum_ == (issuRate_ - 1)) {
    crntCycleNum_++;
    crntSlotNum_ = 0;
    crntRealSlotNum_ = 0;
    return true;
  } else {
    crntSlotNum_++;
    if (inst && machMdl_->IsRealInst(inst->GetInstType()))
      crntRealSlotNum_++;
    return false;
  }
#endif
}

__host__ __device__
bool ConstrainedScheduler::MovToPrevSlot_(int prevRealSlotNum) {
#ifdef __HIP_DEVICE_COMPILE__
  dev_crntRealSlotNum_[GLOBALTID] = prevRealSlotNum;

  // If we are currently in the last slot of the current cycle.
  if (dev_crntSlotNum_[GLOBALTID] == 0) {
    dev_crntCycleNum_[GLOBALTID]--;
    dev_crntSlotNum_[GLOBALTID] = issuRate_ - 1;
    return true;
  } else {
    dev_crntSlotNum_[GLOBALTID]--;
    // if (inst && machMdl_->IsRealInst(inst->GetInstType()))
    // crntRealSlotNum_--;
    return false;
  }
#else
  crntRealSlotNum_ = prevRealSlotNum;

  // If we are currently in the last slot of the current cycle.
  if (crntSlotNum_ == 0) {
    crntCycleNum_--;
    crntSlotNum_ = issuRate_ - 1;
    return true;
  } else {
    crntSlotNum_--;
    // if (inst && machMdl_->IsRealInst(inst->GetInstType()))
    // crntRealSlotNum_--;
    return false;
  }
#endif
}

void ConstrainedScheduler::CleanupCycle_(InstCount cycleNum) {
  if (frstRdyLstPerCycle_[cycleNum] != NULL) {
    delete frstRdyLstPerCycle_[cycleNum];
    frstRdyLstPerCycle_[cycleNum] = NULL;
  }
}

__host__ __device__
bool ConstrainedScheduler::IsTriviallyLegal_(
    const SchedInstruction *inst) const {
  // Scheduling a stall is always legal.
  if (inst == NULL)
    return true;

  // Artificial root and leaf instructions can only be in the ready list if all
  // other dependencies have been satisfied. They are fixed in the first or last
  // slot.
  if (inst->IsRoot() || inst->IsLeaf())
    return true;
  return false;
}

__host__ __device__
bool ConstrainedScheduler::ChkInstLglty_(SchedInstruction *inst) const {
  if (IsTriviallyLegal_(inst))
    return true;

  //rgn_->ChkInstLglty(inst) is defined to always return true
  //so I am removing this statement for now
  // Do region-specific legality check
  //if (rgn_->ChkInstLglty(inst) == false)
    //return false;
#ifdef __HIP_DEVICE_COMPILE__
   // Account for instructions that block the whole cycle.
  if (dev_isCrntCycleBlkd_[GLOBALTID])
    return false;
  // Logger::Info("Cycle not blocked");
  if (inst->BlocksCycle() && dev_crntSlotNum_[GLOBALTID] != 0)
    return false;
  // Logger::Info("Does not block cycle");
  if (includesUnpipelined_ && dev_rsrvSlots_[GLOBALTID] &&
      dev_rsrvSlots_[GLOBALTID][dev_crntSlotNum_[GLOBALTID]].strtCycle != INVALID_VALUE &&
      dev_crntCycleNum_[GLOBALTID] <= 
      dev_rsrvSlots_[GLOBALTID][dev_crntSlotNum_[GLOBALTID]].endCycle) {
    return false;
  }

  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);
  assert(dev_avlblSlotsInCrntCycle_[GLOBALTID][issuType] >= 0);
  // Logger::Info("avlblSlots = %d", avlblSlotsInCrntCycle_[issuType]);
  return (dev_avlblSlotsInCrntCycle_[GLOBALTID][issuType] > 0);
#else
  // Account for instructions that block the whole cycle.
  if (isCrntCycleBlkd_)
    return false;
  // Logger::Info("Cycle not blocked");
  if (inst->BlocksCycle() && crntSlotNum_ != 0)
    return false;
  // Logger::Info("Does not block cycle");
  if (includesUnpipelined_ && rsrvSlots_ &&
      rsrvSlots_[crntSlotNum_].strtCycle != INVALID_VALUE &&
      crntCycleNum_ <= rsrvSlots_[crntSlotNum_].endCycle) {
    return false;
  }

  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);
  assert(avlblSlotsInCrntCycle_[issuType] >= 0);
  // Logger::Info("avlblSlots = %d", avlblSlotsInCrntCycle_[issuType]);
  return (avlblSlotsInCrntCycle_[issuType] > 0);
#endif
}

__host__ __device__
bool ConstrainedScheduler::ChkSchedLglty_(bool isEmptyCycle) {
  if (isEmptyCycle)
    consecEmptyCycles_++;
  else
    consecEmptyCycles_ = 0;
  return consecEmptyCycles_ <= dataDepGraph_->GetMaxLtncy();
}

__host__ __device__
void ConstrainedScheduler::UpdtSlotAvlblty_(SchedInstruction *inst) {
  if (inst == NULL)
    return;
  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);
#ifdef __HIP_DEVICE_COMPILE__
  assert(dev_avlblSlotsInCrntCycle_[GLOBALTID][issuType] > 0);
  dev_avlblSlotsInCrntCycle_[GLOBALTID][issuType]--;
#else
  assert(avlblSlotsInCrntCycle_[issuType] > 0);
  avlblSlotsInCrntCycle_[issuType]--;
#endif
}
