#include "opt-sched/Scheduler/gen_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/sched_region.h"

using namespace llvm::opt_sched;

InstScheduler::InstScheduler(DataDepStruct *dataDepGraph, MachineModel *machMdl,
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

InstScheduler::~InstScheduler() {
  delete[] slotsPerTypePerCycle_;
  delete[] instCntPerIssuType_;
}

void ConstrainedScheduler::AllocRsrvSlots_() {
  rsrvSlots_ = new ReserveSlot[issuRate_];
  ResetRsrvSlots_();
}

void ConstrainedScheduler::ResetRsrvSlots_() {
  assert(includesUnpipelined_);
  assert(rsrvSlots_ != NULL);

  for (int i = 0; i < issuRate_; i++) {
    rsrvSlots_[i].strtCycle = INVALID_VALUE;
    rsrvSlots_[i].endCycle = INVALID_VALUE;
  }

  rsrvSlotCnt_ = 0;
}

ConstrainedScheduler::ConstrainedScheduler(DataDepGraph *dataDepGraph,
                                           MachineModel *machMdl,
                                           InstCount schedUprBound)
    : InstScheduler(dataDepGraph, machMdl, schedUprBound) {
  dataDepGraph_ = dataDepGraph;

  // Allocate the array of first-ready lists - one list per cycle.
  assert(schedUprBound_ > 0);
  frstRdyLstPerCycle_ = new LinkedList<SchedInstruction> *[schedUprBound_];

  for (InstCount i = 0; i < schedUprBound_; i++) {
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

bool ConstrainedScheduler::Initialize_(InstCount trgtSchedLngth,
                                       LinkedList<SchedInstruction> *fxdLst) {
  for (int i = 0; i < totInstCnt_; i++) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(i);
    if (!inst->InitForSchdulng(trgtSchedLngth, fxdLst))
      return false;
  }


  //wipe the ready list per cycle
  for (InstCount i = 0; i<schedUprBound_; ++i) {
    if (frstRdyLstPerCycle_[i])
      frstRdyLstPerCycle_[i]->Reset();
  }

  // Allocate the first entry in the array.
  if (frstRdyLstPerCycle_[0] == NULL) {
    frstRdyLstPerCycle_[0] = new LinkedList<SchedInstruction>;
  }

  frstRdyLstPerCycle_[0]->InsrtElmnt(rootInst_);

  if (rsrvSlots_ != NULL) {
    delete[] rsrvSlots_;
    rsrvSlots_ = NULL;
  }

  rsrvSlotCnt_ = 0;

  // Dynamic data.
  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  crntRealSlotNum_ = 0;
  crntCycleNum_ = 0;
  isCrntCycleBlkd_ = false;
  consecEmptyCycles_ = 0;

  InitNewCycle_();
  rgn_->InitForSchdulng();

  return true;
}

void ConstrainedScheduler::SchdulInst_(SchedInstruction *inst, InstCount) {
  InstCount prdcsrNum, scsrRdyCycle;

  // Notify each successor of this instruction that it has been scheduled.
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
        frstRdyLstPerCycle_[scsrRdyCycle] = new LinkedList<SchedInstruction>;
      }

      // Add this successor to the first-ready list of the future cycle
      // in which we now know it will become ready
      frstRdyLstPerCycle_[scsrRdyCycle]->InsrtElmnt(crntScsr);
    }
  }

  if (inst->BlocksCycle()) {
    isCrntCycleBlkd_ = true;
  }

  schduldInstCnt_++;
}

void ConstrainedScheduler::UnSchdulInst_(SchedInstruction *inst) {
  InstCount prdcsrNum, scsrRdyCycle;

  assert(inst->IsSchduld());

  // Notify each successor of this instruction that it has been unscheduled.
  // The successors are visited in the reverse order so that each one will be
  // at the bottom of its first-ready list (if the scheduling of this
  // instruction has caused it to go there).
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
      frstRdyLstPerCycle_[scsrRdyCycle]->RmvElmnt(crntScsr);
    }
  }

  schduldInstCnt_--;
}

void ConstrainedScheduler::DoRsrvSlots_(SchedInstruction *inst) {
  if (inst == NULL)
    return;

  if (!inst->IsPipelined()) {
    if (rsrvSlots_ == NULL)
      AllocRsrvSlots_();
    rsrvSlots_[crntSlotNum_].strtCycle = crntCycleNum_;
    rsrvSlots_[crntSlotNum_].endCycle = crntCycleNum_ + inst->GetMaxLtncy() - 1;
    rsrvSlotCnt_++;
  }
}

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

bool InstScheduler::IsSchedComplete_() {
  return schduldInstCnt_ == totInstCnt_;
}

void ConstrainedScheduler::InitNewCycle_() {
  assert(crntSlotNum_ == 0 && crntRealSlotNum_ == 0);
  for (int i = 0; i < issuTypeCnt_; i++) {
    avlblSlotsInCrntCycle_[i] = slotsPerTypePerCycle_[i];
  }
  isCrntCycleBlkd_ = false;
}

bool ConstrainedScheduler::MovToNxtSlot_(SchedInstruction *inst) {
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
}

bool ConstrainedScheduler::MovToPrevSlot_(int prevRealSlotNum) {
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
}

void ConstrainedScheduler::CleanupCycle_(InstCount cycleNum) {
  if (frstRdyLstPerCycle_[cycleNum] != NULL) {
    delete frstRdyLstPerCycle_[cycleNum];
    frstRdyLstPerCycle_[cycleNum] = NULL;
  }
}

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

bool ConstrainedScheduler::ChkInstLglty_(SchedInstruction *inst) const {
  if (IsTriviallyLegal_(inst))
    return true;

  // Do region-specific legality check
  if (rgn_->ChkInstLglty(inst) == false)
    return false;

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
}

bool ConstrainedScheduler::ChkSchedLglty_(bool isEmptyCycle) {
  if (isEmptyCycle)
    consecEmptyCycles_++;
  else
    consecEmptyCycles_ = 0;
  return consecEmptyCycles_ <= dataDepGraph_->GetMaxLtncy();
}

void ConstrainedScheduler::UpdtSlotAvlblty_(SchedInstruction *inst) {
  if (inst == NULL)
    return;
  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);
  assert(avlblSlotsInCrntCycle_[issuType] > 0);
  avlblSlotsInCrntCycle_[issuType]--;
}
