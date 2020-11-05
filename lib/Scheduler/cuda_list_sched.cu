#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/bb_spill.h"

using namespace llvm::opt_sched;

__host__ __device__
ListScheduler::ListScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                             InstCount schedUprBound, SchedPriorities prirts)
    : ConstrainedScheduler(dataDepGraph, machMdl, schedUprBound) {
  
  crntSched_ = NULL;

  prirts_ = prirts;

  rdyLst_ = new ReadyList(dataDepGraph_, prirts);
}

__host__ __device__
ListScheduler::~ListScheduler() { 
  delete rdyLst_; 
}

__host__ __device__
SchedInstruction *ListScheduler::PickInst() const {
  SchedInstruction *inst = NULL;
  bool legalInst = false;
  while (!legalInst) {
    inst = rdyLst_->GetNextPriorityInst();
    legalInst = ChkInstLglty_(inst);
  }
  return inst;
}

__host__ __device__
FUNC_RESULT ListScheduler::FindSchedule(InstSchedule *sched, SchedRegion *rgn) {
  InstCount rdyLstSize, maxRdyLstSize = 0, avgRdyLstSize = 0, iterCnt = 0;
  bool isEmptyCycle = true;

  crntSched_ = sched;
  rgn_ = rgn;

  Initialize_();

  while (!IsSchedComplete_()) {
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
    rdyLst_->ResetIterator();

    //debug
    //rdyLst_->Dev_Print();

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

      //Bypass polymorphism since virtual functions cannot be invoked on device
      //we know rgn_ is always BBWithSpill since it is the only child of 
      //SchedRegion
#ifdef __CUDA_ARCH__
      ((BBWithSpill *)rgn_)->Dev_SchdulInst(inst, crntCycleNum_, crntSlotNum_,
	                                    false);
#else
      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
#endif

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
  crntSched_->Print();
#endif

  return RES_SUCCESS;
}

void ListScheduler::CopyPointersToDevice(ListScheduler *dev_lstSchedulr,
                                        DataDepGraph *dev_DDG,
                                        MachineModel *dev_machMdl) {
  size_t memSize;
  dev_lstSchedulr->machMdl_ = dev_machMdl;
  dev_lstSchedulr->dataDepGraph_ = dev_DDG;
  // Copy slotsPerTypePerCycle_
  int *dev_slotsPerTypePerCycle;
  memSize = sizeof(int) * issuTypeCnt_;
  if (cudaSuccess != cudaMalloc(&dev_slotsPerTypePerCycle, memSize))
    Logger::Fatal("Failed to alloc dev mem for slotsPerTypePerCycle");

  if (cudaSuccess != cudaMemcpy(dev_slotsPerTypePerCycle, slotsPerTypePerCycle_,
			        memSize, cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy slotspertypepercycle to device");

  dev_lstSchedulr->slotsPerTypePerCycle_ = dev_slotsPerTypePerCycle;

  // Copy instCntPerIssuType_
  InstCount *dev_instCntPerIssuType;
  memSize = sizeof(InstCount) * issuTypeCnt_;
  if (cudaSuccess != cudaMalloc(&dev_instCntPerIssuType, memSize))
    Logger::Fatal("Failed to alloc dev mem for instCntPerIssuType_");
  
  if (cudaSuccess != cudaMemcpy(dev_instCntPerIssuType, instCntPerIssuType_,
			        memSize, cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy instCntPerIssuType_ to device");

  dev_lstSchedulr->instCntPerIssuType_ = dev_instCntPerIssuType;

  // set root/leaf inst
  dev_lstSchedulr->rootInst_ = dev_DDG->GetRootInst();
  dev_lstSchedulr->leafInst_ = dev_DDG->GetLeafInst();

  // Copy frstRdyLstPerCycle_, an array of NULL values
  ArrayList<InstCount> **dev_frstRdyLstPerCycle;
  memSize = sizeof(ArrayList<InstCount> *) * schedUprBound_;
  if (cudaSuccess != cudaMalloc(&dev_frstRdyLstPerCycle, memSize))
    Logger::Fatal("Failed to alloc dev mem for dev_frstRdyLstPerCycle");

  if (cudaSuccess != cudaMemcpy(dev_frstRdyLstPerCycle, frstRdyLstPerCycle_,
			        memSize, cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy frstRdyLstPerCycle_ to device");

  dev_lstSchedulr->frstRdyLstPerCycle_ = dev_frstRdyLstPerCycle;

  // Copy avlblSlotsInCrntCycle_
  int16_t *dev_avlblSlotsInCrntCycle;
  memSize = sizeof(int16_t) * issuTypeCnt_;
  if (cudaSuccess != cudaMalloc(&dev_avlblSlotsInCrntCycle, memSize))
    Logger::Fatal("Failed to alloc dev mem for avlblSlotsInCrntCycle_");

  if (cudaSuccess != cudaMemcpy(dev_avlblSlotsInCrntCycle, 
			        avlblSlotsInCrntCycle_, memSize,
				cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy avlblSlotsInCrntCycle_ to device");

  dev_lstSchedulr->avlblSlotsInCrntCycle_ = dev_avlblSlotsInCrntCycle;

  // Copy rdyLst_
  ReadyList *dev_rdyLst;
  memSize = sizeof(ReadyList);
  if (cudaSuccess != cudaMallocManaged(&dev_rdyLst, memSize))
    Logger::Fatal("Failed to alloc dev mem for rdyLst_");

  if (cudaSuccess != cudaMemcpy(dev_rdyLst, rdyLst_, memSize,
			        cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy rdyLst_ to device");

  rdyLst_->CopyPointersToDevice(dev_rdyLst, dev_DDG);

  dev_lstSchedulr->rdyLst_ = dev_rdyLst;
}

void ListScheduler::FreeDevicePointers() {
  cudaFree(slotsPerTypePerCycle_);
  cudaFree(instCntPerIssuType_);
  cudaFree(frstRdyLstPerCycle_);
  cudaFree(avlblSlotsInCrntCycle_);
  rdyLst_->FreeDevicePointers();
  cudaFree(rdyLst_);
}

__host__ __device__
SequentialListScheduler::SequentialListScheduler(DataDepGraph *dataDepGraph,
                                                 MachineModel *machMdl,
                                                 InstCount schedUprBound,
                                                 SchedPriorities prirts)
    : ListScheduler(dataDepGraph, machMdl, schedUprBound, prirts) {}

__host__ __device__
bool SequentialListScheduler::ChkInstLglty_(SchedInstruction *inst) const {
  if (IsTriviallyLegal_(inst))
    return true;

  if (!IsSequentialInstruction(inst))
    return false;

  // Do region-specific legality check
  // currently always returns true
  //if (rgn_->ChkInstLglty(inst) == false)
    //return false;

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

__host__ __device__
bool SequentialListScheduler::IsSequentialInstruction(
    const SchedInstruction *Inst) const {
  // Instr with number N-1 must already be scheduled.

  // In the case of Inst num 0, its predecessor (OptSched root) is always
  // scheduled	
  if (Inst->GetNum() == 0)
    return true;

  return crntSched_->GetSchedCycle(Inst->GetNum() - 1) != SCHD_UNSCHDULD;
}

__host__ __device__
void ListScheduler::UpdtRdyLst_(InstCount cycleNum, int slotNum) {
  InstCount prevCycleNum = cycleNum - 1;
  ArrayList<InstCount> *lst1 = NULL;
  ArrayList<InstCount> *lst2 = frstRdyLstPerCycle_[cycleNum];

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
