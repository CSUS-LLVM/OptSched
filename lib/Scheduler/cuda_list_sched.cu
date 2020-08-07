#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/stats.h"

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
ListScheduler::~ListScheduler() { delete rdyLst_; }

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

  //debug
  printf("ListScheduler::FindSchedule: Initializing\n");

  Initialize_();

  while (!IsSchedComplete_()) {
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
    rdyLst_->ResetIterator();

    iterCnt++;
    rdyLstSize = rdyLst_->GetInstCnt();
    if (rdyLstSize > maxRdyLstSize)
      maxRdyLstSize = rdyLstSize;
    avgRdyLstSize += rdyLstSize;

    //debug
    printf("ListScheduler::FindSchedule: Picking inst\n");

    SchedInstruction *inst = PickInst();

    //debug
    printf("ListScheduler::FindSchedule: Picked inst: %d\n", inst->GetNum());

    InstCount instNum;
    // If the ready list is empty.
    if (inst == NULL) {
      instNum = SCHD_STALL;
    } else {
      isEmptyCycle = false;
      instNum = inst->GetNum();

      //debug
      printf("ListScheduler::FindSchedule: Calling SchdulInst_\n");

      SchdulInst_(inst, crntCycleNum_);

      //debug
      printf("ListScheduler::FindSchedule: Done with SchdulInst_\n");

      //debug
      printf("ListScheduler::FindSchedule: Calling inst->Schedule\n");

      inst->Schedule(crntCycleNum_, crntSlotNum_);

      //debug
      printf("ListScheduler::FindSchedule: Done with inst->Schedule\n");

      //debug
      printf("ListScheduler::FindSchedule: Calling rgn_->SchdulInst()\n");

      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);

      //debug
      printf("ListScheduler::FindSchedule: Done with rgn_->SchdulInst()\n");

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

//#ifdef IS_DEBUG_SCHED
  crntSched_->Print();
//#endif

  return RES_SUCCESS;
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

__host__ __device__
bool SequentialListScheduler::IsSequentialInstruction(
    const SchedInstruction *Inst) const {
  // Instr with number N-1 must already be scheduled.
  return crntSched_->GetSchedCycle(Inst->GetNum() - 1) != SCHD_UNSCHDULD;
}

__host__ __device__
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

__device__
void ListScheduler::DevUpdtRdyLst_(InstCount cycleNum, int slotNum) {
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
void ListScheduler::CallUpdtRdyLst_(){
  printf("Device before update: ");
  rdyLst_->DevPrint();
  DevUpdtRdyLst_(crntCycleNum_, crntSlotNum_);
  printf("Device after update: ");
  rdyLst_->DevPrint();
}

//call device version of UpdtRdyLst_
__global__
void DeviceUpdtRdyLst_(ListScheduler *dev_listSched) {
  dev_listSched->CallUpdtRdyLst_();
}

//prepare data and execute kernel call
void ListScheduler::Call_Kernel(){
  ListScheduler *dev_listSched = NULL;
  CopyPointersToDevice(&dev_listSched);
  
  //debug
  //printf("Starting UpdtRdyLst_ kernel\n");
  
  DeviceUpdtRdyLst_<<<1,1>>>(dev_listSched);
  cudaDeviceSynchronize();
  printf("Post Kernel Error: %s\n",cudaGetErrorString(cudaGetLastError()));
  
  //temp listSched for testing correctness
  //ListScheduler *host_listSched = new ListScheduler(dataDepGraph_, machMdl_,
  //                                                  schedUprBound_, prirts_);
  
  ListScheduler *host_listSched = dev_listSched;

  host_listSched->CopyPointersFromDevice(dev_listSched);
}

void ListScheduler::CopyPointersFromDevice(ListScheduler *dev_listSched){
  //printf("Starting copy of ListSched to host\n");
  //TODO: implemenent
  //host_listSched->rdyLst_->CopyPointersFromDevice();
  //cudaFree(dev_listSched->frstRdyLstPerCycle_);
  //cudaFree(dev_listSched->rdyLst_);
  //cudaFree(dev_listSched);
}

//Copies all data in ListScheduler and pointed at by it to device
//dev_ suffix signifies a pointer to device memory
void ListScheduler::CopyPointersToDevice(ListScheduler **dev_ListSched){
  //transfer this instance of ListScheduler to device
  //allocate device memory
  if (cudaSuccess != cudaMallocManaged((void**)dev_ListSched, sizeof(ListScheduler))){
    printf("Error allocating device memory for dev_ListSched!\n");
    printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
    return;
  }
  //copy current ListScheduler to device
  if (cudaSuccess != cudaMemcpy(*dev_ListSched, this, sizeof(ListScheduler), 
			        cudaMemcpyHostToDevice)){
    printf("Error copying ListScheduler to device!\n");
    printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
    return;
  }

  //transfer this->rdyLst_ to device
  //pointer to device instance of rdyLst_
  ReadyList *dev_rdyLst = NULL;
  //allocate device memory
  if (cudaSuccess != cudaMallocManaged((void**)&dev_rdyLst, sizeof(ReadyList))){
    printf("Error allocating device memory for dev_rdyLst!\n");
    return;
  }
  //copy rdyLst_ to device
  if (cudaSuccess != cudaMemcpy(dev_rdyLst, rdyLst_, sizeof(ReadyList), 
			        cudaMemcpyHostToDevice)){
    printf("Error copying rdyLst_ to device!\n");
    return;
  }
  //update dev_ListSched->rdyLst_ pointer to dev_rdyLst, 
  //to reference the readylist on the device
  if (cudaSuccess != cudaMemcpy(&((*dev_ListSched)->rdyLst_), &dev_rdyLst, 
			        sizeof(ReadyList *), cudaMemcpyHostToDevice)){
    printf("Error updating dev_ListSched->rdyLst_!\n");
    return;
  }

  //also copy all objects rdyLst_ points to.
  //pass dev_rdyLst to allow its pointers to be updated
  rdyLst_->CopyPointersToDevice(dev_rdyLst, dataDepGraph_);

  LinkedList<SchedInstruction> **dev_frstRdyLstPerCycle;

  //allocate device memory
  if (cudaSuccess != cudaMallocManaged((void**)&dev_frstRdyLstPerCycle, 
	             schedUprBound_ * sizeof(LinkedList<SchedInstruction> *))){
    printf("Error allocating device memory for dev_frstRdyLstPerCycle!\n");
    printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
    return;
  }
  //copy over array to device
  if (cudaSuccess != cudaMemcpy(dev_frstRdyLstPerCycle, frstRdyLstPerCycle_, 
                     schedUprBound_ * sizeof(LinkedList<SchedInstruction> *), 
		     cudaMemcpyHostToDevice)){
    printf("Error copying frstRdyLstPerCycle_ to device!\n");
    printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
    return;
  }
  //update pointer dev_ListSched->frstRdyLstPerCycle on device
  if (cudaSuccess != cudaMemcpy(&((*dev_ListSched)->frstRdyLstPerCycle_), 
			        &dev_frstRdyLstPerCycle,
                                sizeof(LinkedList<SchedInstruction> **),
                                cudaMemcpyHostToDevice)){
    printf("Error updating dev_ListSched->frstRdyLstPerCycle_ on device!\n");
    printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
    return;
  }
  //copy over all linked lists pointed to by rstRdyLstPerCycle
  LinkedList<SchedInstruction> *dev_linkedList = NULL;
  for (int i = 0; i < schedUprBound_; i++){
    //if pointer is not null copy over what it points to
    if (frstRdyLstPerCycle_[i] != NULL){
      //allocate device memory
      if (cudaSuccess != cudaMallocManaged((void**)&dev_linkedList, 
                                    sizeof(LinkedList<SchedInstruction>))){
        printf("Error allocating dev mem for dev_frstRdyLstPerCycle[%d]!\n",i);
        printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
        return;
      }
      //copy over array to device
      if (cudaSuccess != cudaMemcpy(dev_linkedList, frstRdyLstPerCycle_[i],
                                    sizeof(LinkedList<SchedInstruction>), 
				    cudaMemcpyHostToDevice)){
        printf("Error copying frstRdyLstPerCycle_[%d] to device!\n",i);
        printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
        return;
      }
      //update pointer on device
      if (cudaSuccess != cudaMemcpy(&dev_frstRdyLstPerCycle[i], &dev_linkedList,
                                    sizeof(LinkedList<SchedInstruction> *),
                                    cudaMemcpyHostToDevice)){
        printf("Error updating dev_frstRdyLstPerCycle_[%d] on device!\n",i);
        printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
        return;
      }
      //copy over all pointers for frstRdyLstPerCycle[i],
      //link them to current linkedList
      frstRdyLstPerCycle_[i]->CopyPointersToDevice(dev_linkedList);
    }
  }
}
