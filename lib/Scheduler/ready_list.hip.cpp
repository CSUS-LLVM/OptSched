#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/utilities.h"
#include "opt-sched/Scheduler/cuda_lnkd_lst.cuh"
#include "opt-sched/Scheduler/dev_defines.h"

using namespace llvm::opt_sched;

// pre-compute region info
__host__ __device__
void KeysHelper::initForRegion(DataDepGraph *DDG) {

  uint16_t CurrentOffset = 0, CurrentWidth = 0;

  uint64_t MaxKVs[MAX_SCHED_PRIRTS] = { 0 };

  // Calculate the number of bits needed to hold the maximum value of each
  // priority scheme
  for (int I = 0; I < priorities.cnt; ++I) {
    LISTSCHED_HEURISTIC Heur = priorities.vctr[I];
    uint64_t MaxV = 0;
    switch (Heur) {
    case LSH_CP:
    case LSH_CPR:
      MaxV = DDG->GetRootInst()->GetCrntLwrBound(DIR_BKWRD);
      break;

    case LSH_LUC:
    case LSH_UC:
      MaxV = DDG->GetMaxUseCnt();
      break;

    case LSH_NID:
    case LSH_LLVM:
      MaxV = DDG->GetInstCnt() - 1;
      break;

    case LSH_ISO:
      MaxV = DDG->GetMaxFileSchedOrder();
      break;

    case LSH_SC:
      MaxV = DDG->GetMaxScsrCnt();
      break;

    case LSH_LS:
      MaxV = DDG->GetMaxLtncySum();
      break;
    } // end switch

    // Track the size of the key and the width and location of our values
    CurrentWidth = Utilities::clcltBitsNeededToHoldNum(MaxV);
    Entries[Heur] = PriorityEntry{CurrentWidth, CurrentOffset};
    MaxKVs[Heur] = MaxV;
    CurrentOffset += CurrentWidth;
  }   // end for

  // check to see if the key can fit in our type
  assert(CurrentOffset <= 8 * sizeof(HeurType));

  // set the key size value to the final offset of the key
  KeysSz = CurrentOffset;

  //set maximumvalues needed to compute keys
  MaxNID = DDG->GetInstCnt() - 1;
  MaxISO = MaxKVs[LSH_ISO];

  // mark the object as initialized
  WasInitialized = true;

  // set the max value using the values compute key
  MaxValue = computeKey(MaxKVs);
}

// compute key
__host__ __device__
HeurType KeysHelper::computeKey(SchedInstruction *Inst, bool IncludeDynamic, RegisterFile *RegFiles) const {
  assert(WasInitialized);

  HeurType Key= 0;
  for (int I = 0; I < priorities.cnt; ++I) {
    LISTSCHED_HEURISTIC Heur = priorities.vctr[I];
    HeurType PriorityValue = 0;
    switch (Heur) {
    case LSH_CP:
    case LSH_CPR:
      PriorityValue = Inst->GetCrtclPath(DIR_BKWRD);
      break;

    case LSH_LUC:
      PriorityValue = IncludeDynamic ? Inst->CmputLastUseCnt(RegFiles) : 0;
      break;

    case LSH_UC:
      PriorityValue = Inst->GetUseCnt();
      break;

    case LSH_NID:
    case LSH_LLVM:
      PriorityValue = MaxNID - Inst->GetNodeID();
      break;

    case LSH_ISO:
      PriorityValue = MaxISO - Inst->GetFileSchedOrder();
      break;

    case LSH_SC:
      #ifdef __HIP_DEVICE_COMPILE__
      PriorityValue = Inst->GetScsrCnt_();
      #else
      PriorityValue = Inst->GetScsrCnt();
      #endif
      break;

    case LSH_LS:
      PriorityValue = Inst->GetLtncySum();
      break;
    }
    
    Key <<= Entries[Heur].Width;
    Key |= PriorityValue;
  }
  return Key;
}

__host__ __device__
HeurType KeysHelper::computeKey(const uint64_t *Values) const {
  assert(WasInitialized);

  HeurType Key = 0;

  for (int I = 0; I < priorities.cnt; ++I) {
    LISTSCHED_HEURISTIC Heur = priorities.vctr[I];
    Key <<= Entries[Heur].Width;
    Key |= Values[Heur];
  }

  return Key;
}

__host__
ReadyList::ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts) {
  dataDepGraph_ = dataDepGraph;
  prirts_ = prirts;
  prirtyLst_ = 
      new PriorityArrayList<InstCount>(dataDepGraph_->GetInstCnt());
  latestSubLst_ = new ArrayList<InstCount>(dataDepGraph_->GetInstCnt());

  // Initialize an array of KeyedEntry if a dynamic heuristic is used. This
  // enable fast updating for dynamic heuristics.
  if (prirts_.isDynmc) {
    keyedEntries_ = new KeyedEntry<SchedInstruction, unsigned long>
        *[dataDepGraph->GetInstCnt()];
  } else
    keyedEntries_ = nullptr;

  // Initialize the KeyHelper
  KHelper = KeysHelper(prirts);
  KHelper.initForRegion(dataDepGraph);

  // if we have an luc in the Priorities then lets store some info about it
  // to improve efficiency
  PriorityEntry LUCEntry = KHelper.getPriorityEntry(LSH_LUC);
  if (LUCEntry.Width) {
    useCntBits_ = LUCEntry.Width;
    LUCOffset = LUCEntry.Offset;
  }

#ifdef IS_DEBUG_READY_LIST2
  Logger::Info("The ready list key size is %d bits", KHelper->getKeySizeInBits());
#endif

  }

__host__
ReadyList::~ReadyList() {
  Reset();
  if (prirtyLst_)
    delete prirtyLst_;
  if (latestSubLst_)
    delete latestSubLst_;
  if (keyedEntries_)
    delete keyedEntries_;
}

__host__ __device__
void ReadyList::Reset() {
#ifdef __HIP_DEVICE_COMPILE__
  dev_prirtyLst_[GLOBALTID].Reset();
#else
  prirtyLst_->Reset();
  latestSubLst_->Reset();
#endif
}

__host__ __device__
void ReadyList::CopyList(ReadyList *othrLst) {
  assert(prirtyLst_->GetElmntCnt() == 0);
  assert(latestSubLst_->GetElmntCnt() == 0);
  assert(othrLst != NULL);

  // Copy the ready list and create the array of keyed entries. If a dynamic
  // heuristic is not used then the second parameter should be a nullptr and the
  // array will not be created.
  prirtyLst_->CopyList(othrLst->prirtyLst_);
}

__device__ __host__
unsigned long ReadyList::CmputKey_(SchedInstruction *inst, bool isUpdate,
                                   bool &changed) {

  int16_t OldLastUseCnt, NewLastUseCnt;

  // if we have an LUC Priority then we need to save the oldLUC
  OldLastUseCnt = inst->GetLastUseCnt();

  HeurType Key = KHelper.computeKey(inst, /*IncludeDynamic*/ true);

  //check if the luc value changed
  HeurType Mask = (0x01 << useCntBits_) - 1;
  HeurType LUCVal = (Key >> LUCOffset) & Mask;
  NewLastUseCnt = (int16_t) LUCVal;
  //set changed if the compute is not an update or the luc was changed
  changed = !isUpdate || OldLastUseCnt != NewLastUseCnt;

  return Key;
}

__host__ __device__
void ReadyList::AddLatestSubLists(ArrayList<InstCount> *lst1,
                                  ArrayList<InstCount> *lst2) {
  assert(latestSubLst_->GetElmntCnt() == 0);
  if (lst1 != NULL)
    AddLatestSubList_(lst1);
  if (lst2 != NULL)
    AddLatestSubList_(lst2);
  prirtyLst_->ResetIterator();
}

void ReadyList::Print(std::ostream &out) {
  out << "Ready List: ";
  for (auto crntInst = prirtyLst_->GetFrstElmnt(); crntInst != END;
       crntInst = prirtyLst_->GetNxtElmnt()) {
    out << " " << crntInst;
  }
  out << '\n';

  prirtyLst_->ResetIterator();
}

__device__
void ReadyList::Dev_Print() {
  printf("Ready List: ");
  for (auto crntInst = dev_prirtyLst_[GLOBALTID].GetFrstElmnt(); crntInst != END;
       crntInst = dev_prirtyLst_[GLOBALTID].GetNxtElmnt()) {
    printf(" %d", crntInst);
  }
  printf("\n");

  dev_prirtyLst_[GLOBALTID].ResetIterator();
}

__host__ __device__
void ReadyList::AddLatestSubList_(ArrayList<InstCount> *lst) {
  assert(lst != NULL);

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "Adding to the ready list: ";
#endif

  // Start iterating from the bottom of the list to access the most recent
  // instructions first.
  SchedInstruction *crntInst;
  for (InstCount crntInstNum = lst->GetLastElmnt(); crntInstNum != END;
       crntInstNum = lst->GetPrevElmnt()) {
    // Once an instruction that is already in the ready list has been
    // encountered, this instruction and all the ones above it must be in the
    // ready list already.
    crntInst = dataDepGraph_->GetInstByIndx(crntInstNum);
    if (crntInst->IsInReadyList())
      break;
    AddInst(crntInst);
#ifdef IS_DEBUG_READY_LIST2
    Logger::GetLogStream() << crntInst->GetNum() << ", ";
#endif
    crntInst->PutInReadyList();
    latestSubLst_->InsrtElmnt(crntInstNum);
  }

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "\n";
#endif
}

__host__ __device__
void ReadyList::RemoveLatestSubList() {
#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "Removing from the ready list: ";
#endif
  SchedInstruction *inst;
  for (InstCount instNum = latestSubLst_->GetFrstElmnt(); instNum != END;
       instNum = latestSubLst_->GetNxtElmnt()) {
    inst = dataDepGraph_->GetInstByIndx(instNum);
    assert(inst->IsInReadyList());
    inst->RemoveFromReadyList();
#ifdef IS_DEBUG_READY_LIST2
    Logger::GetLogStream() << inst->GetNum() << ", ";
#endif
  }

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "\n";
#endif
}

__host__ __device__
void ReadyList::ResetIterator() {
#ifdef __HIP_DEVICE_COMPILE__
  dev_prirtyLst_[GLOBALTID].ResetIterator();
#else
  prirtyLst_->ResetIterator(); 
#endif
}

__device__ __host__
void ReadyList::AddInst(SchedInstruction *inst) {
  bool changed;
  unsigned long key = CmputKey_(inst, false, changed);
  assert(changed == true);
#ifdef __HIP_DEVICE_COMPILE__
  dev_prirtyLst_[GLOBALTID].InsrtElmnt(inst->GetNum(), key, true);
#else
  prirtyLst_->InsrtElmnt(inst->GetNum(), key, true);
#endif
}

__device__ __host__
void ReadyList::AddList(ArrayList<InstCount> *lst) {
  SchedInstruction *crntInst;

  if (lst != NULL)
    for (InstCount crntInstNum = lst->GetFrstElmnt(); crntInstNum != END;
         crntInstNum = lst->GetNxtElmnt()) {
      crntInst = dataDepGraph_->GetInstByIndx(crntInstNum);
      AddInst(crntInst);
    }

  prirtyLst_->ResetIterator();
}

__host__ __device__
InstCount ReadyList::GetInstCnt() const {
#ifdef __HIP_DEVICE_COMPILE__
  return dev_prirtyLst_[GLOBALTID].GetElmntCnt();
#else
  return prirtyLst_->GetElmntCnt();
#endif
}

__host__ __device__
SchedInstruction *ReadyList::GetNextPriorityInst() {
#ifdef __HIP_DEVICE_COMPILE__
  return dataDepGraph_->GetInstByIndx(dev_prirtyLst_[GLOBALTID].GetNxtElmnt());
#else
  return dataDepGraph_->GetInstByIndx(prirtyLst_->GetNxtElmnt());
#endif
}

__host__ __device__
SchedInstruction *ReadyList::GetNextPriorityInst(unsigned long &key) {
#ifdef __HIP_DEVICE_COMPILE__
  int indx;
  SchedInstruction *inst = dataDepGraph_->
                    GetInstByIndx(dev_prirtyLst_[GLOBALTID].GetNxtElmnt(indx));
  key = dev_prirtyLst_[GLOBALTID].GetKey(indx);
  return inst;
#else
  int indx;
  SchedInstruction *inst = dataDepGraph_->
	            GetInstByIndx(prirtyLst_->GetNxtElmnt(indx));
  key = prirtyLst_->GetKey(indx);
  return inst;
#endif
}

__host__ __device__
void ReadyList::UpdatePriorities() {
  assert(prirts_.isDynmc);

  SchedInstruction *inst;
  bool instChanged = false;
  for (InstCount instNum = prirtyLst_->GetFrstElmnt(); instNum != END;
       instNum = prirtyLst_->GetNxtElmnt()) {
    inst = dataDepGraph_->GetInstByIndx(instNum);
    unsigned long key = CmputKey_(inst, true, instChanged);
    if (instChanged) {
      prirtyLst_->BoostElmnt(instNum, key);
    }
  }
}

__host__ __device__
void ReadyList::RemoveNextPriorityInst() {
#ifdef __HIP_DEVICE_COMPILE__
  dev_prirtyLst_[GLOBALTID].RmvCrntElmnt();
#else
  prirtyLst_->RmvCrntElmnt();
#endif
}

__host__ __device__
bool ReadyList::FindInst(SchedInstruction *inst, int &hitCnt) {
  return prirtyLst_->FindElmnt(inst->GetNum(), hitCnt);
}

__host__ __device__
unsigned long ReadyList::MaxPriority() { return KHelper.getMaxValue(); }

void ReadyList::AllocDevArraysForParallelACO(int numThreads) {
  size_t memSize;
  // Alloc dev array for dev_prirtyLst_
  memSize = sizeof(PriorityArrayList<InstCount>) * numThreads;
  gpuErrchk(hipMallocManaged(&dev_prirtyLst_, memSize));
}

void ReadyList::CopyPointersToDevice(ReadyList *dev_rdyLst, 
		                     DataDepGraph *dev_DDG, 
				     int numThreads) {
  size_t memSize;
  dev_rdyLst->dataDepGraph_ = dev_DDG;
  // Copy prirtyLst_
  prirtyLst_->ResetIterator();
  memSize = sizeof(PriorityArrayList<InstCount>);
  for (int i = 0; i < numThreads; i++) {
    gpuErrchk(hipMemcpy(&dev_rdyLst->dev_prirtyLst_[i], prirtyLst_, memSize,
	  	         hipMemcpyHostToDevice));
  }
  // Alloc elmnts for each prirtyLst_ in one hipMalloc call
  InstCount *temp_arr;
  memSize = sizeof(InstCount) * prirtyLst_->maxSize_ * numThreads;
  gpuErrchk(hipMalloc(&temp_arr, memSize));
  // Assign a chunk of the large array to each prirtyLst_
  for (int i = 0; i < numThreads; i++)
    dev_rdyLst->dev_prirtyLst_[i].elmnts_ = &temp_arr[i * prirtyLst_->maxSize_];
  // Alloc keys for each prirtyLst_ in one hipMalloc call
  unsigned long *temp_ptr;
  memSize = sizeof(unsigned long) * prirtyLst_->maxSize_ * numThreads;
  gpuErrchk(hipMalloc(&temp_ptr, memSize));
  // Assign a chunk of the large array to each prirtyLst_
  for (int i = 0; i < numThreads; i++)
    dev_rdyLst->dev_prirtyLst_[i].keys_ = &temp_ptr[i * prirtyLst_->maxSize_];
/*
  memSize = sizeof(PriorityArrayList<InstCount>);
  gpuErrchk(hipMallocManaged(&dev_rdyLst->prirtyLst_, memSize));
  gpuErrchk(hipMemcpy(dev_rdyLst->prirtyLst_, prirtyLst_, memSize,
                       hipMemcpyHostToDevice));
  if (prirtyLst_->elmnts_) {
    memSize = sizeof(InstCount) * prirtyLst_->maxSize_;
    gpuErrchk(hipMalloc(&dev_rdyLst->prirtyLst_->elmnts_, memSize));
    memSize = sizeof(unsigned long) * prirtyLst_->maxSize_;
    gpuErrchk(hipMalloc(&dev_rdyLst->prirtyLst_->keys_, memSize));
  }
*/
  memSize = sizeof(PriorityArrayList<InstCount>) * numThreads;
  gpuErrchk(hipMemPrefetchAsync(dev_prirtyLst_, memSize, 0)); 
}

void ReadyList::FreeDevicePointers(int numThreads) {
  hipFree(dev_prirtyLst_[0].keys_);
  hipFree(dev_prirtyLst_[0].elmnts_);
  hipFree(dev_prirtyLst_);
/*
  hipFree(prirtyLst_->elmnts_);
  hipFree(prirtyLst_->keys_);
  hipFree(prirtyLst_);
*/
}
