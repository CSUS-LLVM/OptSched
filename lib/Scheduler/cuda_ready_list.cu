#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/utilities.h"
#include "opt-sched/Scheduler/cuda_lnkd_lst.cuh"
#include "opt-sched/Scheduler/dev_defines.h"

using namespace llvm::opt_sched;

__host__ __device__
ReadyList::ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts) {
  dataDepGraph_ = dataDepGraph;
  prirts_ = prirts;
  prirtyLst_ = NULL;
  int i;
  uint16_t totKeyBits = 0;

  // Initialize an array of KeyedEntry if a dynamic heuristic is used. This
  // enable fast updating for dynamic heuristics.
/*  if (prirts_.isDynmc)
    keyedEntries_ = new KeyedEntry<SchedInstruction, unsigned long>
        *[dataDepGraph->GetInstCnt()];
  else*/
    keyedEntries_ = nullptr;

  useCntBits_ = crtclPathBits_ = scsrCntBits_ = ltncySumBits_ = nodeID_Bits_ =
      inptSchedOrderBits_ = 0;

  // Calculate the number of bits needed to hold the maximum value of each
  // priority scheme
  for (i = 0; i < prirts.cnt; i++) {
    switch (prirts.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      maxCrtclPath_ = dataDepGraph->GetRootInst()->GetCrntLwrBound(DIR_BKWRD);
      crtclPathBits_ = Utilities::clcltBitsNeededToHoldNum(maxCrtclPath_);
      totKeyBits += crtclPathBits_;
      break;

    case LSH_LUC:
      /*for (int j = 0; j < dataDepGraph->GetInstCnt(); j++) {
        keyedEntries_[j] = NULL;
      }*/
      maxUseCnt_ = dataDepGraph->GetMaxUseCnt();
      useCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxUseCnt_);
      totKeyBits += useCntBits_;
      break;

    case LSH_UC:
      maxUseCnt_ = dataDepGraph->GetMaxUseCnt();
      useCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxUseCnt_);
      totKeyBits += useCntBits_;
      break;

    case LSH_NID:
    case LSH_LLVM:
      maxNodeID_ = dataDepGraph->GetInstCnt() - 1;
      nodeID_Bits_ = Utilities::clcltBitsNeededToHoldNum(maxNodeID_);
      totKeyBits += nodeID_Bits_;
      break;

    case LSH_ISO:
      maxInptSchedOrder_ = dataDepGraph->GetMaxFileSchedOrder();
      inptSchedOrderBits_ =
          Utilities::clcltBitsNeededToHoldNum(maxInptSchedOrder_);
      totKeyBits += inptSchedOrderBits_;
      break;

    case LSH_SC:
      maxScsrCnt_ = dataDepGraph->GetMaxScsrCnt();
      scsrCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxScsrCnt_);
      totKeyBits += scsrCntBits_;
      break;

    case LSH_LS:
      maxLtncySum_ = dataDepGraph->GetMaxLtncySum();
      ltncySumBits_ = Utilities::clcltBitsNeededToHoldNum(maxLtncySum_);
      totKeyBits += ltncySumBits_;
      break;
    } // end switch
  }   // end for

  assert(totKeyBits <= 8 * sizeof(unsigned long));

#ifdef IS_DEBUG_READY_LIST2
  Logger::Info("The ready list key size is %d bits", totKeyBits);
#endif

  prirtyLst_ = 
      new PriorityArrayList<InstCount>(dataDepGraph_->GetInstCnt());
  latestSubLst_ = new ArrayList<InstCount>(dataDepGraph_->GetInstCnt());

  int16_t keySize = 0;
  maxPriority_ = 0;
  for (i = 0; i < prirts_.cnt; i++) {
    switch (prirts_.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      AddPrirtyToKey_(maxPriority_, keySize, crtclPathBits_, maxCrtclPath_,
                      maxCrtclPath_);
      break;
    case LSH_LUC:
    case LSH_UC:
      AddPrirtyToKey_(maxPriority_, keySize, useCntBits_, maxUseCnt_,
                      maxUseCnt_);
      break;
    case LSH_NID:
    case LSH_LLVM:
      AddPrirtyToKey_(maxPriority_, keySize, nodeID_Bits_, maxNodeID_,
                      maxNodeID_);
      break;
    case LSH_ISO:
      AddPrirtyToKey_(maxPriority_, keySize, inptSchedOrderBits_,
                      maxInptSchedOrder_, maxInptSchedOrder_);
      break;
    case LSH_SC:
      AddPrirtyToKey_(maxPriority_, keySize, scsrCntBits_, maxScsrCnt_,
                      maxScsrCnt_);
      break;
    case LSH_LS:
      AddPrirtyToKey_(maxPriority_, keySize, ltncySumBits_, maxLtncySum_,
                      maxLtncySum_);
      break;
    }
  }
}

__host__ __device__
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
#ifdef __CUDA_ARCH__
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
  unsigned long key = 0;
  int16_t keySize = 0;
  int i;
  int16_t oldLastUseCnt, newLastUseCnt;
  changed = true;
  if (isUpdate)
    changed = false;

  for (i = 0; i < prirts_.cnt; i++) {
    switch (prirts_.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      AddPrirtyToKey_(key, keySize, crtclPathBits_,
                      inst->GetCrtclPath(DIR_BKWRD), maxCrtclPath_);
      break;

    case LSH_LUC:
      oldLastUseCnt = inst->GetLastUseCnt();
      newLastUseCnt = inst->CmputLastUseCnt();
      if (newLastUseCnt != oldLastUseCnt)
        changed = true;

      AddPrirtyToKey_(key, keySize, useCntBits_, newLastUseCnt, maxUseCnt_);
      break;

    case LSH_UC:
      AddPrirtyToKey_(key, keySize, useCntBits_, inst->GetUseCnt(), maxUseCnt_);
      break;

    case LSH_NID:
    case LSH_LLVM:
      AddPrirtyToKey_(key, keySize, nodeID_Bits_,
                      maxNodeID_ - inst->GetNodeID(), maxNodeID_);
      break;

    case LSH_ISO:
      AddPrirtyToKey_(key, keySize, inptSchedOrderBits_,
                      maxInptSchedOrder_ - inst->GetFileSchedOrder(),
                      maxInptSchedOrder_);
      break;

    case LSH_SC:
      AddPrirtyToKey_(key, keySize, scsrCntBits_, inst->GetScsrCnt(),
                      maxScsrCnt_);
      break;

    case LSH_LS:
      AddPrirtyToKey_(key, keySize, ltncySumBits_, inst->GetLtncySum(),
                      maxLtncySum_);
      break;
    }
  }
  return key;
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

__host__ __device__
void ReadyList::Dev_Print() {
  printf("Ready List: ");
  for (auto crntInst = prirtyLst_->GetFrstElmnt(); crntInst != END;
       crntInst = prirtyLst_->GetNxtElmnt()) {
    printf(" %d", crntInst);
  }
  printf("\n");

  prirtyLst_->ResetIterator();
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
  return dev_prirtyLst_[GLOBALTID].GetElmntCnt();
#else
  return prirtyLst_->GetElmntCnt();
#endif
}

__host__ __device__
SchedInstruction *ReadyList::GetNextPriorityInst() {
#ifdef __CUDA_ARCH__
  return dataDepGraph_->GetInstByIndx(dev_prirtyLst_[GLOBALTID].GetNxtElmnt());
#else
  return dataDepGraph_->GetInstByIndx(prirtyLst_->GetNxtElmnt());
#endif
}

__host__ __device__
SchedInstruction *ReadyList::GetNextPriorityInst(unsigned long &key) {
#ifdef __CUDA_ARCH__
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
#ifdef __CUDA_ARCH__
  dev_prirtyLst_[GLOBALTID].RmvCrntElmnt();
#else
  prirtyLst_->RmvCrntElmnt();
#endif
}

__host__ __device__
bool ReadyList::FindInst(SchedInstruction *inst, int &hitCnt) {
  return prirtyLst_->FindElmnt(inst->GetNum(), hitCnt);
}

__device__ __host__
void ReadyList::AddPrirtyToKey_(unsigned long &key, int16_t &keySize,
                                int16_t bitCnt, unsigned long val,
                                unsigned long maxVal) {
  assert(val <= maxVal);
  if (keySize > 0)
    key <<= bitCnt;
  key |= val;
  keySize += bitCnt;
}

__host__ __device__
unsigned long ReadyList::MaxPriority() { return maxPriority_; }

void ReadyList::AllocDevArraysForParallelACO(int numThreads) {
  size_t memSize;
  // Alloc dev array for dev_prirtyLst_
  memSize = sizeof(PriorityArrayList<InstCount>) * numThreads;
  gpuErrchk(cudaMallocManaged(&dev_prirtyLst_, memSize));
}

void ReadyList::CopyPointersToDevice(ReadyList *dev_rdyLst, 
		                     DataDepGraph *dev_DDG, 
				     int numThreads) {
  size_t memSize;
  dev_rdyLst->dataDepGraph_ = dev_DDG;
  // Copy prirtyLst_
  memSize = sizeof(PriorityArrayList<InstCount>);
  for (int i = 0; i < numThreads; i++) {
    gpuErrchk(cudaMemcpy(&dev_rdyLst->dev_prirtyLst_[i], prirtyLst_, memSize,
	  	         cudaMemcpyHostToDevice));
  }
/*
  // debug
  Logger::Info("Testing prirtyLst_ copy to device:");
  for (int i = 0; i < numThreads; i++)
    Logger::Info("size of dev_rdyLst->dev_prirtyLst_[%d] = %d", i, dev_rdyLst->dev_prirtyLst_[i].size_);
*/
  // Alloc elmnts for each prirtyLst_ in one cudaMalloc call
  InstCount *temp_arr;
  memSize = sizeof(InstCount) * prirtyLst_->maxSize_ * numThreads;
  gpuErrchk(cudaMalloc(&temp_arr, memSize));
  // Assign a chunk of the large array to each prirtyLst_
  for (int i = 0; i < numThreads; i++)
    dev_rdyLst->dev_prirtyLst_[i].elmnts_ = &temp_arr[i * prirtyLst_->maxSize_];
  // Alloc keys for each prirtyLst_ in one cudaMalloc call
  unsigned long *temp_ptr;
  memSize = sizeof(unsigned long) * prirtyLst_->maxSize_ * numThreads;
  gpuErrchk(cudaMalloc(&temp_ptr, memSize));
  // Assign a chunk of the large array to each prirtyLst_
  for (int i = 0; i < numThreads; i++)
    dev_rdyLst->dev_prirtyLst_[i].keys_ = &temp_ptr[i * prirtyLst_->maxSize_];
/*
  memSize = sizeof(PriorityArrayList<InstCount>);
  gpuErrchk(cudaMallocManaged(&dev_rdyLst->prirtyLst_, memSize));
  gpuErrchk(cudaMemcpy(dev_rdyLst->prirtyLst_, prirtyLst_, memSize,
                       cudaMemcpyHostToDevice));
  if (prirtyLst_->elmnts_) {
    memSize = sizeof(InstCount) * prirtyLst_->maxSize_;
    gpuErrchk(cudaMalloc(&dev_rdyLst->prirtyLst_->elmnts_, memSize));
    memSize = sizeof(unsigned long) * prirtyLst_->maxSize_;
    gpuErrchk(cudaMalloc(&dev_rdyLst->prirtyLst_->keys_, memSize));
  }
*/
}

void ReadyList::FreeDevicePointers(int numThreads) {
  cudaFree(dev_prirtyLst_[0].keys_);
  cudaFree(dev_prirtyLst_[0].elmnts_);
  cudaFree(dev_prirtyLst_);
/*
  cudaFree(prirtyLst_->elmnts_);
  cudaFree(prirtyLst_->keys_);
  cudaFree(prirtyLst_);
*/
}
