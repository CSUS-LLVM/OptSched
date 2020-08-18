#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/utilities.h"
#include "opt-sched/Scheduler/cuda_lnkd_lst.cuh"

using namespace llvm::opt_sched;

__host__ __device__
ReadyList::ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts) {
  prirts_ = prirts;
  prirtyLst_ = NULL;
  int i;
  uint16_t totKeyBits = 0;

  // Initialize an array of KeyedEntry if a dynamic heuristic is used. This
  // enable fast updating for dynamic heuristics.
  if (prirts_.isDynmc)
    keyedEntries_ = new KeyedEntry<SchedInstruction, unsigned long>
        *[dataDepGraph->GetInstCnt()];
  else
    keyedEntries_ = nullptr;

  useCntBits_ = crtclPathBits_ = scsrCntBits_ = ltncySumBits_ = nodeID_Bits_ =
      inptSchedOrderBits_ = 0;

  // Calculate the number of bits needed to hold the maximum value of each
  // priority scheme
  for (i = 0; i < prirts.cnt; i++) {
    switch (prirts.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      //if creating readylist on device, use non virtual, device version of
      //GetRootInst(), Dev_GetRootInst()
#ifdef __CUDA_ARCH__
      maxCrtclPath_ = dataDepGraph->Dev_GetRootInst()->GetCrntLwrBound(DIR_BKWRD);
#else
      maxCrtclPath_ = dataDepGraph->GetRootInst()->GetCrntLwrBound(DIR_BKWRD);
#endif
      crtclPathBits_ = Utilities::clcltBitsNeededToHoldNum(maxCrtclPath_);
      totKeyBits += crtclPathBits_;
      break;

    case LSH_LUC:
      for (int j = 0; j < dataDepGraph->GetInstCnt(); j++) {
        keyedEntries_[j] = NULL;
      }
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

  prirtyLst_ = new PriorityList<SchedInstruction>;
  latestSubLst_ = new LinkedList<SchedInstruction>;

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

  //debug
  //printf("In ~ReadyList\n");

  Reset();

  //debug
  //printf("Done with Reset()\n");

  if (prirtyLst_)
    delete prirtyLst_;

  //debug
  //printf("Deleted prirtyLst_\n");

  if (latestSubLst_)
    delete latestSubLst_;

  //debug
  //printf("Deleted latestSubLst_\n");

  if (keyedEntries_)
    delete keyedEntries_;

  //debug
  //printf("Deleted keyedEntries_\n");

  //debug
  //printf("Done with ~ReadyList\n");
}

__host__ __device__
void ReadyList::Reset() {
 
  //debug
  //printf("Inside Reset()\n");
  prirtyLst_->Reset();

  //debug
  //printf("Done with prirtyLst_->Reset()\n");

  latestSubLst_->Reset();

  //debug
  //printf("Done with latesetSubLst_->Reset()\n");
}

__host__ __device__
void ReadyList::CopyList(ReadyList *othrLst) {
  assert(prirtyLst_->GetElmntCnt() == 0);
  assert(latestSubLst_->GetElmntCnt() == 0);
  assert(othrLst != NULL);

  // Copy the ready list and create the array of keyed entries. If a dynamic
  // heuristic is not used then the second parameter should be a nullptr and the
  // array will not be created.
  prirtyLst_->CopyList(othrLst->prirtyLst_, keyedEntries_);
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
void ReadyList::AddLatestSubLists(LinkedList<SchedInstruction> *lst1,
                                  LinkedList<SchedInstruction> *lst2) {
  assert(latestSubLst_->GetElmntCnt() == 0);
  if (lst1 != NULL)
    AddLatestSubList_(lst1);
  if (lst2 != NULL)
    AddLatestSubList_(lst2);
  prirtyLst_->ResetIterator();
}

void ReadyList::Print(std::ostream &out) {
  out << "Ready List: ";
  for (const auto *crntInst = prirtyLst_->GetFrstElmnt(); crntInst != NULL;
       crntInst = prirtyLst_->GetNxtElmnt()) {
    out << " " << crntInst->GetNum();
  }
  out << '\n';

  prirtyLst_->ResetIterator();
}

__device__
void ReadyList::DevPrint() {
  printf("Ready List: ");
  for (const auto *crntInst = prirtyLst_->GetFrstElmnt(); crntInst != NULL;
       crntInst = prirtyLst_->GetNxtElmnt()) {
    printf(" %d", crntInst->GetNum());
  }
  printf("\n");

  prirtyLst_->ResetIterator();
}

__host__ __device__
void ReadyList::AddLatestSubList_(LinkedList<SchedInstruction> *lst) {
  assert(lst != NULL);

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "Adding to the ready list: ";
#endif

  // Start iterating from the bottom of the list to access the most recent
  // instructions first.
  for (SchedInstruction *crntInst = lst->GetLastElmnt(); crntInst != NULL;
       crntInst = lst->GetPrevElmnt()) {
    // Once an instruction that is already in the ready list has been
    // encountered, this instruction and all the ones above it must be in the
    // ready list already.
    if (crntInst->IsInReadyList())
      break;
    AddInst(crntInst);
#ifdef IS_DEBUG_READY_LIST2
    Logger::GetLogStream() << crntInst->GetNum() << ", ";
#endif
    crntInst->PutInReadyList();
    latestSubLst_->InsrtElmnt(crntInst);
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

  for (SchedInstruction *inst = latestSubLst_->GetFrstElmnt(); inst != NULL;
       inst = latestSubLst_->GetNxtElmnt()) {
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
void ReadyList::ResetIterator() { prirtyLst_->ResetIterator(); }

__device__ __host__
void ReadyList::AddInst(SchedInstruction *inst) {
  bool changed;
  unsigned long key = CmputKey_(inst, false, changed);
  assert(changed == true);
  KeyedEntry<SchedInstruction, unsigned long> *entry =
      prirtyLst_->InsrtElmnt(inst, key, true);
  InstCount instNum = inst->GetNum();
  if (prirts_.isDynmc)
    keyedEntries_[instNum] = entry;
}

__device__ __host__
void ReadyList::AddList(LinkedList<SchedInstruction> *lst) {
  SchedInstruction *crntInst;

  if (lst != NULL)
    for (crntInst = lst->GetFrstElmnt(); crntInst != NULL;
         crntInst = lst->GetNxtElmnt()) {
      AddInst(crntInst);
    }

  prirtyLst_->ResetIterator();
}

__host__ __device__
InstCount ReadyList::GetInstCnt() const { return prirtyLst_->GetElmntCnt(); }

__host__ __device__
SchedInstruction *ReadyList::GetNextPriorityInst() {
  return prirtyLst_->GetNxtPriorityElmnt();
}

__host__ __device__
SchedInstruction *ReadyList::GetNextPriorityInst(unsigned long &key) {
  return prirtyLst_->GetNxtPriorityElmnt(key);
}

__host__ __device__
void ReadyList::UpdatePriorities() {
  assert(prirts_.isDynmc);

  SchedInstruction *inst;
  bool instChanged = false;
  for (inst = prirtyLst_->GetFrstElmnt(); inst != NULL;
       inst = prirtyLst_->GetNxtElmnt()) {
    unsigned long key = CmputKey_(inst, true, instChanged);
    if (instChanged) {
      prirtyLst_->BoostEntry(keyedEntries_[inst->GetNum()], key);
    }
  }
}

__host__ __device__
void ReadyList::RemoveNextPriorityInst() { prirtyLst_->RmvCrntElmnt(); }

__host__ __device__
bool ReadyList::FindInst(SchedInstruction *inst, int &hitCnt) {
  return prirtyLst_->FindElmnt(inst, hitCnt);
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


//Copies objects readylist points at to device
//dev_ suffix signifies a device pointer
//dataDepGraph needed for allocation of keyedEntries
void ReadyList::CopyPointersToDevice(ReadyList *dev_rdyLst, 
		                     DataDepGraph *dataDepGraph) {
  //copy prirtyLst_ to device
  //declare device pointer
  PriorityList<SchedInstruction> *dev_prirtyLst = NULL;

  //debug
/*
  if (prirts_.isDynmc) {
    printf("ReadyList priorities are Dynamic\n");
  } else {
    printf("ReadyList priorities are Static\n");
  }
*/

  //allocate device memory
  if (cudaSuccess != cudaMallocManaged((void**)&dev_prirtyLst, 
			        sizeof(PriorityList<SchedInstruction>))) {
    printf("Error allocating device memory for dev_prirtyLst: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
  }
  //copy prirtyLst_ to device
  if (cudaSuccess != cudaMemcpy(dev_prirtyLst, prirtyLst_, 
                                sizeof(PriorityList<SchedInstruction>), 
				cudaMemcpyHostToDevice)) {
    printf("Error copying prirtyLst_ to device: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
  }
  //update dev_rdyLst->prirtyLst_ pointer to dev_priorityList, 
  //to reference the prioritylist on the device
  if (cudaSuccess != cudaMemcpy(&(dev_rdyLst->prirtyLst_), &dev_prirtyLst, 
                                sizeof(PriorityList<SchedInstruction> *), 
				cudaMemcpyHostToDevice)) {
    printf("Error updating dev_rdyLst->prirtyLst_: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
  } 

  //Allocate/copy keyedEntries
  KeyedEntry<SchedInstruction, unsigned long> **dev_keyedEntries = NULL;
  
  if (keyedEntries_) {
    //allocate device memory
    if (cudaSuccess != cudaMallocManaged((void**)&dev_keyedEntries, 
			        dataDepGraph->GetInstCnt() * 
                sizeof(KeyedEntry<SchedInstruction, unsigned long> *))) {
      printf("Error allocating device memory for dev_keyedEntries: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
    }
    //copy array to device
    if (cudaSuccess != cudaMemcpy(dev_keyedEntries, keyedEntries_,
                                dataDepGraph->GetInstCnt() * 
                          sizeof(KeyedEntry<SchedInstruction, unsigned long> *),
                          cudaMemcpyHostToDevice)) {
      printf("Error copying keyedEntries_ to device: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
    }
    //update pointer on device
    if (cudaSuccess != cudaMemcpy(&(dev_rdyLst->keyedEntries_), &dev_keyedEntries,
                         sizeof(KeyedEntry<SchedInstruction, unsigned long> **),
                         cudaMemcpyHostToDevice)) {
      printf("Error updating keyedEntries_ on device: %s\n", 
  		    cudaGetErrorString(cudaGetLastError()));
    }
  }

  //pass dev_keyedEntries in order to update its pointers to device pointers
  prirtyLst_->CopyPointersToDevice(dev_prirtyLst, dev_keyedEntries);

  //copy latestSubLst_ to device
  //declare device pointer
  LinkedList<SchedInstruction> *dev_latestSubLst = NULL;
  //allocate device memory
  if (cudaSuccess != cudaMallocManaged((void**)&dev_latestSubLst, 
			        sizeof(LinkedList<SchedInstruction>))) {
    printf("Error allocating device memory for dev_latestSubLst: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
  }
  //copy latestSubLst_ to device
  if (cudaSuccess != cudaMemcpy(dev_latestSubLst, latestSubLst_,
                                sizeof(LinkedList<SchedInstruction>), 
				cudaMemcpyHostToDevice)) {
    printf("Error copying latestSubLst_ to device: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
  }
  //update dev_rdyLst->latestSubLst_ to dev_latestSubLst
  if (cudaSuccess != cudaMemcpy(&(dev_rdyLst->latestSubLst_), &dev_latestSubLst,
                                sizeof(LinkedList<SchedInstruction> *), 
				cudaMemcpyHostToDevice)) {
    printf("Error updating dev_rdyLst->latestSubLst_: %s\n", 
                    cudaGetErrorString(cudaGetLastError()));
  }

  latestSubLst_->CopyPointersToDevice(dev_latestSubLst);
}
