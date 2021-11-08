#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/simplified_aco_ds.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/dev_defines.h"
#include <algorithm>
#include <cstddef>
#include <utility>
//aco simplified ds impl

using namespace llvm::opt_sched;

//use the log message macro to make GPU porting easier
#define LOG_MESSAGE(...) Logger::Info(__VA_ARGS__)

// ----
// ACOReadyList
// ----

ACOReadyList::ACOReadyList() {
  InstrCount = 0;
  CurrentSize = 0;
  CurrentCapacity = PrimaryBufferCapacity = 0;
  Overflowed = false;

  // create new allocations for the data
  IntAllocation = nullptr;
  HeurAllocation = nullptr;
  ScoreAllocation = nullptr;

  //build shortcut pointers
  InstrBase = nullptr;
  ReadyOnBase = nullptr;
  HeurBase = nullptr;
  ScoreBase = nullptr;

}

ACOReadyList::ACOReadyList(InstCount RegionSize) {
  InstrCount = RegionSize;
  CurrentSize = 0;
  CurrentCapacity = PrimaryBufferCapacity = computePrimaryCapacity(InstrCount);
  Overflowed = false;

  // create new allocations for the data
  IntAllocation = new InstCount[2*CurrentCapacity];
  HeurAllocation = new HeurType[CurrentCapacity];
  ScoreAllocation = new pheromone_t[CurrentCapacity];

  //build shortcut pointers
  InstrBase = IntAllocation;
  ReadyOnBase = IntAllocation + CurrentCapacity;
  HeurBase = HeurAllocation;
  ScoreBase = ScoreAllocation;
}

ACOReadyList::ACOReadyList(const ACOReadyList &Other) {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // create new allocations for the data
  IntAllocation = new InstCount[2*CurrentCapacity];
  HeurAllocation = new HeurType[CurrentCapacity];
  ScoreAllocation = new pheromone_t[CurrentCapacity];

  //build shortcut pointers
  InstrBase = IntAllocation;
  ReadyOnBase = IntAllocation + CurrentCapacity;
  HeurBase = HeurAllocation;
  ScoreBase = ScoreAllocation;

  // copy the allocation's entries
  for (InstCount I = 0; I < CurrentSize; ++I) {
    InstrBase[I] = Other.InstrBase[I];
    ReadyOnBase[I] = Other.ReadyOnBase[I];
    HeurBase[I] = Other.HeurBase[I];
    ScoreBase[I] = Other.ScoreBase[I];
  }
}

ACOReadyList &ACOReadyList::operator=(const ACOReadyList &Other) {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // delete current allocations
  delete[] IntAllocation;
  delete[] HeurAllocation;
  delete[] ScoreAllocation;

  // create new allocations for the data
  IntAllocation = new InstCount[2*CurrentCapacity];
  HeurAllocation = new HeurType[CurrentCapacity];
  ScoreAllocation = new pheromone_t[CurrentCapacity];

  //build shortcut pointers
  InstrBase = IntAllocation;
  ReadyOnBase = IntAllocation + CurrentCapacity;
  HeurBase = HeurAllocation;
  ScoreBase = ScoreAllocation;

  // copy over the allocation's entries
  for (InstCount I = 0; I < CurrentSize; ++I) {
    InstrBase[I] = Other.InstrBase[I];
    ReadyOnBase[I] = Other.ReadyOnBase[I];
    HeurBase[I] = Other.HeurBase[I];
    ScoreBase[I] = Other.ScoreBase[I];
  }

  return *this;
}

ACOReadyList::ACOReadyList(ACOReadyList &&Other) noexcept {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // copy over the old ready lists allocations and set them to NULL
  // so that the data we took won't get deleted
  IntAllocation = Other.IntAllocation;
  HeurAllocation = Other.HeurAllocation;
  ScoreAllocation = Other.ScoreAllocation;
  Other.IntAllocation = nullptr;
  Other.HeurAllocation = nullptr;
  Other.ScoreAllocation = nullptr;

  InstrBase = Other.InstrBase;
  ReadyOnBase = Other.ReadyOnBase;
  HeurBase = Other.HeurBase;
  ScoreBase = Other.ScoreBase;
}

ACOReadyList &ACOReadyList::operator=(ACOReadyList &&Other) noexcept {
  InstrCount = Other.InstrCount;
  PrimaryBufferCapacity = Other.PrimaryBufferCapacity;
  Overflowed = Other.Overflowed;
  CurrentCapacity = Other.CurrentCapacity;
  CurrentSize = Other.CurrentSize;

  // swap the allocations to give Other our allocations to delete
  std::swap(IntAllocation, Other.IntAllocation);
  std::swap(HeurAllocation, Other.HeurAllocation);
  std::swap(ScoreAllocation, Other.ScoreAllocation);

  InstrBase = Other.InstrBase;
  ReadyOnBase = Other.ReadyOnBase;
  HeurBase = Other.HeurBase;
  ScoreBase = Other.ScoreBase;

  return *this;
}

ACOReadyList::~ACOReadyList() {
  delete[] IntAllocation;
  delete[] HeurAllocation;
  delete[] ScoreAllocation;
}


// This is just a heuristic for the ready list size.
// A better function should be chosen experimentally
InstCount ACOReadyList::computePrimaryCapacity(InstCount RegionSize) {
  return RegionSize;
}

__host__ __device__
void ACOReadyList::addInstructionToReadyList(const ACOReadyListEntry &Entry) {
  #ifdef __CUDA_ARCH__
    // Ready List should never be resized on device, assume it will never overflow capacity
    assert(dev_CurrentSize[GLOBALTID] < CurrentCapacity);

    //add the instruction to the ready list
    dev_InstrBase[dev_CurrentSize[GLOBALTID]*numThreads_ + GLOBALTID] = Entry.InstId;
    dev_ReadyOnBase[dev_CurrentSize[GLOBALTID]*numThreads_ + GLOBALTID] = Entry.ReadyOn;
    dev_HeurBase[dev_CurrentSize[GLOBALTID]*numThreads_ + GLOBALTID] = Entry.Heuristic;
    dev_ScoreBase[dev_CurrentSize[GLOBALTID]*numThreads_ + GLOBALTID] = Entry.Score;
    ++dev_CurrentSize[GLOBALTID];
  #else
    // check to see if we need to expand the allocation/get a new allocation
    if (CurrentSize == CurrentCapacity) {
      int OldCap = CurrentCapacity;
      bool PrevOverflowed = Overflowed;

      // get a new allocation to put the data in
      // The expansion formula is to make the new allocation 1.5 times the size of the old one
      // consider making this formula more aggressive
      int NewCap = OldCap + OldCap/2 + 1;
      InstCount *NewIntFallback = new InstCount[2*NewCap];
      HeurType *NewHeurFallback = new HeurType[NewCap];
      pheromone_t *NewScoreFallback = new pheromone_t[NewCap];

      // copy the data
      InstCount NewInstrOffset = 0, NewReadyOnOffset = NewCap, HeurOffset = 0, ScoreOffset = 0;
      for (int I = 0; I < CurrentSize; ++I) {
        NewIntFallback[NewInstrOffset + I] = InstrBase[I];
        NewIntFallback[NewReadyOnOffset + I] = ReadyOnBase[I];
        NewHeurFallback[HeurOffset + I] = HeurBase[I];
        NewScoreFallback[ScoreOffset + I] = ScoreBase[I];
      }

      //delete the old allocations
      delete[] IntAllocation;
      delete[] HeurAllocation;
      delete[] ScoreAllocation;

      //copy the new allocations
      IntAllocation = NewIntFallback;
      HeurAllocation = NewHeurFallback;
      ScoreAllocation = NewScoreFallback;

      // update/recompute pointers and other values
      InstrBase = IntAllocation + NewInstrOffset;
      ReadyOnBase = IntAllocation + NewReadyOnOffset;
      HeurBase = HeurAllocation + HeurOffset;
      ScoreBase = ScoreAllocation + ScoreOffset;
      Overflowed = true;
      CurrentCapacity = NewCap;

      //print out a notice/error message
      //Welp this may be a performance disaster if this is happening too much
      LOG_MESSAGE("Overflowed ReadyList capacity. Old Cap:%d, New Cap:%d, Primary Cap:%d, Prev Overflowed:%B", OldCap, NewCap, PrimaryBufferCapacity, PrevOverflowed);
    }

    //add the instruction to the ready list
    InstrBase[CurrentSize] = Entry.InstId;
    ReadyOnBase[CurrentSize] = Entry.ReadyOn;
    HeurBase[CurrentSize] = Entry.Heuristic;
    ScoreBase[CurrentSize] = Entry.Score;
    ++CurrentSize;
  #endif
}

// We copy the instruction at the end of the array to the instruction at the target index
// then we decrement the Ready List's CurrentSize
// This function has undefined behavior if CurrentSize == 0
__host__ __device__
ACOReadyListEntry ACOReadyList::removeInstructionAtIndex(InstCount Indx) {
  #ifdef __CUDA_ARCH__
    assert(dev_CurrentSize[GLOBALTID] <= 0 || Indx >= dev_CurrentSize[GLOBALTID] || Indx < 0);
    ACOReadyListEntry E{dev_InstrBase[Indx*numThreads_ + GLOBALTID], 
                        dev_ReadyOnBase[Indx*numThreads_ + GLOBALTID], 
                        dev_HeurBase[Indx*numThreads_ + GLOBALTID], 
                        dev_ScoreBase[Indx*numThreads_ + GLOBALTID]};
    InstCount EndIndx = --dev_CurrentSize[GLOBALTID];
    dev_InstrBase[Indx*numThreads_ + GLOBALTID] = dev_InstrBase[EndIndx*numThreads_ + GLOBALTID];
    dev_ReadyOnBase[Indx*numThreads_ + GLOBALTID] = dev_ReadyOnBase[EndIndx*numThreads_ + GLOBALTID];
    dev_HeurBase[Indx*numThreads_ + GLOBALTID] = dev_HeurBase[EndIndx*numThreads_ + GLOBALTID];
    dev_ScoreBase[Indx*numThreads_ + GLOBALTID] = dev_ScoreBase[EndIndx*numThreads_ + GLOBALTID];
    return E;
  #else
    assert(CurrentSize <= 0 || Indx >= CurrentSize || Indx < 0);
    ACOReadyListEntry E{InstrBase[Indx], ReadyOnBase[Indx], HeurBase[Indx], ScoreBase[Indx]};
    InstCount EndIndx = --CurrentSize;
    InstrBase[Indx] = InstrBase[EndIndx];
    ReadyOnBase[Indx] = ReadyOnBase[EndIndx];
    HeurBase[Indx] = HeurBase[EndIndx];
    ScoreBase[Indx] = ScoreBase[EndIndx];
    return E;
  #endif
}

void ACOReadyList::AllocDevArraysForParallelACO(int numThreads) {
  size_t memSize;
  numThreads_ = numThreads;

  // Alloc dev array for dev_IntAllocation
  memSize = sizeof(InstCount) * CurrentCapacity * numThreads_ * 2;
  gpuErrchk(cudaMalloc(&dev_IntAllocation, memSize));

  // Alloc dev array for dev_HeurAllocation
  memSize = sizeof(HeurType) * CurrentCapacity * numThreads_;
  gpuErrchk(cudaMalloc(&dev_HeurAllocation, memSize));

  // Alloc dev array for dev_ScoreAllocation
  memSize = sizeof(pheromone_t) * CurrentCapacity * numThreads_;
  gpuErrchk(cudaMalloc(&dev_ScoreAllocation, memSize));

  // Alloc dev array for dev_CurrentSize
  memSize = sizeof(InstCount) * numThreads_;
  gpuErrchk(cudaMalloc(&dev_CurrentSize, memSize));

  // Alloc dev array for dev_CurrentSize
  memSize = sizeof(pheromone_t) * numThreads_;
  gpuErrchk(cudaMalloc(&dev_ScoreSum, memSize));

  //build shortcut pointers
  dev_InstrBase = dev_IntAllocation;
  dev_ReadyOnBase = &dev_IntAllocation[CurrentCapacity*numThreads];
  dev_HeurBase = dev_HeurAllocation;
  dev_ScoreBase = dev_ScoreAllocation;
}

void ACOReadyList::FreeDevicePointers() {
  cudaFree(dev_IntAllocation);
  cudaFree(dev_HeurAllocation);
  cudaFree(dev_ScoreAllocation);
  cudaFree(dev_CurrentSize);
  cudaFree(dev_ScoreSum);
}
