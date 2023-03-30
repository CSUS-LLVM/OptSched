/*******************************************************************************
Description:  Defines a ReadyList class, which is one of the main data
              structures that an instruction scheduler needs. The ready list is
              a sored list of instructions whose data dependences have been
              satisfied (their predecessors in the data dependence graph have
              been scheduled).
Author:       Ghassan Shobaki
Created:      Apr. 2002
Last Update:  Sept. 2013
*******************************************************************************/

#ifndef OPTSCHED_BASIC_READY_LIST_H
#define OPTSCHED_BASIC_READY_LIST_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/cuda_lnkd_lst.cuh"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include <cstdio>
#include <hip/hip_runtime.h>

namespace llvm {
namespace opt_sched {

struct PriorityEntry {
  uint16_t Width;
  uint16_t Offset;
};

class KeysHelper1 {
  public:
  __host__ __device__
  KeysHelper1(SchedPriorities prirts) : priorities(prirts), Entries{} {};
  __host__ __device__
  KeysHelper1() : KeysHelper1(SchedPriorities{}) {};

  // pre-compute region info
  __host__ __device__
  void initForRegion(DataDepGraph *DDG);

  // compute key
  __host__ __device__
  HeurType computeKey(SchedInstruction *Inst, bool IncludeDynamic, RegisterFile *RegFiles = NULL, DataDepGraph *ddg = NULL) const;
  __host__ __device__
  HeurType computeKey(const uint64_t *Values) const;

  // get information about a keys layout
  __host__ __device__
  PriorityEntry getPriorityEntry(int16_t Indx) const { return Entries[Indx]; }

  //get the max key size and value
  __host__ __device__
  HeurType getKeySizeInBits() const { return KeysSz; }
  __host__ __device__
  HeurType getMaxValue() const { return MaxValue; }

  // Allocates arrays to hold independent values for each device thread during
  // parallel ACO
  // void AllocDevArraysForParallelACO(int numThreads);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  // void FreeDevicePointers(int numThreads);

  private:
  // private member variables
  // scheduling priorities used for this KeysHelper
  SchedPriorities priorities;

  // width and offset info for each priority
  PriorityEntry Entries[MAX_SCHED_PRIRTS];

  // pre-computed size of all keys for this region
  uint16_t KeysSz = 0;

  // pre-computed max key value;
  HeurType MaxValue = 0;
  HeurType MaxNID = 0;
  HeurType MaxISO = 0;

  // Field to store if this KeyHelper was initialized
  bool WasInitialized = false;
};

class KeysHelper2 {
  public:
  __host__ __device__
  KeysHelper2(SchedPriorities prirts1, SchedPriorities prirts2) : priorities1(prirts1), priorities2(prirts2), Entries1{}, Entries2{} {};
  __host__ __device__
  KeysHelper2() : KeysHelper2(SchedPriorities{}, SchedPriorities{}) {};

  // pre-compute region info
  __host__ __device__
  void initForRegion(DataDepGraph *DDG);

  // compute key
  __host__ __device__
  HeurType computeKey(SchedInstruction *Inst, bool IncludeDynamic, RegisterFile *RegFiles = NULL, DataDepGraph *ddg = NULL) const;
  __host__ __device__
  HeurType computeKey(const uint64_t *Values, int whichPrirts = 1) const;

  // get information about a keys layout
  __host__ __device__
  PriorityEntry getPriorityEntry(int16_t Indx, int whichPrirts = 1) const {
    if (whichPrirts == 1)
      return Entries1[Indx];
    else
      return Entries2[Indx];
  }

  //get the max key size and value
  __host__ __device__
  HeurType getKeySizeInBits(int whichPrirts = 1) const {
    if (whichPrirts == 1)
      return KeysSz1;
    else
      return KeysSz2;
    }
  __host__ __device__
  HeurType getMaxValue(int whichPrirts = 1) const {
    if (whichPrirts == 1)
      return MaxValue1;
    else
      return MaxValue2;
    }

  // Allocates arrays to hold independent values for each device thread during
  // parallel ACO
  // void AllocDevArraysForParallelACO(int numThreads);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  // void FreeDevicePointers(int numThreads);

  private:
  // private member variables
  // scheduling priorities used for this KeysHelper
  SchedPriorities priorities1;
  SchedPriorities priorities2;

  // width and offset info for each priority
  PriorityEntry Entries1[MAX_SCHED_PRIRTS];
  PriorityEntry Entries2[MAX_SCHED_PRIRTS];

  // pre-computed size of all keys for this region
  uint16_t KeysSz1 = 0;
  uint16_t KeysSz2 = 0;

  // pre-computed max key value;
  HeurType MaxValue1 = 0;
  HeurType MaxValue2 = 0;
  HeurType MaxNID1 = 0;
  HeurType MaxNID2 = 0;
  HeurType MaxISO1 = 0;
  HeurType MaxISO2 = 0;

  // Field to store if this KeyHelper was initialized
  bool WasInitialized = false;
};

// A priority list of instruction that are ready to schedule at a given point
// during the scheduling process.
class ReadyList {
public:
  // Constructs a ready list for the specified dependence graph with the
  // specified priorities.
  __host__
  ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts);
  // Destroys the ready list and deallocates the memory used by it.
  __host__ __device__
  ~ReadyList();

  // Resets the list and removes all elements from it.
  __host__ __device__
  void Reset();

  // Adds an instruction to the ready list.
  __host__ __device__
  void AddInst(SchedInstruction *inst);

  // Adds a list of instructions to the ready list.
  __host__ __device__
  void AddList(ArrayList<InstCount> *lst);

  // An iterator that allows accessing the instructions at the current time
  // in priority order. The first call will return the top priority
  // instruction, the next will return the instruction with the second rank,
  // and so on.
  __host__ __device__
  SchedInstruction *GetNextPriorityInst();
  __host__ __device__
  SchedInstruction *GetNextPriorityInst(unsigned long &key);

  // Removes the instruction returned by the last call to
  // GetNextPriorityInst().
  __host__ __device__
  void RemoveNextPriorityInst();

  // Returns the number of instructions currently in the list.
  __host__ __device__
  InstCount GetInstCnt() const;

  // Resets the list iterator to point back to the first instruction.
  __host__ __device__
  void ResetIterator();

  // Adds instructions at the bottoms of the given two lists which have
  // not been added to the ready list already, and advance the internal time.
  // TODO(max): Elaborate.
  __host__ __device__
  void AddLatestSubLists(ArrayList<InstCount> *lst1,
                         ArrayList<InstCount> *lst2);

  // Removes the most recently added sublist of instructions.
  // TODO(max): Elaborate.
  __host__ __device__
  void RemoveLatestSubList();

  // Copies this list to another. Both lists must be empty.
  __host__ __device__
  void CopyList(ReadyList *otherLst);

  // Searches the list for an instruction, returning whether it has been found
  // or not and writing the number of times it was found into hitCnt.
  __host__ __device__
  bool FindInst(SchedInstruction *inst, int &hitCnt);

  // Update instruction priorities within the list
  // Called only if the priorities change dynamically during scheduling
  __host__ __device__
  void UpdatePriorities();

  __host__ __device__
  unsigned long MaxPriority();

  // Prints out the ready list, nicely formatted, into an output stream.
  __host__
  void Print(std::ostream &out);
  // Cannot use cout on device, invoke this on device instead
  __device__
  void Dev_Print();
  // Copy pointers to device and link them to passed device pointer
  void CopyPointersToDevice(ReadyList *dev_rdyLst, DataDepGraph *dev_DDG,
		            int numThreads);
  // Allocates arrays to hold independent values for each device thread during
  // parallel ACO
  void AllocDevArraysForParallelACO(int numThreads);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  void FreeDevicePointers(int numThreads);
  // Constructs the priority-list key based on the schemes listed in prirts_.
  __host__ __device__
  unsigned long CmputKey_(SchedInstruction *inst, bool isUpdate, bool &changed);

private:
  // A pointer to the DDG for the region
  DataDepGraph *dataDepGraph_;
  // An ordered vector of priorities
  SchedPriorities prirts_;

  // The KeysHelper for the key computations
  KeysHelper1 KHelper;

  // The priority list containing the actual instructions.
  PriorityArrayList<InstCount> *prirtyLst_;
  // An array of PArrayLists of size numThreads_ to allow each thread
  // to have an independent PAL
  PriorityArrayList<InstCount> *dev_prirtyLst_;

  // TODO(max): Document.
  ArrayList<InstCount> *latestSubLst_;

  // Array of pointers to KeyedEntry objects
  KeyedEntry<SchedInstruction, unsigned long> **keyedEntries_;

  // The number of bits for each part of the priority key.
  int16_t useCntBits_;
  int16_t LUCOffset;

  // Adds instructions at the bottom of a given list which have not been added
  // to the ready list already.
  __host__ __device__
  void AddLatestSubList_(ArrayList<InstCount> *lst);
};

} // namespace opt_sched
} // namespace llvm

#endif
