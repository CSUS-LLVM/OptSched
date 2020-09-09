/*******************************************************************************
Description:  Defines a ReadyList class, which is one of the main data
              structures that an instruction scheduler needs. The ready list is
              a sorted list of instructions whose data dependencies have been
              satisfied (their predecessors in the data dependence graph have
              been scheduled).
Author:       Ghassan Shobaki
Created:      Apr. 2002
Last Update:  Sept. 2013
*******************************************************************************/

#ifndef OPTSCHED_BASIC_READY_LIST_H
#define OPTSCHED_BASIC_READY_LIST_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdio>

namespace llvm {
namespace opt_sched {

// A priority list of instruction that are ready to schedule at a given point
// during the scheduling process.
class ReadyList {
public:
  // Constructs a ready list for the specified dependence graph with the
  // specified priorities.
  ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts);
  // Destroys the ready list and deallocates the memory used by it.
  ~ReadyList();

  // Resets the list and removes all elements from it.
  void Reset();

  // Adds an instruction to the ready list.
  void AddInst(SchedInstruction *inst);

  // Adds a list of instructions to the ready list.
  void AddList(LinkedList<SchedInstruction> *lst);

  // An iterator that allows accessing the instructions at the current time
  // in priority order. The first call will return the top priority
  // instruction, the next will return the instruction with the second rank,
  // and so on.
  SchedInstruction *GetNextPriorityInst();
  SchedInstruction *GetNextPriorityInst(unsigned long &key);

  // Removes the instruction returned by the last call to
  // GetNextPriorityInst().
  void RemoveNextPriorityInst();

  // Returns the number of instructions currently in the list.
  InstCount GetInstCnt() const;

  // Resets the list iterator to point back to the first instruction.
  void ResetIterator();

  // Adds instructions at the bottoms of the given two lists which have
  // not been added to the ready list already, and advance the internal time.
  // TODO(max): Elaborate.
  void AddLatestSubLists(LinkedList<SchedInstruction> *lst1,
                         LinkedList<SchedInstruction> *lst2);

  // Removes the most recently added sublist of instructions.
  // TODO(max): Elaborate.
  void RemoveLatestSubList();

  // Copies this list to another. Both lists must be empty.
  void CopyList(ReadyList *otherLst);

  // Searches the list for an instruction, returning whether it has been found
  // or not and writing the number of times it was found into hitCnt.
  bool FindInst(SchedInstruction *inst, int &hitCnt);

  // Update instruction priorities within the list
  // Called only if the priorities change dynamically during scheduling
  void UpdatePriorities();

  unsigned long MaxPriority();

  // Prints out the ready list, nicely formatted, into an output stream.
  void Print(std::ostream &out);

  // Constructs the priority-list key based on the schemes listed in prirts_.
  unsigned long CmputKey_(SchedInstruction *inst, bool isUpdate, bool &changed);

private:
  // An ordered vector of priorities
  SchedPriorities prirts_;

  // The priority list containing the actual instructions.
  PriorityList<SchedInstruction> prirtyLst_;

  // TODO(max): Document.
  LinkedList<SchedInstruction> latestSubLst_;

  // Array of pointers to KeyedEntry objects
  llvm::SmallVector<KeyedEntry<SchedInstruction, unsigned long> *, 0>
      keyedEntries_;

  // Is there a priority scheme that needs to be changed dynamically
  //    bool isDynmcPrirty_;

  // The maximum values for each part of the priority key.
  InstCount maxUseCnt_;
  InstCount maxCrtclPath_;
  InstCount maxScsrCnt_;
  InstCount maxLtncySum_;
  InstCount maxNodeID_;
  InstCount maxInptSchedOrder_;

  unsigned long maxPriority_;

  // The number of bits for each part of the priority key.
  int16_t useCntBits_;
  int16_t crtclPathBits_;
  int16_t scsrCntBits_;
  int16_t ltncySumBits_;
  int16_t nodeID_Bits_;
  int16_t inptSchedOrderBits_;

  // Adds instructions at the bottom of a given list which have not been added
  // to the ready list already.
  void AddLatestSubList_(LinkedList<SchedInstruction> *lst);

  // Calculates a new priority key given an existing key of size keySize by
  // appending bitCnt bits holding the value val, assuming val < maxVal.
  static void AddPrirtyToKey_(unsigned long &key, int16_t &keySize,
                              int16_t bitCnt, unsigned long val,
                              unsigned long maxVal);
};

} // namespace opt_sched
} // namespace llvm

#endif
