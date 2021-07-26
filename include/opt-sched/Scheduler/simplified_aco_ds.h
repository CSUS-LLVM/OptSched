/*******************************************************************************
Description:  Implements smaller more performant data structures for ACO
Author:       Paul McHugh
Created:      Jun. 2021
*******************************************************************************/

#ifndef OPTSCHED_SIMPLIFIED_ACO_H
#define OPTSCHED_SIMPLIFIED_ACO_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include <cstdint>

namespace llvm {
namespace opt_sched {

//forward declarations to reduce the number of classes that need to be included
class DataDepGraph;

//class for tracking the schedule cycle state
class ACOCycleState {

public:
  ACOCycleState(InstCount IssueRate) : MIssueRate(IssueRate), cycle(0), slot(0) {}

  //stores the issue rate of the CPU (here for convienience)
  const InstCount MIssueRate;

  //schedule cycle and slot
  InstCount cycle;
  InstCount slot;

};

struct ACOReadyListEntry {
  InstCount InstId, ReadyOn;
  HeurType Heuristic;
  pheromone_t Score;
};

//this aco specific readylist stores each ready instruction, its dynamic heuristic score, and the cycle it becomes ready
//It uses a (generous) heuristic to decide how much space to allocate.  If that space is exceeded then it gracefully handles it
//by making a larger allocation and copying the data to it.  THIS WILL KILL PERFORMANCE(ESPECIALLY ON THE GPU).  That is why it
//will also make a report that its heuristic max size was overrun.  Strongly consider fixing such warnings
class ACOReadyList {

protected:

  //used for the sizing heuristic
  InstCount InstrCount;
  InstCount PrimaryBufferCapacity;

  bool Overflowed;
  InstCount CurrentCapacity;
  InstCount CurrentSize;

  //allocation pointers
  InstCount *IntAllocation;
  HeurType *HeurAllocation;
  pheromone_t *ScoreAllocation;

  //pointers to areas in the InstCount allocation that store ready list entry attributes
  InstCount *InstrBase;
  InstCount *ReadyOnBase;
  HeurType *HeurBase;
  pheromone_t *ScoreBase;

  //function to decide how large the primary buffer's capacity should be
  InstCount computePrimaryCapacity(InstCount RegionSize);

public:

  ACOReadyList();
  explicit ACOReadyList(InstCount RegionSize);
  ACOReadyList(const ACOReadyList &Other);
  ACOReadyList &operator=(const ACOReadyList &Other);
  ACOReadyList(ACOReadyList &&Other) noexcept;
  ACOReadyList &operator=(ACOReadyList &&Other) noexcept;
  ~ACOReadyList();

  //used to store the total score of all instructions in the ready list
  pheromone_t ScoreSum;

  //get the total size of both the primary and fallback allocations
  size_t getTotalSizeInBytes() const;

  //gets the number of insturctions in the ready list
  InstCount getReadyListSize() const { return CurrentSize; }

  //IMPORTANT NOTE: ADDING OR REMOVING INSTRUCTIONS CAN/WILL CAUSE THE INSTRUCTIONS IN THE READY LIST TO BE MOVED TO NEW INDICES
  //DO NOT RELY ON AN INSTRUCTION'S INDEX IN THE READY LIST STAYING THE SAME FOLLOWING A REOMVAL/INSERTION
  //get instruction into at an index
  InstCount *getInstIdAtIndex(InstCount Indx) const;
  InstCount *getInstReadyOnAtIndex(InstCount Indx) const;
  HeurType *getInstHeuristicAtIndex(InstCount Indx) const;
  pheromone_t *getInstScoreAtIndex(InstCount Indx) const;

  //add a new instruction to the ready list
  void addInstructionToReadyList(const ACOReadyListEntry &Entry);
  ACOReadyListEntry removeInstructionAtIndex(InstCount Indx);
  void clearReadyList();

};

// ----
// ACOReadyList
// ----
inline size_t ACOReadyList::getTotalSizeInBytes() const {
  return (2 * sizeof(*IntAllocation) + sizeof(*HeurAllocation) + sizeof(*ScoreAllocation)) * CurrentCapacity;
}

inline InstCount *ACOReadyList::getInstIdAtIndex(InstCount Indx) const {
  return InstrBase + Indx;
}

inline InstCount *ACOReadyList::getInstReadyOnAtIndex(InstCount Indx) const {
  return ReadyOnBase + Indx;
}

inline HeurType *ACOReadyList::getInstHeuristicAtIndex(InstCount Indx) const {
  return HeurBase + Indx;
}

inline pheromone_t *ACOReadyList::getInstScoreAtIndex(InstCount Indx) const {
  return ScoreBase + Indx;
}

inline void ACOReadyList::clearReadyList() {
  CurrentSize = 0;
  ScoreSum = 0;
}

} // namespace opt_sched
} // namespace llvm

#endif
