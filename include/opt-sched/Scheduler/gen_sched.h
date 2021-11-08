/*******************************************************************************
Description:  Defines two levels of abstract scheduler classes:
              * The InstScheduler class includes basic data structures and
                methods that are common to all scheduling techniques
                (constrianed or restricted)
              * The ConstrainedScheduler class is another abstract class that
                includes the basic data structures and methods that only
                constrained schedulers, as opposed to relaxed schedulers, need.
              The most fundamental and first implemented example of constrained
              schedulers is the list scheduler defined in list_sched.h.
Author:       Ghassan Shobaki
Created:      Apr. 2002
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_BASIC_GEN_SCHED_H
#define OPTSCHED_BASIC_GEN_SCHED_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/cuda_lnkd_lst.cuh"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include <cuda_runtime.h>

namespace llvm {
namespace opt_sched {

// A cycle window reserved by an unpipelined instruction.
struct ReserveSlot {
  InstCount strtCycle;
  InstCount endCycle;
};

enum SchedulerType {
  // List scheduler.
  SCHED_LIST,
  // Sequential list scheduler.
  SCHED_SEQ
};

// Forward declarations used to reduce the number of #includes.
class MachineModel;
class DataDepStruct;
class DataDepGraph;
class InstSchedule;
class ReadyList;
class SchedRegion;

// An abstract base class for all schedulers. Includes basic data structures and
// methods that are common to all scheduling techniques (constrianed or
// restricted).
class InstScheduler {
public:
  // Constructs a scheduler for the given machine and dependence graph, with
  // the specified upper bound.
  __host__ __device__
  InstScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                InstCount schedUprBound);
  // Deallocates memory used by the scheduler.
  __host__ __device__
  virtual ~InstScheduler();

  __host__ __device__
  InstCount GetTotInstCnt() { return totInstCnt_; }

protected:
  // A pointer to the machine which this scheduler uses
  MachineModel *machMdl_;

  // The issue rate of the underlying machine model.
  // TODO(ghassan): Eliminate.
  int issuRate_;
  // The number of issue types (pipelines) of the underlying machine model.
  int issuTypeCnt_;
  // The maximum number of total issue slots per cycle, for all issue types.
  int issuSlotCnt_;
  // How many slots of each issue type the machine has per cycle.
  int *slotsPerTypePerCycle_;
  // How many instructions of each issue type does the dependence graph
  // contain.
  InstCount *instCntPerIssuType_;

  // The total number of instructions to be scheduled.
  InstCount totInstCnt_;
  // The number of instructions that have been scheduled so far. When this is
  // equal to totInstCnt_ we have a complete schedule.
  InstCount schduldInstCnt_;
  // pointer to a device array used to store schduldInstCnt_ for 
  // each thread by parallel ACO
  InstCount *dev_schduldInstCnt_;
  // An absolute upper bound on the schedule length, used to determine data
  // structure sizes.
  InstCount schedUprBound_;

  // A pointer to the instruction at the root node of the dependence graph.
  SchedInstruction *rootInst_;
  // A pointer to the instruction at the leaf node of the dependence graph.
  SchedInstruction *leafInst_;

  // Whether the dependence graph includes unpipelined instructions.
  bool includesUnpipelined_;

  // Returns whether all instructions have been scheduled.
  __host__ __device__
  bool IsSchedComplete_();
};

// An abstract base class for constrained schedulers, regular schedulers that
// solve the NP-hard scheduling problem with latency constraints.
class ConstrainedScheduler : public InstScheduler {
public:
  // Constructs a constrained scheduler for the given machine and dependence
  // graph, with the specified upper bound.
  __host__ __device__
  ConstrainedScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                       InstCount schedUprBound, bool ACOEn=false);
  // Deallocates memory used by the scheduler.
  __host__ __device__
  virtual ~ConstrainedScheduler();

  // Calculates the schedule and returns it in the passed argument.
  //__host__ __device__
  //FUNC_RESULT FindSchedule(InstSchedule *sched, SchedRegion *rgn) = 0;

protected:
  // Whether this instance of ConstrainedScheduler is being used with ACO and
  // therefore can use the aco optimizations
  bool IsACO;
  // The data dependence graph to be scheduled.
  DataDepGraph *dataDepGraph_;
  // The current schedule.
  InstSchedule *crntSched_;
  // The ready list.
  ReadyList *rdyLst_;

  // The number of the current cycle to be used in cycle-by-cycle scheduling.
  InstCount crntCycleNum_;
  // pointer to a device array used to store crntCycleNum_ for 
  // each thread by parallel ACO
  InstCount *dev_crntCycleNum_;
  // The number of the next available slot within the current cycle.
  // Initially, when no instructions have been scheduled yet, this points to
  // slot #0 in cycle #0.
  InstCount crntSlotNum_;
  // pointer to a device array used to store crntSlotNum_ for
  // each thread by parallel ACO
  InstCount *dev_crntSlotNum_;
  // As above, but only for "real" instructions, as opposed to artificial.
  InstCount crntRealSlotNum_;
  // pointer to a device array used to store crntRealSlotNum_ for
  // each thread by parallel ACO
  InstCount *dev_crntRealSlotNum_;
  // The current number of consecutive empty cycles (filled only with stalls)
  // since an instruction was scheduled.
  InstCount consecEmptyCycles_;

  // Whether the current cycle is blocked by an instruction that blocks the
  // whole cycle.
  bool isCrntCycleBlkd_;
  // pointer to a device array used to store crntCycleNum_ for
  // each thread by parallel ACO
  bool *dev_isCrntCycleBlkd_;

  // An array of lists indexed by cycle number. For each cycle, there is a
  // list that holds all the instructions that first become ready in that
  // cycle. Whenever the scheduler gets to a new cycle, it inserts the
  // corresponding first-ready list of that cycle into the global sorted ready
  // list.
  ArrayList<InstCount> **frstRdyLstPerCycle_;

  // An array holding the number of issue slots available for each issue type
  // in the current machine cycle.
  int16_t *avlblSlotsInCrntCycle_;
  // pointer to a device array used to store avlblSlotsInCrntCycle_ for
  // each thread by parallel ACO
  int16_t **dev_avlblSlotsInCrntCycle_;

  // The reserved scheduling slots.
  ReserveSlot *rsrvSlots_;
  // pointer to a device array used to store rsrvSlots_ for
  // each thread by parallel ACO
  ReserveSlot **dev_rsrvSlots_;
  // The number of elements in rsrvSlots_.
  int16_t rsrvSlotCnt_;
  // pointer to a device array used to store rsrvSlotCnt_ for
  // each thread by parallel ACO
  int16_t *dev_rsrvSlotCnt_;

  // A pointer to the scheduling region. Needed to perform region-specific
  // calculations.
  SchedRegion *rgn_;
  // pointer to a device array used to store rgn_ for
  // each thread by parallel ACO
  SchedRegion *dev_rgn_;

  SchedPriorities prirts_;

  // Resets slot availability and cycle blocking states to prepare the
  // scheduler for a new cycle.
  __host__ __device__
  void InitNewCycle_();

  // Schedules an instruction in a given cycle and notifies its successors
  // to update their readiness status. This will cause each of these
  // successors to become partially ready or completely ready depending on
  // whether this instruction was the last unscheduled predecessor.
  __host__ __device__
  void SchdulInst_(SchedInstruction *inst, InstCount cycleNum);

  // Undoes the effects of scheduling an instruction by notifying its
  // successors to update their readiness status, and removing them from the
  // first-ready lists if necessary.
  __host__ __device__
  void UnSchdulInst_(SchedInstruction *inst);

  // Allocates memory for the reserve slots.
  __host__ __device__
  void AllocRsrvSlots_();
  // Initializes the previously allocated reserve slots.
  __host__ __device__
  void ResetRsrvSlots_();

  // Fills a new reserve slot with the appropriate cycle numbers (starts at
  // current slot and lasts for the length of the given instruction's latency
  // plus 1. No-op if given a pipelined instruction.
  __host__ __device__
  void DoRsrvSlots_(SchedInstruction *inst);
  // Empties a previously filled reserve slot. No-op if given a pipelined
  // instruction.
  __host__ __device__
  void UndoRsrvSlots_(SchedInstruction *inst);

  // Initialized the scheduler for a new iteration. Should be called whenever
  // a new iteration of the scheduler is started.
  __host__ __device__
  bool Initialize_(InstCount trgtSchedLngth = INVALID_VALUE,
                   LinkedList<SchedInstruction> *fxdLst = NULL);

  // Moves forward by one slot and updates the cycle and slot numbers. Returns
  // true if the cycle is advanced.
  __host__ __device__
  bool MovToNxtSlot_(SchedInstruction *inst);

  // Moves backward by one slot and updates the cycle and slot numbers.
  // Returns true if this causes a move back to the previous cycle.
  __host__ __device__
  bool MovToPrevSlot_(int prevRealSlotNum);

  // Cleans up the first-ready array of the current cycle (if any).
  __host__ __device__
  void CleanupCycle_(InstCount cycleNum);

  // Checks the legality of issuing an instruction of a given issue type.
  __host__ __device__
  bool ChkInstLglty_(SchedInstruction *inst) const;

  // Early check for instruction legality.
  __host__ __device__
  bool IsTriviallyLegal_(const SchedInstruction *inst) const;

  // Checks the legality of the current schedule.
  __host__ __device__
  bool ChkSchedLglty_(bool isEmptyCycle);

  // Updates the slot availability information to reflect the scheduling of
  // the given instruction.
  __host__ __device__
  void UpdtSlotAvlblty_(SchedInstruction *inst);

  // A pure virtual function for updating the ready list. Each concrete
  // scheduler should define its own version.
  //__host__ __device__
  //virtual void UpdtRdyLst_(InstCount cycleNum, int slotNum) = 0;
};

} // namespace opt_sched
} // namespace llvm

#endif
