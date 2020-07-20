/*******************************************************************************
Description:  Defines a list scheduler, based on the defintions of a generic
              scheduler and a constrained scheduler in gen_sched.h, which
              collectively include the meat of the implementation.
Author:       Ghassan Shobaki
Created:      Apr. 2002
Last Update:  Sept. 2013
*******************************************************************************/

#ifndef OPTSCHED_LIST_SCHED_LIST_SCHED_H
#define OPTSCHED_LIST_SCHED_LIST_SCHED_H

#include "opt-sched/Scheduler/gen_sched.h"
#include <cuda_runtime.h>

namespace llvm {
namespace opt_sched {

class ListScheduler : public ConstrainedScheduler {
public:
  // Creates a list scheduler for the given dependence graph, machine and
  // schedule upper bound, using the specified heuristic.
  __host__ __device__
  ListScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                InstCount schedUprBound, SchedPriorities prirts);
  __host__ __device__
  virtual ~ListScheduler();

  // Calculates the schedule and returns it in the passed argument.
  __host__ __device__
  FUNC_RESULT FindSchedule(InstSchedule *sched, SchedRegion *rgn);
  //Copies all objects pointed to by listsched to device
  void CopyPointersToDevice(ListScheduler **dev_ListSched);
  //Copies all objects pointer to by dev_listSched to Host
  void CopyPointersFromDevice(ListScheduler *dev_listSched);
  //Calls UpdtRdyLst_ when executing on device
  //needed since the method is private and kernel cannot be a member function
  __device__
  void CallUpdtRdyLst_();
protected:
  bool isDynmcPrirty_;
  // Adds the instructions that have just become ready at this cycle to the
  // ready list.
  __host__ __device__
  void UpdtRdyLst_(InstCount cycleNum, int slotNum);
  
  //device version of UpdtRdyLst_ (depreciated)
  __device__
  void DevUpdtRdyLst_(InstCount cycleNum, int slotNum);
  //Copy data and initiate Dev_UpdtRdyLst_ kernel
  void Call_Kernel();

  // Pick next instruction to be scheduled. Returns NULL if no instructions are
  // ready.
  __host__ __device__
  virtual SchedInstruction *PickInst() const;
};

// Force the list scheduler to maintain the source ordering of the instructions
// regardless of latency or machine model constraints.
class SequentialListScheduler : public ListScheduler {
public:
  __host__ __device__
  SequentialListScheduler(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                          InstCount schedUprBound, SchedPriorities prirts);

private:
  // Does this instruction come next in the source ordering after all currently
  // scheduled instructions, e.g. 0, 1, 2, 3, 4.
  __host__ __device__
  bool IsSequentialInstruction(const SchedInstruction *Inst) const;

  __host__ __device__
  bool ChkInstLglty_(SchedInstruction *inst) const override;
};

} // namespace opt_sched
} // namespace llvm

#endif
