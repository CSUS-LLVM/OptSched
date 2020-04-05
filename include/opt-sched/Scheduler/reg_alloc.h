/*******************************************************************************
Description:  Defines register allocation classes. By looking at the effect
              of scheduling decisions on the number of spills added during
simulated register allocation, we can evaluate the performance of the scheduler.

Author:       Austin Kerbow
Created:      Oct. 2017
Last Update:  Oct. 2017
*******************************************************************************/
#ifndef OPTSCHED_BASIC_REG_ALLOC_H
#define OPTSCHED_BASIC_REG_ALLOC_H

#include "opt-sched/Scheduler/data_dep.h"
#include <map>
#include <queue>
#include <stack>
#include <vector>

namespace llvm {
namespace opt_sched {

using namespace std;

/**
 * Class for performing basic top-down register allocation.
 */
class LocalRegAlloc {
public:
  typedef struct RegMap {
    // A queue of instruction numbers that this virtual register is used in.
    queue<int> nextUses;
    // Do we need to spill this virtual register.
    bool isDirty;
    // The physical register that this virtual register is mapped to. If this
    // virtual register is not mapped to a physical register, set to -1.
    int assignedReg;
  } RegMap;

  LocalRegAlloc(InstSchedule *instSchedule, DataDepGraph *dataDepGraph);
  virtual ~LocalRegAlloc();
  // Try to allocate registers in the region and count the number of spills
  // added.
  virtual void AllocRegs();
  // Initialize data for register allocation.
  virtual void SetupForRegAlloc();
  // Print information about the amount of spilling in the region after register
  // allocation.
  virtual void PrintSpillInfo(const char *dagName);
  // Return the spill cost of region after register allocation.
  virtual int GetCost() const;
  // Return the number of loads
  int GetNumLoads() const { return numLoads_; }
  // Return the number of stores
  int GetNumStores() const { return numStores_; }

private:
  InstSchedule *instSchedule_;
  DataDepGraph *dataDepGraph_;
  int numLoads_;
  int numStores_;
  int numRegTypes_;
  // For each register type, there is a stack that tracks free physical
  // registers.
  vector<stack<int>> freeRegs_;
  // For each virtual register, track the next use and the currently assigned
  // physical register.
  vector<map<int, RegMap>> regMaps_;
  // For each register type, we have a list of physical registers and the
  // current virtual register that is loaded. If the regsiter is free, set to
  // -1.
  vector<vector<int>> physRegs_;

  // Find all instructions that use each register.
  void ScanUses_();
  void AllocateReg_(int16_t regType, int virtRegNum);
  // Find a candidate physical register to spill.
  int FindSpillCand_(std::map<int, RegMap> &regMaps, vector<int> &physRegs);
  // Load live-in virtual registers. Live-in registers are defined by the
  // artificial entry instruction.
  void AddLiveIn_(SchedInstruction *artificialEntry);
  // Spill all dirty registers.
  void SpillAll_();
};

} // namespace opt_sched
} // namespace llvm

#endif
