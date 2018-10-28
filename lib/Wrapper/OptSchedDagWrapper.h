/*******************************************************************************
Description:  A wrapper that convert an LLVM ScheduleDAG to an OptSched
              DataDepGraph.
*******************************************************************************/

#ifndef OPTSCHED_DAG_WRAPPER_H
#define OPTSCHED_DAG_WRAPPER_H

#include "llvm/CodeGen/MachineScheduler.h"
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <map>
#include <vector>

namespace opt_sched {

class LLVMRegTypeFilter;

class LLVMDataDepGraph : public DataDepGraph {
public:
  LLVMDataDepGraph(llvm::MachineSchedContext *context,
                   llvm::ScheduleDAGMILive *llvmDag, LLVMMachineModel *machMdl,
                   LATENCY_PRECISION ltncyPrcsn, llvm::MachineBasicBlock *BB,
                   GraphTransTypes graphTransTypes,
                   const std::vector<unsigned> &RegionPressure,
                   bool treatOrderDepsAsDataDeps, int maxDagSizeForPrcisLtncy,
                   int regionNum);
  ~LLVMDataDepGraph() = default;

  // Returns a pointer to the SUnit at a given node index.
  llvm::SUnit *GetSUnit(size_t index) const;

  // Counts the maximum number of virtual registers of each type used by the
  // graph.
  virtual void CountDefs(RegisterFile regFiles[]);
  // Counts the number of definitions and usages for each register and updates
  // instructions to point to the registers they define/use.
  virtual void AddDefsAndUses(RegisterFile regFiles[]);

protected:
  // A convenience machMdl_ pointer casted to LLVMMachineModel*.
  LLVMMachineModel *llvmMachMdl_;
  // A reference to the nodes of the LLVM DAG.
  std::vector<llvm::SUnit> &llvmNodes_;
  // An reference to the LLVM scheduler root class, used to access environment
  // and target info.
  llvm::MachineSchedContext *context_;
  // An reference to the LLVM Schedule DAG.
  llvm::ScheduleDAGMILive *schedDag_;
  // Precision of latency info
  LATENCY_PRECISION ltncyPrcsn_;
  // An option to treat data dependencies of type ORDER as data dependencies
  bool treatOrderDepsAsDataDeps_;
  // The maximum DAG size to be scheduled using precise latency information
  int maxDagSizeForPrcisLtncy_;
  // The index of the last "assigned" register for each register type.
  std::vector<int> regIndices_;
  // Count each definition of a virtual register with the same resNo
  // as a seperate register in our model. Each resNo is also associated
  // with multiple pressure sets which are treated as seperate registers
  std::map<unsigned, std::vector<Register *>> lastDef_;
  // LLVM object with information about the machine we are targeting
  const llvm::TargetMachine &target_;
  // Peak register pressure before scheduling calculate by LLVM.
  const std::vector<unsigned> &RegionPressure;
  // Allow the DAG builder to filter our register types that have low peak pressure.
  bool ShouldFilterRegisterTypes = false;
  // Use to ignore non-critical register types.
  std::unique_ptr<LLVMRegTypeFilter> RTFilter;
  // Check is SUnit is a root node
  bool isRootNode(const llvm::SUnit &unit);
  // Check is SUnit is a leaf node
  bool isLeafNode(const llvm::SUnit &unit);
  // Check if two nodes are equivalent and if we can order them arbitrarily
  bool nodesAreEquivalent(const llvm::SUnit &srcNode,
                          const llvm::SUnit &dstNode);
  // Get the weight of the regsiter class in LLVM
  int GetRegisterWeight_(const unsigned resNo) const;
  // Add a live-in register.
  void AddLiveInReg_(unsigned resNo, RegisterFile regFiles[]);
  // Add a live-out register.
  void AddLiveOutReg_(unsigned resNo, RegisterFile regFiles[]);
  // Add a Use.
  void AddUse_(unsigned resNo, InstCount nodeIndex, RegisterFile regFiles[]);
  // Add a Def.
  void AddDef_(unsigned resNo, InstCount nodeIndex, RegisterFile regFiles[]);
  // Add registers that are defined-and-not-used.
  void AddDefAndNotUsed_(Register *reg, RegisterFile regFiles[]);
  // Converts the LLVM nodes saved in llvmNodes_ to opt_sched::DataDepGraph.
  // Should be called only once, by the constructor.
  void ConvertLLVMNodes_();
  // Returns the register pressure set types of an instruction result.
  std::vector<int> GetRegisterType_(const unsigned resNo) const;
  // Print register information for the region.
  void dumpRegisters(const RegisterFile regFiles[]) const;

  // Holds a register live range, mapping a producer to a set of consumers.
  struct LiveRange {
    // The node which defines the register tracked by this live range.
    SchedInstruction *producer;
    // The nodes which use the register tracked by this live range.
    std::vector<SchedInstruction *> consumers;
  };
};

// Disallow certain registers from being visible to the scheduler. Use LLVM's
// register pressure tracker to find the MAX register pressure for each register
// type (pressure set). If the MAX pressure is below a certain threshold don't
// track that register.
class LLVMRegTypeFilter {
public:
  LLVMRegTypeFilter(const MachineModel *MM, const llvm::TargetRegisterInfo *TRI,
                    const std::vector<unsigned> &RegionPressure,
                    float RegFilterFactor = .7f);
  ~LLVMRegTypeFilter() = default;

  // The proportion of the register pressure set limit that a register's Max
  // pressure must be higher than in order to not be filtered out. (default .7)
  // The idea is that there is no point in trying to reduce the register
  // pressure
  // of a register type that is in no danger of causing spilling. If the
  // RegFilterFactor is .7, and a random register type has a pressure limit of
  // 10, then we filter out the register types if the MAX pressure for that type
  // is below 7. (10 * .7 = 7)
  void setRegFilterFactor(const float RegFilterFactor);
  // Return true if this register type should be filtered out.
  // Indexed by RegTypeID
  bool shouldFilter(const int16_t RegTypeID) const;
  // Return true if this register type should be filtered out.
  // Indexed by RegTypeName
  bool shouldFilter(const char *RegTypeName) const;
  // Return true if this register type should be filtered out.
  // Indexed by RegTypeID
  bool operator[](const int16_t RegTypeID) const;
  // Return true if this register type should be filtered out.
  // Indexed by RegTypeName
  bool operator[](const char *RegTypeName) const;

private:
  const MachineModel *MM;
  const llvm::TargetRegisterInfo *TRI;
  const std::vector<unsigned> &RegionPressure;
  float RegFilterFactor;
  std::map<const int16_t, bool> RegTypeIDFilteredMap;
  std::map<const char *, bool> RegTypeNameFilteredMap;

  // The current implementation of this class filters register by
  // TRI->getRegPressureSetLimit
  void FindPSetsToFilter();
};

} // end namespace opt_sched

#endif
