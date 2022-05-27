//===- OptimizingScheduler.h - Combinatorial scheduler ----------*- C++ -*-===//
//
// Integrates an alternative scheduler into LLVM which implements a
// combinatorial scheduling algorithm.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_OPTIMIZING_SCHEDULER_H
#define LLVM_OPT_SCHED_OPTIMIZING_SCHEDULER_H

#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"
#include <chrono>
#include <memory>
#include <vector>

using namespace llvm;

namespace llvm {
namespace opt_sched {

class OptSchedDDGWrapperBasic;

class ScheduleDAGOptSched : public ScheduleDAGMILive {

private:
  enum SchedPassStrategy {
    OptSchedMinRP,
    OptSchedBalanced,
    OptSchedReducedLatency,
    OptSchedSeqScheduler
  };

  // Vector of scheduling passes to execute.
  SmallVector<SchedPassStrategy, 4> SchedPasses;

protected:
  // Vector of regions recorded for later rescheduling
  SmallVector<
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>, 32>
      Regions;

  // Path to opt-sched config options directory.
  SmallString<128> PathCfg;

  // Path to the scheduler options configuration file for opt-sched.
  SmallString<128> PathCfgS;

  // Path to the list of hot functions to schedule using opt-sched.
  SmallString<128> PathCfgHF;

  // Path to the machine model specification file for opt-sched.
  SmallString<128> PathCfgMM;

  // Path to the occupancy limits file for specified kernels using opt-sched
  SmallString<128> PathCfgOCL;

  // Bool value indicating that the scheduler is in the second
  // pass. Used for the two pass scheduling approach.
  bool SecondPass;

  // Region number uniquely identifies DAGs.
  unsigned RegionNumber = ~0u;

  MachineSchedContext *C;

  // The OptSched target machine.
  std::unique_ptr<OptSchedTarget> OST;

  // into the OptSched machine model
  std::unique_ptr<OptSchedMachineModel> MM;

  // A list of functions that are indicated as candidates for the
  // OptScheduler
  Config HotFunctions;

  // A list of kernels / functions and the occupancy limit the maximizes
  // performance
  Config OccupancyLimits;

  // Struct for setting the pruning strategy
  Pruning PruningStrategy;

  // If we should schedule for register pressure only and ignore ilp.
  bool SchedForRPOnly;

  // Flag indicating whether the optScheduler should be enabled for this
  // function
  bool OptSchedEnabled;

  // Flag indicating whether the two pass scheduling approach should be used
  // instead of the original one pass scheduling.
  bool TwoPassEnabled;

  /// The order of the passes to execute.
  std::list<std::string> PassOrder;

  // Flag indicating whether or not the two pass scheduling approach
  // has started. The two pass scheduling approach starts in finalizeSchedule.
  bool TwoPassSchedulingStarted;

  /// Flag indicating whether or not the ILP Reduced Latency pass has started.
  bool LatencyPassStarted;

  /// Settings to control the ILP Reduced Latencies pass
  /// If latency for an edge is greater than this amount then reduce the
  /// latency by
  int LatencyTarget;

  /// Settings to control the ILP Reduced Latencies pass
  /// The amount to divide the original latency by.
  int LatencyDivisor;

  /// Settings to control the ILP Reduced Latencies pass
  /// The minimum amount latency can be reduced to.
  int LatencyMinimun;

  /// Flag indicating whether or not to run another pass with actual latencies.
  /// This is useful for gathering actual schedule length data but can be
  /// disabled if schedule length data is not required.
  bool CompileTimeDataPass;

  /// Begin recording regions that we want to reschedule
  bool RecordTimedOutRegions;

  /// Store which region to reschedule.
  llvm::BitVector RescheduleRegions;

  // Precision of latency info
  LATENCY_PRECISION LatencyPrecision;

  // A time limit for the whole region (basic block) in milliseconds.
  // Defaults to no limit.
  int RegionTimeout;

  // A time limit for each schedule length in milliseconds.
  int LengthTimeout;

  // Time limits for two pass scheduling. Same as the above time limits
  // but for each individual pass.
  int FirstPassRegionTimeout;
  int FirstPassLengthTimeout;
  int SecondPassRegionTimeout;
  int SecondPassLengthTimeout;

  int TimeoutPerMemblock;

  int OccupancyLimit;

  bool ShouldLimitOccupancy;
  OCC_LIMIT_TYPE OccupancyLimitSource;

  // How to interpret the timeout value? Timeout per instruction or
  // timout per block
  bool IsTimeoutPerInst;

  // The maximum number of instructions to schedule with our scheduler.
  // Beyond that, it uses the heuristic scheduler.
  unsigned MaxRegionInstrs;

  // The maximum number of instructions that a block can contain to be
  // Treat data dependencies of type ORDER as data dependencies
  bool TreatOrderAsDataDeps;

  // The number of bits in the hash table used in history-based domination.
  int16_t HistTableHashBits;

  // Whether to verify that calculated schedules are optimal. Defaults to NO.
  bool VerifySchedule;

  // Whether to enumerate schedules containing stalls (no-op instructions).
  // In certain cases, such as having unpipelined instructions, this may
  // result in a better schedule. Defaults to YES
  bool EnumStalls;

  // Whether to apply LLVM mutations to the DAG before scheduling
  bool EnableMutations;

  // The weight of the spill cost in the objective function. This factor
  // defines the importance of spill cost relative to schedule length. A good
  // value for this factor should be found experimentally, but is is expected
  // to be large on architectures with hardware scheduling like x86 (thus
  // making spill cost minimization the primary objective) and smaller on
  // architectures with in-order execution like SPARC (thus making scheduling
  // the primary objective).
  int SCW;

  // In ISO mode this is the original DAG before ISO conversion.
  std::vector<SUnit> OriginalDAG;

  // The schedule generated by LLVM for ISO mode.
  std::vector<unsigned> ISOSchedule;

  // The spill cost function to be used.
  SPILL_COST_FUNCTION SCF;
  SPILL_COST_FUNCTION SecondPassSCF;

  // The algorithm to use for determining the lower bound. Valid values are
  LB_ALG LowerBoundAlgorithm;

  // The heuristic used for the list scheduler.
  SchedPriorities HeuristicPriorities;

  // The heuristic used for the enumerator.
  SchedPriorities EnumPriorities;

  // The heuristic used for the second pass enumerator in the two-pass
  // scheduling approach.
  SchedPriorities SecondPassEnumPriorities;

  GT_POSITION GraphTransPosition = GT_POSITION::NONE;
  GT_POSITION GraphTransPosition2ndPass = GT_POSITION::NONE;

  // Static node superiority RP only graph transformation.
  bool StaticNodeSup;

  // ILP Static Node Superiority graph transformation
  bool ILPStaticNodeSup;

  // Occupancy-preserving ILP Static Node Superiority graph transformation
  bool OccupancyPreservingILPStaticNodeSup;

  // Occupancy-preserving ILP Static Node Superiority graph transformation
  bool OccupancyPreservingILPStaticNodeSup2ndPass;

  // Run multiple passes of the static node superiority algorithm
  // (StaticNodeSup must be enabled).
  bool MultiPassStaticNodeSup;

  // Should we run the LLVM converging scheduler as input to the enumerator.
  bool UseLLVMScheduler;

  // The number of simulated register spills in this function
  int SimulatedSpills;

  // What list scheduler should be used to find an initial feasible schedule.
  SchedulerType HeurSchedType;

  // Load config files for the OptScheduler and set flags
  void loadOptSchedConfig();

  // Get lower bound algorithm
  LB_ALG parseLowerBoundAlgorithm() const;

  // Get spill cost function
  SPILL_COST_FUNCTION parseSpillCostFunc() const;

  // Get the GT_POSITION
  static GT_POSITION parseGraphTransPosition(llvm::StringRef Str);

  OCC_LIMIT_TYPE parseOccLimit(const std::string Str);

  // Return true if the OptScheduler should be enabled for the function this
  // ScheduleDAG was created for
  bool isOptSchedEnabled() const;

  // Return true if the two pass scheduling approach should be enabled
  bool isTwoPassEnabled() const;

  // get latency precision setting
  LATENCY_PRECISION fetchLatencyPrecision() const;

  // Get OptSched heuristic setting
  SchedPriorities parseHeuristic(const std::string &str);

  // Return true if we should print spill count for the current function
  bool shouldPrintSpills() const;

  // Reset the flags (e.g undef) before reverting scheduling
  void ResetFlags(SUnit &SU);

  // Add node to llvm schedule
  void ScheduleNode(SUnit *SU, unsigned CurCycle);

  // Setup dag and calculate register pressue in region
  void SetupLLVMDag();

  // Check for a mismatch between LLVM and OptSched register pressure values.
  bool rpMismatch(InstSchedule *sched);

  // Is simulated register allocation enabled.
  bool isSimRegAllocEnabled() const;

  // Find the real paths to optsched-cfg files.
  void getRealCfgPaths();

  // Create and add OptSched DDG mutations.
  void addGraphTransformations(OptSchedDDGWrapperBasic *BDDG);

public:
  ScheduleDAGOptSched(MachineSchedContext *C,
                      std::unique_ptr<MachineSchedStrategy> S);

  // The fallback LLVM scheduler
  void fallbackScheduler();

  // Print out total block spills for the function.
  void finalizeSchedule() override;

  // Schedule the current region using the OptScheduler
  void schedule() override;

  // Calculate OptSched scheduling stats based on ordering of SUnits
  void getOptSchedStats();

  // Setup and select schedulers for the two pass scheduling approach.
  virtual void initSchedulers();

  // Execute a scheduling pass on the function.
  void runSchedPass(SchedPassStrategy S);

  // Run OptSched in RP only configuration.
  void scheduleOptSchedMinRP();

  // Run OptSched in ILP/RP balanced mode.
  virtual void scheduleOptSchedBalanced();

  /// If any region timed out then re-schedule them with the latencies reduced
  /// artificially.
  void scheduleWithReducedLatencies();

  /// Compile-time data gathering pass for the ILP Reduced Latency pass.
  void scheduleWithSeqScheduler();

  // Print info for all LLVM registers that are used or defined in the region.
  void dumpLLVMRegisters() const;

  // Getter for region number
  int getRegionNum() const { return RegionNumber; }

  int getLatencyTarget() const { return LatencyTarget; }

  int getLatencyDivisor() const { return LatencyDivisor; }

  int getLatencyMinimun() const { return LatencyMinimun; }

  bool reducedLatencyPassStarted() const { return LatencyPassStarted; }

  // Return the boundary instruction for this region if it is not a sentinel
  // value.
  const MachineInstr *getRegionEnd() const {
    return (RegionEnd == BB->end() ? nullptr : &*RegionEnd);
  }

  LATENCY_PRECISION getLatencyType() const { return LatencyPrecision; }
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_OPT_SCHED_OPTIMIZING_SCHEDULER_H
