//===- GCNOptSched.h - AMDGCN Combinatorial scheudler -----------*- C++ -*-===//
//
//  OptSched combinatorial scheduler driver targeting AMDGCN.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GCN_OPT_SCHED_H
#define LLVM_GCN_OPT_SCHED_H

#include "../OptimizingScheduler.h"
#include "GCNRegPressure.h"
#include "OptSchedGCNTarget.h"

namespace llvm {
namespace opt_sched {

class ScheduleDAGOptSchedGCN : public ScheduleDAGOptSched {
private:
  enum SchedPassStrategy {
    GCNMaxOcc,
    OptSchedMaxOcc,
    OptSchedBalanced,
    OptSchedLowerOccAnalysis,
    OptSchedCommitLowerOcc
  };

  /// Get the minimum occupancy value from the sched.ini settings file. Check
  /// if the value is between 1-10 and gives an error if it is not between the
  /// valid range.
  unsigned getMinOcc();

  /// Analyze the possible improvements from lowering the target occupancy
  /// and decide if we should keep the schedules.
  bool shouldCommitLowerOccSched();

  // Vector of scheduling passes to execute.
  SmallVector<SchedPassStrategy, 4> SchedPasses;

  unsigned MinOcc;

public:
  ScheduleDAGOptSchedGCN(llvm::MachineSchedContext *C,
                         std::unique_ptr<MachineSchedStrategy> S);

  // After the scheduler is initialized and the scheduling regions have been
  // recorded, execute the actual scheduling passes here.
  void finalizeSchedule() override;

  // Setup and select schedulers.
  void initSchedulers() override;

  // TODO: After we refactor OptSched scheduler options put each scheduling
  // pass into its own class.

  // Execute a scheduling pass on the function.
  void runSchedPass(SchedPassStrategy S);

  // Run GCN max occupancy scheduler.
  void scheduleGCNMaxOcc();

  // Run OptSched in RP only (max occupancy) configuration.
  void scheduleOptSchedMaxOcc();

  // Run OptSched in ILP/RP balanced mode.
  void scheduleOptSchedBalanced() override;

  // Lower occupancy and run OptSched in ILP/RP balanced mode for analysis.
  void scheduleOptSchedLowerOccAnalysis();

  // Lower occupancy and run OptSched in ILP/RP balanced mode to commit
  // scheduling in analysis pass.
  void scheduleCommitLowerOcc();
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_GCN_OPT_SCHED_H
