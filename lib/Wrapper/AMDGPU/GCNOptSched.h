//===- GCNOptSched.h - AMDGCN Combinatorial scheudler -----------*- C++ -*-===//
//
//  OptSched combinatorial scheduler driver targeting AMDGCN.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GCN_OPT_SCHED_H
#define LLVM_GCN_OPT_SCHED_H

#include "../OptimizingScheduler.h"
#include "GCNRegPressure.h"

namespace llvm {
namespace opt_sched {

class ScheduleDAGOptSchedGCN : public ScheduleDAGOptSched {
private:
  enum SchedPassStrategy { GCNMaxOcc, OptSchedMaxOcc, OptSchedBalanced };

  // Vector of scheduling passes to execute.
  SmallVector<SchedPassStrategy, 4> SchedPasses;

  // Vector of regions recorded for later rescheduling
  SmallVector<
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>, 32>
      Regions;

public:
  ScheduleDAGOptSchedGCN(llvm::MachineSchedContext *C,
                         std::unique_ptr<MachineSchedStrategy> S);

  // Rely on the machine scheduler to split the MBB into scheduling regions. In
  // the first pass record the regions here, but don't do any actual scheduling
  // until finalizeSchedule is called.
  void schedule() override;

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
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_GCN_OPT_SCHED_H
