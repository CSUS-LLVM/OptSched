//===- TwoPassScheduler.h - Combinatorial scheduler ----------*- C++ -*-===//
//
// Integrates a two pass scheduler based on our optimizing scheduler and
// our work done on the AMDGPU.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_TWO_PASS_SCHEDULER_H
#define LLVM_OPT_SCHED_TWO_PASS_SCHEDULER_H

#include "OptimizingScheduler.h"

namespace llvm {
namespace opt_sched {

class ScheduleTwoPassOptSched : public ScheduleDAGOptSched {
private:
  enum SchedPassStrategy {
    OptSchedMinRP,
    OptSchedBalanced
  };

  // Vector of scheduling passes to execute.
  SmallVector<SchedPassStrategy, 4> SchedPasses;

  // Vector of regions recorded for later rescheduling
  SmallVector<std::pair<MachineBasicBlock::iterator,
                        MachineBasicBlock::iterator>, 32> Regions;
	
public:
  ScheduleTwoPassOptSched(llvm::MachineSchedContext *C,
                         std::unique_ptr<MachineSchedStrategy> S);
						 
  // Rely on the machine scheduler to split the MBB into scheduling regions. In
  // the first pass record the regions here, but don't do any actual scheduling
  // until finalizeSchedule is called.
  void schedule() override;

  // After the scheduler is initialized and the scheduling regions have been
  // recorded, execute the actual scheduling passes here.
  void finalizeSchedule() override;

  // Setup and select schedulers.
  void initSchedulers();

  // TODO: After we refactor OptSched scheduler options put each scheduling
  // pass into its own class.

  // Execute a scheduling pass on the function.
  void runSchedPass(SchedPassStrategy S);
  
  // Run OptSched in RP only configuration.
  void scheduleOptSchedMinRP();

  // Run OptSched in ILP/RP balanced mode.
  void scheduleOptSchedBalanced();
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_OPT_SCHED_TWO_PASS_SCHEDULER_H
