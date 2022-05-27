#ifndef OPT_SCHED_REG
#define OPT_SCHED_REG

#include "OptimizingScheduler.h"
#include "llvm/CodeGen/MachineScheduler.h"

using namespace llvm;

namespace llvm {
namespace opt_sched {

// Create OptSched ScheduleDAG.
static ScheduleDAGInstrs *createOptSched(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG =
      new ScheduleDAGOptSched(C, std::make_unique<GenericScheduler>(C));
  DAG->addMutation(createCopyConstrainDAGMutation(DAG->TII, DAG->TRI));
  // README: if you need the x86 mutations uncomment the next line.
  // addMutation(createX86MacroFusionDAGMutation());
  // You also need to add the next line somewhere above this function
  //#include "../../../../../llvm/lib/Target/X86/X86MacroFusion.h"
  return DAG;
}

// Register the machine scheduler.
static MachineSchedRegistry OptSchedMIRegistry("optsched",
                                               "Use the OptSched scheduler.",
                                               createOptSched);

} // namespace opt_sched
} // namespace llvm