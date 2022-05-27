#ifndef OPT_SCHED_REG
#define OPT_SCHED_REG

#include "Wrapper/AMDGPU/GCNOptSched.h"
#include "Wrapper/AMDGPU/OptSchedGCNTarget.cpp"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
namespace opt_sched {

// Create OptSched ScheduleDAG.
static ScheduleDAGInstrs *createOptSchedGCN(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG = new ScheduleDAGOptSchedGCN(
      C, std::make_unique<GCNMaxOccupancySchedStrategy>(C));
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createStoreClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

static MachineSchedRegistry
    OptSchedGCNMIRegistry("gcn-optsched", "Use the GCN OptSched scheduler.",
                          createOptSchedGCN);

} // namespace opt_sched
} // namespace llvm

#endif
