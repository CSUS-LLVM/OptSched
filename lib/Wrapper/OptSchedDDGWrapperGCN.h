//===-- OptSchedDDGWrapperGCN.h - GCN DDG Wrapper ---------------*- C++ -*-===//
//
// Conversion from LLVM ScheduleDAG to OptSched DDG for amdgcn target.
//
//===----------------------------------------------------------------------===//

#include "OptSchedDDGWrapperBasic.h"
#include "OptimizingScheduler.h"

#ifndef LLVM_OPT_SCHED_DDG_WRAPPER_GCN_H
#define LLVM_OPT_SCHED_DDG_WRAPPER_GCN_H

namespace llvm {

namespace opt_sched {

class OptSchedDDGWrapperGCN : public OptSchedDDGWrapperBasic {

public:
  OptSchedDDGWrapperGCN(llvm::MachineSchedContext *Context,
                        ScheduleDAGOptSched *DAG, LLVMMachineModel *MM,
                        LATENCY_PRECISION LatencyPrecision,
                        GraphTransTypes GraphTransTypes,
                        const std::string &RegionID);

  void convertRegFiles() override;
};

} // end namespace opt_sched

} // end namespace llvm

#endif // LLVM_OPT_SCHED_DDG_WRAPPER_GCN_H
