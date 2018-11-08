//===- OptSchedDDGWrapperGCN.cpp - GCN DDG Wrapper ------------------------===//
//
// Conversion from LLVM ScheduleDAG to OptSched DDG for amdgcn target.
//
//===----------------------------------------------------------------------===//

#include "OptSchedDDGWrapperGCN.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "optsched"

using namespace llvm;
using namespace llvm::opt_sched;

OptSchedDDGWrapperGCN::OptSchedDDGWrapperGCN(MachineSchedContext *Context,
                                             ScheduleDAGOptSched *DAG,
                                             LLVMMachineModel *MM,
                                             LATENCY_PRECISION LatencyPrecision,
                                             GraphTransTypes GraphTransTypes,
                                             const std::string &RegionID)
    : OptSchedDDGWrapperBasic(Context, DAG, MM, LatencyPrecision,
                              GraphTransTypes, RegionID) {}

void OptSchedDDGWrapperGCN::convertRegFiles() {}
