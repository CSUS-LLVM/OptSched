//===- OptSchedDDGWrapperBase.h - Interface for DDG wrapper -----*- C++-*--===//
//
// Convert an LLVM ScheduleDAG into an OptSched DDG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_DDG_WRAPPER_BASE_H
#define LLVM_OPT_SCHED_DDG_WRAPPER_BASE_H

#include <hip/hip_runtime.h>

namespace llvm {
namespace opt_sched {

class OptSchedDDGWrapperBase {
public:
  __host__ __device__
  virtual ~OptSchedDDGWrapperBase() {}

  virtual void convertSUnits(bool IgnoreRealEdges, bool IgnoreArtificialEdges) = 0;

  virtual void convertRegFiles() = 0;
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_OPT_SCHED_DDG_WRAPPER_BASE_H
