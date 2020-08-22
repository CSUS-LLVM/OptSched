//===- OptSchedDDGWrapperBase.h - Interface for DDG wrapper -----*- C++-*--===//
//
// Convert an LLVM ScheduleDAG into an OptSched DDG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_DDG_WRAPPER_BASE_H
#define LLVM_OPT_SCHED_DDG_WRAPPER_BASE_H

namespace llvm {
namespace opt_sched {

class OptSchedDDGWrapperBase {
public:
  virtual ~OptSchedDDGWrapperBase() = default;

  virtual void convertSUnits(bool IgnoreRealEdges,
                             bool IgnoreArtificialEdges) = 0;

  virtual void convertRegFiles() = 0;
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_OPT_SCHED_DDG_WRAPPER_BASE_H
