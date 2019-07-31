//===-- OptSchedDDGWrapperGCN.h - GCN DDG Wrapper ---------------*- C++ -*-===//
//
// Conversion from LLVM ScheduleDAG to OptSched DDG for amdgcn target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_DDG_WRAPPER_GCN_H
#define LLVM_OPT_SCHED_DDG_WRAPPER_GCN_H

#include "GCNRegPressure.h"
#include "Wrapper/OptSchedDDGWrapperBasic.h"
#include "Wrapper/OptimizingScheduler.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LiveIntervals.h"

namespace llvm {
namespace opt_sched {

class SubRegSet {
private:
  // Index subreg lanes to OptSched register numbers. Even though we can't map
  // a LaneBitmask index to a specific sub-register, we can still accurately
  // model the correct number of live subregs using lane mask interference.
  SmallVector<unsigned, 8> OptSchedRegMap;

public:
  using iterator = SmallVector<unsigned, 8>::iterator;
  // The max number of subregs for this virtual register.
  unsigned Size;
  // OptSched register type
  int16_t Type;

  iterator begin() { return OptSchedRegMap.begin(); }
  iterator end() { return OptSchedRegMap.end(); }

  SubRegSet(unsigned Size_, int16_t Type_) : Size(Size_), Type(Type_) {
    OptSchedRegMap.resize(Size);
  }
  ~SubRegSet() = default;
};

class OptSchedDDGWrapperGCN : public OptSchedDDGWrapperBasic {
private:
  // Map sub-registers in LLVM to a list of live subreg lanes for that register.
  // Each live lane represents either a VGPR32 or SGPR32. In our model each live
  // subreg lane is identified by a separate OptSched register.
  using RegsMap = DenseMap<unsigned, std::unique_ptr<SubRegSet>>;
  RegsMap RegionRegs;
  const std::vector<llvm::SUnit> &SUnits;
  const llvm::LiveIntervals *LIS;
  const llvm::MachineRegisterInfo &MRI;

  unsigned getRegKind(unsigned Reg) const;

  void addLiveSubRegsAtInstr(const MachineInstr *MI, bool After);

  void addSubRegDefs(SchedInstruction *Instr, unsigned Reg,
                     const LaneBitmask &LiveMask, bool LiveIn = false);

  void addSubRegUses(SchedInstruction *Instr, unsigned Reg,
                     const LaneBitmask &LiveMask, bool LiveOut = false);

public:
  // FIXME: Track VGPR/SGPR tuples or refactor Scheduler to use LLVM/GCN RP
  // tracker.
  enum SubRegKind { SGPR32, VGPR32, TOTAL_KINDS };

  OptSchedDDGWrapperGCN(llvm::MachineSchedContext *Context,
                        ScheduleDAGOptSched *DAG, OptSchedMachineModel *MM,
                        LATENCY_PRECISION LatencyPrecision,
                        const std::string &RegionID);

  void convertRegFiles() override;
};

} // end namespace opt_sched
} // end namespace llvm

#endif // LLVM_OPT_SCHED_DDG_WRAPPER_GCN_H
