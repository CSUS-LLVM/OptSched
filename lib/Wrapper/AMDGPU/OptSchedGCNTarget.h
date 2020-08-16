#ifndef LLVM_GCN_OPT_SCHED_TARGET_H
#define LLVM_GCN_OPT_SCHED_TARGET_H

#include "OptSchedDDGWrapperGCN.h"
#include "SIMachineFunctionInfo.h"
#include "Wrapper/OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using namespace llvm::opt_sched;

class OptSchedGCNTarget : public OptSchedTarget {
public:
  std::unique_ptr<OptSchedMachineModel>
  createMachineModel(const char *ConfigPath) override {
    return llvm::make_unique<OptSchedMachineModel>(ConfigPath);
  }

  std::unique_ptr<OptSchedDDGWrapperBase>
  createDDGWrapper(llvm::MachineSchedContext *Context, ScheduleDAGOptSched *DAG,
                   OptSchedMachineModel *MM, LATENCY_PRECISION LatencyPrecision,
                   const std::string &RegionID) override {
    return llvm::make_unique<OptSchedDDGWrapperGCN>(Context, DAG, MM,
                                                    LatencyPrecision, RegionID);
  }

  void initRegion(llvm::ScheduleDAGInstrs *DAG, MachineModel *MM_) override;

  void finalizeRegion(const InstSchedule *Schedule) override;

  // Returns occupancy cost with number of VGPRs and SGPRs from PRP for
  // a partial or complete schedule.
  InstCount getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const override;

  void dumpOccupancyInfo(const InstSchedule *Schedule) const;

  // Revert scheduing if we decrease occupancy.
  bool shouldKeepSchedule() override;

  void limitOccupancy(unsigned Limit);
  unsigned getTargetOcc() { return TargetOccupancy; }
  void setTargetOcc(unsigned Target);

private:
  const llvm::MachineFunction *MF;
  SIMachineFunctionInfo *MFI;
  ScheduleDAGOptSched *DAG;
  const GCNSubtarget *ST;

  unsigned RegionStartingOccupancy;
  unsigned RegionEndingOccupancy;
  unsigned TargetOccupancy;

  // Max occupancy with local memory size;
  unsigned MaxOccLDS;

  // In RP only (max occupancy) scheduling mode we should try to find
  // a min-RP schedule without considering perf hints which suggest limiting
  // occupancy. Returns true if we should consider perf hints.
  bool shouldLimitWaves() const;

  // Find occupancy with spill cost.
  unsigned getOccupancyWithCost(const InstCount Cost) const;
};

#endif
