//===- OptSchedTarget.h - OptSched Target -----------------------*- C++-*--===//
//
// Interface for target specific functionality in OptSched. This is a workaround
// to avoid needing to modify or use target code in the trunk.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_TARGET_H
#define LLVM_OPT_SCHED_TARGET_H

#include "opt-sched/Scheduler/OptSchedDDGWrapperBase.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {
namespace opt_sched {

class OptSchedMachineModel;
class ScheduleDAGOptSched;

class OptSchedTarget {
public:
  MachineModel *MM;

  virtual ~OptSchedTarget() = default;

  virtual std::unique_ptr<OptSchedMachineModel>
  createMachineModel(const char *configFile) = 0;

  virtual std::unique_ptr<OptSchedDDGWrapperBase>
  createDDGWrapper(MachineSchedContext *Context, ScheduleDAGOptSched *DAG,
                   OptSchedMachineModel *MM, LATENCY_PRECISION LatencyPrecision,
                   const std::string &RegionID) = 0;

  virtual void initRegion(ScheduleDAGInstrs *DAG, MachineModel *MM) = 0;
  virtual void finalizeRegion(const InstSchedule *Schedule) = 0;
  // FIXME: This is a shortcut to doing the proper thing and creating a RP class
  // that targets can override. It's hard to justify spending the extra time
  // when we will be refactoring RP tracking in general if we do a rewrite to
  // fully integrate the scheduler in LLVM.
  //
  // Get target specific cost from peak register pressure (e.g. occupancy for
  // AMDGPU)
  virtual InstCount
  getCost(const llvm::SmallVectorImpl<unsigned> &PRP) const = 0;

  // Targets that wish to discard the finalized schedule for any reason can
  // override this.
  virtual bool shouldKeepSchedule() { return true; }
};

template <typename FactoryT> class OptSchedRegistryNode {
public:
  llvm::SmallString<16> Name;
  FactoryT Factory;
  OptSchedRegistryNode *Next;

  OptSchedRegistryNode(llvm::StringRef Name_, FactoryT Factory_)
      : Name(Name_), Factory(Factory_) {}
};

template <typename FactoryT> class OptSchedRegistry {
private:
  OptSchedRegistryNode<FactoryT> *List = nullptr;
  OptSchedRegistryNode<FactoryT> *Default = nullptr;

public:
  void add(OptSchedRegistryNode<FactoryT> *Node) {
    Node->Next = List;
    List = Node;
  }

  FactoryT getFactoryWithName(llvm::StringRef Name) {
    FactoryT Factory = nullptr;
    for (auto I = List; I; I = I->Next)
      if (I->Name == Name) {
        Factory = I->Factory;
        break;
      }
    return Factory;
  }

  void setDefault(llvm::StringRef Name) {
    OptSchedRegistryNode<FactoryT> Node = nullptr;
    for (auto I = List; I; I = I->Next)
      if (I->Name == Name) {
        Node = I;
        break;
      }
    assert(Node && "Could not set default factory! None in list with name.");
    Default = Node;
  }

  FactoryT getDefaultFactory() {
    assert(Default && "Default factory not set.");
    return Default->Factory;
  }
};

class OptSchedTargetRegistry
    : public OptSchedRegistryNode<std::unique_ptr<OptSchedTarget> (*)()> {
public:
  using OptSchedTargetFactory = std::unique_ptr<OptSchedTarget> (*)();
  static OptSchedRegistry<OptSchedTargetFactory> Registry;

  OptSchedTargetRegistry(llvm::StringRef Name_, OptSchedTargetFactory Factory_)
      : OptSchedRegistryNode(Name_, Factory_) {
    Registry.add(this);
  }
};

} // namespace opt_sched
} // namespace llvm

#endif // LLVM_OPT_SCHED_TARGET_H
