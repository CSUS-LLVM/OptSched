#ifndef OPTSCHED_TESTS_DDG_H
#define OPTSCHED_TESTS_DDG_H

#include "opt-sched/Scheduler/data_dep.h"
#include "simple_machine_model.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

std::shared_ptr<llvm::opt_sched::DataDepGraph>
makeDDG(const std::string &DDG,
        llvm::opt_sched::MachineModel *Model = nullptr) {
  using namespace llvm::opt_sched;

  class SimpleDDG : public DataDepGraph {
  public:
    using DataDepGraph::DataDepGraph;

    void convertSUnits(bool, bool) override {
      FAIL() << "Unsupported operation convertSUnits()";
    }
    void convertRegFiles() override {
      FAIL() << "Unsupported operation convertRegFile()";
    }
  };

  struct DDGData {
    std::unique_ptr<MachineModel> Model = nullptr;
    SimpleDDG DDG;

    DDGData(MachineModel *Model) : DDG(Model) {}
    DDGData()
        : Model(llvm::make_unique<MachineModel>(simpleMachineModel())),
          DDG(Model.get()) {}
  };

  auto Result =
      Model ? std::make_shared<DDGData>(Model) : std::make_shared<DDGData>();
  auto Ret = Result->DDG.ReadFromString(DDG);
  EXPECT_TRUE(Ret != RES_ERROR && Ret != RES_FAIL && Ret != RES_TIMEOUT)
      << "Failed to parse DDG";
  return std::shared_ptr<DataDepGraph>(Result, &Result->DDG);
}

#endif
