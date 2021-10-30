#include "ddg.h"

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

using namespace llvm::opt_sched;

namespace {
TEST(SimpleDDG, CanBeMade) {
  std::shared_ptr<DataDepGraph> DDG = makeDDG(R"(
dag 7 "Simple"
{
dag_id fake:3
dag_weight 1.000000
compiler LLVM
dag_lb -1
dag_ub -1
nodes
node 0 "Inst"
    sched_order 0
    issue_cycle 0
node 1 "Inst"
    sched_order 1
    issue_cycle 1
node 2 "Inst"
    sched_order 2
    issue_cycle 2
node 3 "Inst"
    sched_order 3
    issue_cycle 3
node 4 "Inst"
    sched_order 4
    issue_cycle 4
node 5 "artificial"  "__optsched_entry"
node 6 "artificial"
dependencies
dep 0 1 "other" 0
dep 1 2 "other" 0
dep 2 6 "other" 0
dep 3 4 "data" 1
dep 4 6 "other" 0
dep 5 3 "other" 0
dep 5 0 "other" 0
}
  )");

  EXPECT_EQ(7, DDG->GetNodeCnt());
}

TEST(SimpleDDG, CanBeMadeWithRealData) {
  MachineModel Model = simpleMachineModel();
  {
    InstTypeInfo Info;
    Info.issuType = Model.getDefaultIssueType();
    Info.name = "ATOMIC_FENCE";
    Info.isCntxtDep = false;
    Info.ltncy = 0;
    Info.pipelined = true;
    Info.sprtd = true;
    Info.blksCycle = true;
    Model.AddInstType(Info);

    Info.name = "S_BARRIER";
    Model.AddInstType(Info);

    Info.name = "S_ADD_I32";
    Info.ltncy = 1;
    Model.AddInstType(Info);

    Info.name = "S_CMP_LT_U32";
    Info.ltncy = 1;
    Model.AddInstType(Info);
  }

  std::shared_ptr<DataDepGraph> DDG = makeDDG(R"(
dag 7 "Simple"
{
dag_id kernel_c18_sdk_94:3
dag_weight 1.000000
compiler LLVM
dag_lb -1
dag_ub -1
nodes
node 0 "ATOMIC_FENCE"
    sched_order 0
    issue_cycle 0
node 1 "S_BARRIER"  "S_BARRIER"
    sched_order 1
    issue_cycle 1
node 2 "ATOMIC_FENCE"  "ATOMIC_FENCE"
    sched_order 2
    issue_cycle 2
node 3 "S_ADD_I32"  "S_ADD_I32"
    sched_order 3
    issue_cycle 3
node 4 "S_CMP_LT_U32"  "S_CMP_LT_U32"
    sched_order 4
    issue_cycle 4
node 5 "artificial"  "__optsched_entry"
node 6 "artificial"
dependencies
dep 0 1 "other" 0
dep 1 2 "other" 0
dep 2 6 "other" 0
dep 3 4 "data" 1
dep 4 6 "other" 0
dep 5 3 "other" 0
dep 5 0 "other" 0
}
  )",
                                              &Model);

  EXPECT_EQ(7, DDG->GetNodeCnt());
}

TEST(SimpleDDG, VirtualFunctionsFailWell) {
  static std::shared_ptr<DataDepGraph> DDG = makeDDG(R"(
dag 3 "Simple"
{
dag_id fake:3
dag_weight 1.000000
compiler LLVM
dag_lb -1
dag_ub -1
nodes
node 0 "Inst"
    sched_order 0
    issue_cycle 0
node 1 "artificial"  "__optsched_entry"
node 2 "artificial"
dependencies
dep 0 1 "other" 0
dep 1 2 "other" 0
}
  )");

  EXPECT_FATAL_FAILURE(DDG->convertRegFiles(),
                       "Unsupported operation convertRegFiles");
  EXPECT_FATAL_FAILURE(DDG->convertSUnits(true, true),
                       "Unsupported operation convertSUnits");
}

TEST(SimpleDDG, MakeInvalidDDGFailsWell) {
  EXPECT_NONFATAL_FAILURE(makeDDG(R"(
dag 3 "Simple"
{
dag_id fake:3
dag_weight 1.000000
compiler LLVM
dag_lb -1
dag_ub -1
nodes
node 0 "Inst"
    sched_order 0
    issue_cycle 0
node 1 "artificial"  "__optsched_entry"
node 2 "artificial"
dependencies
}
  )"),
                          "parse DDG");
}
} // namespace
