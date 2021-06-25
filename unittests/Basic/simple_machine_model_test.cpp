#include "simple_machine_model.h"

#include "gtest/gtest.h"

using llvm::opt_sched::MachineModel;

namespace {
TEST(SimpleMachineModel, CanBeLoaded) {
  MachineModel Model = simpleMachineModel();
  EXPECT_EQ(1, Model.GetIssueRate());
}
} // namespace
