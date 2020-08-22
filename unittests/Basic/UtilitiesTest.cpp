#include "opt-sched/Scheduler/utilities.h"

#include "gtest/gtest.h"

namespace utils = llvm::opt_sched::Utilities;

namespace {

TEST(Utilities, clcltBitsNeededToHoldNum) {
  EXPECT_EQ(0, utils::clcltBitsNeededToHoldNum(0));
  EXPECT_EQ(1, utils::clcltBitsNeededToHoldNum(1));
  EXPECT_EQ(2, utils::clcltBitsNeededToHoldNum(2));
  EXPECT_EQ(2, utils::clcltBitsNeededToHoldNum(3));
  EXPECT_EQ(3, utils::clcltBitsNeededToHoldNum(4));

  EXPECT_EQ(16, utils::clcltBitsNeededToHoldNum(0x8000));
}

} // namespace
