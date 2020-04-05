#include "opt-sched/Scheduler/logger.h"

#include <sstream>

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm::opt_sched;

namespace {
class LoggerTest : public ::testing::Test {
protected:
  LoggerTest() : old{Logger::GetLogStream()} { Logger::SetLogStream(log); }

  ~LoggerTest() override { Logger::SetLogStream(old); }

  std::string getLog() const { return log.str(); }

private:
  std::ostream &old;
  std::ostringstream log;
};

TEST_F(LoggerTest, EventWorks) {
  Logger::Event("SomeEventID", "key", 42, "key2", "value2", "key3", true,
                "key4", 123ull, "key5", -123ll);
  EXPECT_THAT(
      getLog(),
      ::testing::MatchesRegex(
          R"(EVENT: \{"event_id": "SomeEventID", "key": 42, "key2": "value2", "key3": true, "key4": 123, "key5": -123, "time": [0-9]+\})"
          "\n"));
}

TEST_F(LoggerTest, EmptyEventIncludesOnlyTime) {
  Logger::Event("SomeEventID");
  EXPECT_THAT(getLog(),
              ::testing::MatchesRegex(
                  R"(EVENT: \{"event_id": "SomeEventID", "time": [0-9]+\})"
                  "\n"));
}
} // namespace
