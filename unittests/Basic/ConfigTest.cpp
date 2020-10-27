#include "opt-sched/Scheduler/config.h"

#include <sstream>

#include "gtest/gtest.h"

using llvm::opt_sched::Config;
using llvm::opt_sched::SchedulerOptions;

namespace {

TEST(Config, ReadString) {
  Config config;
  std::istringstream input(R"(
        KEY VALUE
    )");
  config.Load(input);

  EXPECT_EQ("VALUE", config.GetString("KEY"));
}

TEST(Config, ReadStringPath) {
  Config config;
  std::istringstream input(R"(
        KEY some/path/
    )");
  config.Load(input);

  EXPECT_EQ("some/path/", config.GetString("KEY"));
}

TEST(Config, ReadInt) {
  Config config;
  std::istringstream input(R"(
        KEY 1
    )");
  config.Load(input);

  EXPECT_EQ(1, config.GetInt("KEY"));
}

TEST(Config, ReadFloat) {
  Config config;
  std::istringstream input(R"(
        KEY 1.3
    )");
  config.Load(input);

  EXPECT_EQ(1.3f, config.GetFloat("KEY"));
}

class TrueBoolTest : public testing::TestWithParam<std::string> {};

TEST_P(TrueBoolTest, ReadBool) {
  Config config;
  std::istringstream input("KEY " + GetParam());
  config.Load(input);

  EXPECT_TRUE(config.GetBool("KEY"));
}

INSTANTIATE_TEST_CASE_P(TrueBoolStrings, TrueBoolTest,
                        testing::Values("1", "yes", "YES", "true", "TRUE"), );

class FalseBoolTest : public testing::TestWithParam<std::string> {};

TEST_P(FalseBoolTest, ReadBool) {
  Config config;
  std::istringstream input("KEY " + GetParam());
  config.Load(input);

  EXPECT_FALSE(config.GetBool("KEY"));
}

INSTANTIATE_TEST_CASE_P(FalseBoolStrings, FalseBoolTest,
                        testing::Values("0", "no", "NO", "false", "FALSE"), );

class StringListTest : public testing::TestWithParam<
                           std::pair<std::vector<std::string>, std::string>> {};

TEST_P(StringListTest, ReadStrings) {
  Config config;
  std::istringstream input("KEY " + GetParam().second);
  config.Load(input);

  const auto result = config.GetStringList("KEY");
  const std::vector<std::string> strings(result.begin(), result.end());

  const std::vector<std::string> expected = GetParam().first;

  EXPECT_EQ(expected, strings);
}

INSTANTIATE_TEST_CASE_P(
    TestCases, StringListTest,
    testing::ValuesIn(
        std::vector<std::pair<std::vector<std::string>, std::string>>{
            {{"singleton"}, "singleton"},
            {{"a", "b"}, "a,b"},
            {{"a", "b", "c", "d", "E", "f", "g", "h", "i"},
             "a,b,c,d,E,f,g,h,i"},

            {{}, ""},
            {{"singleton"}, "singleton,"},
            {{"a", "b"}, "a,b,\nc,d"},
            {{"a"}, "a, b"},
        }), );

class IntListTest : public testing::TestWithParam<
                        std::pair<std::vector<int64_t>, std::string>> {};

TEST_P(IntListTest, ReadInts) {
  Config config;
  std::istringstream input("KEY " + GetParam().second);
  config.Load(input);

  const auto result = config.GetIntList("KEY");
  const std::vector<int64_t> ints(result.begin(), result.end());

  const std::vector<int64_t> expected = GetParam().first;

  EXPECT_EQ(expected, ints);
}

INSTANTIATE_TEST_CASE_P(
    TestCases, IntListTest,
    testing::ValuesIn(std::vector<std::pair<std::vector<int64_t>, std::string>>{
        {{1}, "1"},
        {{-1, 0}, "-1,0"},

        {{}, ""},
        {{-2, -3}, "-2,-3\n4,5"},
        {{832, 43}, "832,43"},
    }), );

class FloatListTest : public testing::TestWithParam<
                          std::pair<std::vector<float>, std::string>> {};

TEST_P(FloatListTest, ReadFloats) {
  Config config;
  std::istringstream input("KEY " + GetParam().second);
  config.Load(input);

  const auto result = config.GetFloatList("KEY");
  const std::vector<float> ints(result.begin(), result.end());

  const std::vector<float> expected = GetParam().first;

  EXPECT_EQ(expected, ints);
}

INSTANTIATE_TEST_CASE_P(
    TestCases, FloatListTest,
    testing::ValuesIn(std::vector<std::pair<std::vector<float>, std::string>>{
        {{1.0f}, "1"},
        {{-1.5f, 0.02f}, "-1.5,0.02"},

        {{}, ""},
        {{-0.2f, -3}, "-0.2,-3\n4,5"},
        {{832.123f, 43}, "832.123,43"},
    }), );

} // namespace
