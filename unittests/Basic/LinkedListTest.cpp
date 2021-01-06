#include "opt-sched/Scheduler/lnkd_lst.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

using namespace llvm::opt_sched;

namespace {

class LinkedListTest : public testing::TestWithParam<int> {};

TEST_P(LinkedListTest, CanBeIteratedOverUsingStandardIteration) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  LinkedList<int> list(GetParam());

  for (int &x : numbers) {
    list.InsrtElmnt(&x);
  }

  ASSERT_EQ(numbers.size(), std::distance(list.begin(), list.end()));

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.end(), mismatch.first);
  EXPECT_EQ(list.end(), mismatch.second);
}

TEST_P(LinkedListTest, EmptyStandardIterationYieldsNoIteration) {
  LinkedList<int> list(GetParam());

  bool wasTouched = false;
  for (auto &&x : list) {
    (void)x;
    wasTouched = true;
  }

  EXPECT_FALSE(wasTouched);
}

TEST_P(LinkedListTest, SingletonStandardIterationIteratesExactlyOnce) {
  LinkedList<int> list(GetParam());
  int n = 42;
  list.InsrtElmnt(&n);

  std::vector<int> result;
  for (auto &&x : list) {
    result.push_back(x);
  }

  EXPECT_EQ(std::vector<int>{n}, result);
}

TEST(LinkedListFixedFull, SingletonStandardIterationIteratesExactlyOnce) {
  LinkedList<int> list(1);
  int n = 42;
  list.InsrtElmnt(&n);

  std::vector<int> result;
  for (auto &&x : list) {
    result.push_back(x);
  }

  EXPECT_EQ(std::vector<int>{n}, result);
}

TEST(LinkedListFixed, CanAddToMaxCapacityAfterRemove) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> numbers2 = numbers;

  LinkedList<int> list(20);

  const int index = 4;

  for (int &x : numbers2) {
    list.InsrtElmnt(&x);
  }

  auto it = list.begin();
  auto numIt = numbers.begin();

  for (int i = 0; i < index; ++i) {
    ++it;
    ++numIt;
  }

  list.RemoveAt(it);
  numbers.erase(numIt);

  while (list.GetElmntCnt() < 20) {
    list.InsrtElmnt(&numbers2[3]);
    numbers.push_back(numbers2[3]);
  }

  ASSERT_EQ(numbers.size(), list.GetElmntCnt());

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.size(), mismatch.first - numbers.begin())
      << "Expected: " << ::testing::PrintToString(numbers);
  EXPECT_EQ(list.GetElmntCnt(), std::distance(list.begin(), mismatch.second))
      << "Actual:   " << ::testing::PrintToString(list);
}

INSTANTIATE_TEST_CASE_P(DynamicAndFixed, LinkedListTest,
                        testing::Values(INVALID_VALUE, 20), );

class LinkedListRemoveIndexTest
    : public testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(LinkedListRemoveIndexTest, EntriesCanBeRemovedByIterator) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> numbers2 = numbers;

  LinkedList<int> list(std::get<0>(GetParam()));

  int index = std::get<1>(GetParam());
  if (index == -1) {
    index = numbers.size() - 1;
  }

  for (int &x : numbers2) {
    list.InsrtElmnt(&x);
  }

  auto it = list.begin();
  auto numIt = numbers.begin();

  for (int i = 0; i < index; ++i) {
    ++it;
    ++numIt;
  }

  list.RemoveAt(it);
  numbers.erase(numIt);
  ASSERT_EQ(numbers.size(), list.GetElmntCnt());

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.size(), mismatch.first - numbers.begin())
      << "Expected: " << ::testing::PrintToString(numbers);
  EXPECT_EQ(list.GetElmntCnt(), std::distance(list.begin(), mismatch.second))
      << "Actual:   " << ::testing::PrintToString(list);
}

TEST_P(LinkedListRemoveIndexTest, CanAddAfterRemove) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> numbers2 = numbers;

  LinkedList<int> list(std::get<0>(GetParam()));

  int index = std::get<1>(GetParam());
  if (index == -1) {
    index = numbers.size() - 1;
  }

  for (int &x : numbers2) {
    list.InsrtElmnt(&x);
  }

  auto it = list.begin();
  auto numIt = numbers.begin();

  for (int i = 0; i < index; ++i) {
    ++it;
    ++numIt;
  }

  list.RemoveAt(it);
  numbers.erase(numIt);
  list.InsrtElmnt(&numbers[3]);
  numbers.push_back(numbers[3]);

  ASSERT_EQ(numbers.size(), list.GetElmntCnt());

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.size(), mismatch.first - numbers.begin())
      << "Expected: " << ::testing::PrintToString(numbers);
  EXPECT_EQ(list.GetElmntCnt(), std::distance(list.begin(), mismatch.second))
      << "Actual:   " << ::testing::PrintToString(list);
}

INSTANTIATE_TEST_CASE_P(RemoveSeveralIndices, LinkedListRemoveIndexTest,
                        testing::Combine(testing::Values(INVALID_VALUE, 11, 20),
                                         testing::Values(0, 1, 2, 3, -1)), );

TEST(PriorityList, CorruptedByRemovingFirst) {
  std::vector<int> numbers = {0, 1, 2, 3, 4};

  // The structure of this PriorityList is specified as it is because the values
  // come from a bug. That is, the maxSize and which elements we're
  // removing/re-adding are from a bug, and this test is to test against that
  // bug.
  PriorityList<int, int> list(488);
  list.InsrtElmnt(&numbers[4], 4, true);
  list.InsrtElmnt(&numbers[2], 2, true);
  list.InsrtElmnt(&numbers[1], 1, true);

  list.RemoveAt(list.begin());
  list.InsrtElmnt(&numbers[3], 3, true);

  EXPECT_EQ(3, list.GetElmntCnt());

  auto it = list.begin();
  ASSERT_NE(it, list.end());
  ASSERT_EQ(*it, 3);
  ++it;
  ASSERT_NE(it, list.end());
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_NE(it, list.end());
  ASSERT_EQ(*it, 1);

  ++it;
  ASSERT_EQ(it, list.end());
}

} // namespace
