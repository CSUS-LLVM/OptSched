#include "opt-sched/Scheduler/lnkd_lst.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

using namespace llvm::opt_sched;

namespace {

TEST(LinkedList, CanBeIteratedOverUsingStandardIteration) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  LinkedList<int> list{};

  for (int &x : numbers) {
    list.InsrtElmnt(&x);
  }

  ASSERT_EQ(numbers.size(), std::distance(list.begin(), list.end()));

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.end(), mismatch.first);
  EXPECT_EQ(list.end(), mismatch.second);
}

TEST(LinkedListFixed, CanBeIteratedOverUsingStandardIteration) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  LinkedList<int> list(20);

  for (int &x : numbers) {
    list.InsrtElmnt(&x);
  }

  ASSERT_EQ(numbers.size(), std::distance(list.begin(), list.end()));

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.end(), mismatch.first);
  EXPECT_EQ(list.end(), mismatch.second);
}

TEST(LinkedList, EmptyStandardIterationYieldsNoIteration) {
  LinkedList<int> list{};

  bool wasTouched = false;
  for (auto &&x : list) {
    (void)x;
    wasTouched = true;
  }

  EXPECT_FALSE(wasTouched);
}

TEST(LinkedListFixed, EmptyStandardIterationYieldsNoIteration) {
  LinkedList<int> list(20);

  bool wasTouched = false;
  for (auto &&x : list) {
    (void)x;
    wasTouched = true;
  }

  EXPECT_FALSE(wasTouched);
}

TEST(LinkedList, SingletonStandardIterationIteratesExactlyOnce) {
  LinkedList<int> list{};
  int n = 42;
  list.InsrtElmnt(&n);

  std::vector<int> result;
  for (auto &&x : list) {
    result.push_back(x);
  }

  EXPECT_EQ(std::vector<int>{n}, result);
}

TEST(LinkedListFixed, SingletonStandardIterationIteratesExactlyOnce) {
  LinkedList<int> list(20);
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

class LinkedListRemoveIndexTest : public testing::TestWithParam<int> {};

TEST_P(LinkedListRemoveIndexTest, EntriesCanBeRemovedByIterator) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> numbers2 = numbers;

  LinkedList<int> list{};

  int index = GetParam();
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

TEST_P(LinkedListRemoveIndexTest,
       FixedListAllocationEntriesCanBeRemovedByIterator) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> numbers2 = numbers;

  LinkedList<int> list(20);

  int index = GetParam();
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

INSTANTIATE_TEST_CASE_P(RemoveSeveralIndices, LinkedListRemoveIndexTest,
                        testing::Values(0, 1, 2, 3, -1), );

TEST(LinkedList, CanAddAfterRemove) {
  std::vector<int> numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> numbers2 = numbers;

  LinkedList<int> list{};

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
  list.InsrtElmnt(&numbers[3]);
  numbers.push_back(numbers[3]);

  ASSERT_EQ(numbers.size(), list.GetElmntCnt());

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.size(), mismatch.first - numbers.begin())
      << "Expected: " << ::testing::PrintToString(numbers);
  EXPECT_EQ(list.GetElmntCnt(), std::distance(list.begin(), mismatch.second))
      << "Actual:   " << ::testing::PrintToString(list);
}

TEST(LinkedListFixed, CanAddAfterRemove) {
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
  list.InsrtElmnt(&numbers[3]);
  numbers.push_back(numbers[3]);

  ASSERT_EQ(numbers.size(), list.GetElmntCnt());

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.size(), mismatch.first - numbers.begin())
      << "Expected: " << ::testing::PrintToString(numbers);
  EXPECT_EQ(list.GetElmntCnt(), std::distance(list.begin(), mismatch.second))
      << "Actual:   " << ::testing::PrintToString(list);
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
    list.InsrtElmnt(&numbers[3]);
    numbers.push_back(numbers[3]);
  }

  ASSERT_EQ(numbers.size(), list.GetElmntCnt());

  auto mismatch = std::mismatch(numbers.begin(), numbers.end(), list.begin());

  EXPECT_EQ(numbers.size(), mismatch.first - numbers.begin())
      << "Expected: " << ::testing::PrintToString(numbers);
  EXPECT_EQ(list.GetElmntCnt(), std::distance(list.begin(), mismatch.second))
      << "Actual:   " << ::testing::PrintToString(list);
}

} // namespace
