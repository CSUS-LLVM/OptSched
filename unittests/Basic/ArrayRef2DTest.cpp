#include "opt-sched/Scheduler/array_ref2d.h"

#include <array>

#include "gtest/gtest.h"

using namespace llvm::opt_sched;

namespace {
TEST(ArrayRef2D, CanAccessElements) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  ArrayRef2D<int> Ref(Arr, 2, 3);
  EXPECT_EQ(1, (Ref[{0, 0}]));
  EXPECT_EQ(2, (Ref[{0, 1}]));
  EXPECT_EQ(3, (Ref[{0, 2}]));
  EXPECT_EQ(4, (Ref[{1, 0}]));
  EXPECT_EQ(5, (Ref[{1, 1}]));
  EXPECT_EQ(6, (Ref[{1, 2}]));
}

TEST(ArrayRef2D, CanGetRowsAndColumns) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  ArrayRef2D<int> Ref(Arr, 2, 3);
  EXPECT_EQ(2, Ref.rows());
  EXPECT_EQ(3, Ref.columns());
}

TEST(ArrayRef2D, AccessReturnsReferenceToElements) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  ArrayRef2D<int> Ref(Arr, 2, 3);
  EXPECT_EQ(&Arr[0], &(Ref[{0, 0}]));
}

TEST(ArrayRef2D, AccessDoesNotAllowChanges) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  ArrayRef2D<int> Ref(Arr, 2, 3);
  static_assert(std::is_same<const int &, decltype(Ref[{0, 0}])>::value, "");
}

TEST(ArrayRef2D, RequiresRectangle) {
  int Arr[] = {
      1, 2, 3, //
      4, 5,
  };

  EXPECT_DEBUG_DEATH(ArrayRef2D<int>(Arr, 2, 3), ".*");
}

TEST(ArrayRef2D, AccessingFailsForOutOfBounds) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  ArrayRef2D<int> Ref(Arr, 2, 3);
  EXPECT_DEBUG_DEATH((Ref[{5, 10}]), ".*");
}

TEST(ArrayRef2D, WorksForEmpty) {
  std::array<int, 0> Arr{};

  ArrayRef2D<int> Ref(Arr, 0, 0);
  EXPECT_EQ(0u, Ref.rows());
  EXPECT_EQ(0u, Ref.columns());
  EXPECT_EQ(0u, Ref.underlyingData().size());
}

TEST(ArrayRef2D, AccessingEmptyRefFails) {
  std::array<int, 0> Arr{};

  ArrayRef2D<int> Ref(Arr, 0, 0);
  EXPECT_DEBUG_DEATH((Ref[{0, 0}]), ".*");
}

TEST(ArrayRef2D, UnderlyingDataIsArrayRef) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  ArrayRef2D<int> Ref(Arr, 2, 3);
  static_assert(
      std::is_same<llvm::ArrayRef<int>, decltype(Ref.underlyingData())>::value,
      "");
}

TEST(MutableArrayRef2D, IsConvertibleToArrayRef2D) {
  static_assert(
      std::is_convertible<MutableArrayRef2D<int>, ArrayRef2D<int>>::value, "");
}

TEST(MutableArrayRef2D, UnderlyingDataIsMutableArrayRef) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  MutableArrayRef2D<int> Ref(Arr, 2, 3);
  static_assert(std::is_same<llvm::MutableArrayRef<int>,
                             decltype(Ref.underlyingData())>::value,
                "");
}

TEST(MutableArrayRef2D, CanMutateViaAccess) {
  int Arr[] = {
      1, 2, 3, //
      4, 5, 6,
  };

  MutableArrayRef2D<int> Ref(Arr, 2, 3);
  Ref[{1, 1}] = -5;
  EXPECT_EQ(-5, (Ref[{1, 1}]));
  EXPECT_EQ(-5, Arr[4]);
}
} // namespace
