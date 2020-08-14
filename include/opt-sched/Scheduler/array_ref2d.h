#ifndef OPTSCHED_ARRAY_REF_2D_H
#define OPTSCHED_ARRAY_REF_2D_H

#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <cstddef>

namespace llvm {
namespace opt_sched {

/**
 * \brief Provides a 2D view over a single allocation
 *
 * \details 2D arrays are best implemented by using a single allocation, then
 * computing the index into this single allocation based on the 2D location we
 * are trying to access. This type abstracts away that work, doing it for you.
 *
 * \see MutableArrayRef2D
 */
template <typename T> class ArrayRef2D {
public:
  /**
   * \brief Constructs an ArrayRef2D with the specified dimensions.
   * \param Ref     Must have a size precisely Rows * Columns.
   * \param Rows    The number of rows in this 2D matrix.
   * \param Columns The number of columns in this 2D matrix.
   */
  explicit ArrayRef2D(llvm::ArrayRef<T> Ref, size_t Rows, size_t Columns)
      : Ref(Ref), Rows(Rows), Columns(Columns) {
    assert(Rows * Columns == Ref.size());
  }

  size_t rows() const { return Rows; }
  size_t columns() const { return Columns; }

  /**
   * \brief Access an element at the specified row and columns. `[{row, col}]`
   * \detail
   * A C-style array `int arr[10][20]` is a single contiguous block of memory.
   * It would be accessed as `arr[row][col]`.
   * For ArrayRef2D, a single block of memory such as
   * `int* arr = new int[10 * 20]` is accessed as `ref[{row, col}]`.
   *
   * If you want to do x, y indexing, prefer `ref[{y, x}]` over `ref[{x, y}]`.
   * When accessed in this way, consecutive x values are placed together in
   * memory, which is usually what is expected.
   */
  const T &operator[](size_t(&&RowCol)[2]) const {
    return Ref[computeIndex(RowCol[0], RowCol[1], Rows, Columns)];
  }

  /**
   * \brief Recovers the underlying ArrayRef.
   */
  llvm::ArrayRef<T> underlyingData() const { return Ref; }

private:
  llvm::ArrayRef<T> Ref;
  size_t Rows;
  size_t Columns;

  static size_t computeIndex(size_t row, size_t col, size_t Rows,
                             size_t Columns) {
    assert(row < Rows && "Invalid row");
    assert(col < Columns && "Invalid column");
    size_t index = row * Columns + col;
    assert(index < Rows * Columns); // Should be redundant with prior asserts.
    return index;
  }
};

/**
 * \brief An ArrayRef2D which allows mutation.
 * \note Inherits from ArrayRef2D, allowing slicing from this type to
 * ArrayRef2D in the same manner as LLVM's ArrayRef and MutableArrayRef
 *
 * \see ArrayRef2D
 */
template <typename T> class MutableArrayRef2D : public ArrayRef2D<T> {
public:
  explicit MutableArrayRef2D(llvm::MutableArrayRef<T> Ref, size_t Rows,
                             size_t Columns)
      : ArrayRef2D<T>(Ref, Rows, Columns) {}

  /**
   * \brief Access an element at the specified row and columns. `[{row, col}]`
   * \returns a _mutable_ reference to the element at the specified location.
   */
  T &operator[](size_t(&&RowCol)[2]) const {
    ArrayRef2D<T> cref = *this;
    return const_cast<T &>(cref[{RowCol[0], RowCol[1]}]);
  }

  /**
   * \brief Recovers the underlying MutableArrayRef.
   */
  llvm::MutableArrayRef<T> underlyingData() const {
    return static_cast<const llvm::MutableArrayRef<T> &>(
        ArrayRef2D<T>::underlyingData());
  }
};
} // namespace opt_sched
} // namespace llvm

#endif
