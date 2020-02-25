/*******************************************************************************
Description:  Implements application-level memory management used avoid the OS
              overhead in performance-critical sections of the code.
Author:       Ghassan Shobaki
Created:      Mar. 2003
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_MEM_MNGR_H
#define OPTSCHED_GENERIC_MEM_MNGR_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"
#include "opt-sched/Scheduler/logger.h"
#include <cstring>

namespace llvm {
namespace opt_sched {

template <class T> class MemAlloc {
public:
  // Allocates a new memory block of an initial size with an optional maximum
  // size. If no maximum size is specified, the memory is allocated
  // dynamically. The size is in the number of objects of type T.
  inline MemAlloc(int blockSize, int maxSize = INVALID_VALUE);
  // Deallocates the memory.
  inline ~MemAlloc();
  // Marks all allocated memory as unused (and available for reuse).
  inline void Reset();
  // Returns an allocated object.
  inline T *GetObject();
  // Frees an object and recycles it for future use.
  inline void FreeObject(T *obj);

protected:
  // The number of objects in each memory block allocated.
  int blockSize_;
  // The maximum number of objects to keep allocated.
  int maxSize_;
  // A buffer including an allocated block of objects
  T *currentBlock_;
  // The index of the next available object in the current block.
  int currentIndex_;
  // A linked list of previously allocated and fully used blocks.
  LinkedList<T> allocatedBlocks_;
  // A linked list of free objects available for reuse.
  Stack<T> availableObjects_;
  // Whether an of the already allocated blocks are still unused.
  bool allocatedBlocksAvailable_;

  // Makes sure currentBlock_ points to an unused block, allocating a new one
  // if needed.
  inline void GetNewBlock_();
  // Allocates a new block.
  inline void AllocNewBlock_();
  // Returns a pointer to an array of count unused objects.
  inline T *GetObjects_(int count);
};

template <class T> class ArrayMemAlloc : public MemAlloc<T> {
public:
  // Allocates a memory block that contains arraysPerBlock arrays, each
  // containing arraySize elements of type T.
  inline ArrayMemAlloc(int arraysPerBlock, int arraySize)
      : MemAlloc<T>(arraysPerBlock * arraySize) {
    arraySize_ = arraySize;
  }
  // Returns an allocated array of objects.
  inline T *GetArray() { return MemAlloc<T>::GetObjects_(arraySize_); }
  // Frees an array of objects and recycle it for future use.
  inline void FreeArray(T *array) { FreeObject(array); }

protected:
  // The size of each array.
  int arraySize_;
};

template <class T>
inline MemAlloc<T>::MemAlloc(int blockSize, int maxSize)
    : availableObjects_(maxSize) {
  assert(maxSize == INVALID_VALUE || blockSize <= maxSize);
  blockSize_ = blockSize;
  maxSize_ = maxSize;
  currentIndex_ = 0;
  currentBlock_ = NULL;
  allocatedBlocksAvailable_ = false;
  GetNewBlock_();
}

template <class T> inline MemAlloc<T>::~MemAlloc() {
  for (T *blk = allocatedBlocks_.GetFrstElmnt(); blk != NULL;
       blk = allocatedBlocks_.GetNxtElmnt()) {
    delete[] blk;
  }
}

template <class T> inline void MemAlloc<T>::Reset() {
  assert(allocatedBlocks_.GetElmntCnt() >= 1);
  currentBlock_ = allocatedBlocks_.GetFrstElmnt();
  currentIndex_ = 0;
  availableObjects_.Reset();
  allocatedBlocksAvailable_ = true;
}

template <class T> inline void MemAlloc<T>::GetNewBlock_() {
  currentBlock_ = NULL;

  if (allocatedBlocksAvailable_) {
    currentBlock_ = allocatedBlocks_.GetNxtElmnt();
    currentIndex_ = 0;
  }

  if (currentBlock_ == NULL) {
    allocatedBlocksAvailable_ = false;
    AllocNewBlock_();
  }
}

template <class T> inline void MemAlloc<T>::AllocNewBlock_() {
  T *blk = new T[blockSize_];
  allocatedBlocks_.InsrtElmnt(blk);
  currentIndex_ = 0;
  currentBlock_ = blk;
}

template <class T> inline T *MemAlloc<T>::GetObjects_(int count) {
  T *obj = availableObjects_.ExtractElmnt();

  if (obj == NULL) {
    // If there are no recycled objects available for reuse.
    assert(currentIndex_ <= blockSize_);

    if (currentIndex_ == blockSize_) {
      // If the current block is all used up.
      assert(maxSize_ == INVALID_VALUE);
      GetNewBlock_();
      assert(currentIndex_ == 0);
    }

    obj = currentBlock_ + currentIndex_;
    currentIndex_ += count;
  }

  assert(obj != NULL);
  return obj;
}

template <class T> inline T *MemAlloc<T>::GetObject() { return GetObjects_(1); }

template <class T> inline void MemAlloc<T>::FreeObject(T *obj) {
  availableObjects_.InsrtElmnt(obj);
}

} // namespace opt_sched
} // namespace llvm

#endif
