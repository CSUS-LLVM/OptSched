/*******************************************************************************
Description:  Defines a generic linked list template class with a number of
              commonly used derivatives like queues, stacks and priority lists.
              Warning: the code within has evolved over many years.
Author:       Ghassan Shobaki
Created:      Oct. 1997
Last Update:  May  2020
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_LNKD_LST_H
#define OPTSCHED_GENERIC_LNKD_LST_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/logger.h"
#include <cstring>

namespace llvm {
namespace opt_sched {

// A container class for the object to be stored in a linked list.
template <class T> struct Entry {
  T *element;

  inline Entry(T *element = NULL, Entry *next = NULL, Entry *prev = NULL)
      : element(element), next(next), prev(prev) {}
  virtual ~Entry() {}
  virtual Entry *GetNext() const { return next; }
  virtual Entry *GetPrev() const { return prev; }
  virtual void SetNext(Entry *e) { next = e; }
  virtual void SetPrev(Entry *e) { prev = e; }

protected:
  Entry *next;
  Entry *prev;
};

// A container class for the object to be stored in a priority list.
template <class T, class K = unsigned long> struct KeyedEntry : Entry<T> {
  K key;

  inline KeyedEntry(T *element = NULL, K key = 0, Entry<T> *next = NULL,
                    Entry<T> *prev = NULL) {
    this->key = key;
    Entry<T>::element = element;
    Entry<T>::next = next;
    Entry<T>::prev = prev;
  }
  virtual ~KeyedEntry() {}

  inline KeyedEntry(Entry<T> const *const entry, K key = 0) {
    this->key = key;
    Entry<T>::element = entry->element;
    Entry<T>::next = entry->GetNext();
    Entry<T>::prev = entry->GetPrev();
  }
  virtual KeyedEntry *GetNext() const { return (KeyedEntry *)Entry<T>::next; }
  virtual KeyedEntry *GetPrev() const { return (KeyedEntry *)Entry<T>::prev; }
  virtual void SetNext(Entry<T> *e) { Entry<T>::next = (Entry<T> *)e; }
  virtual void SetPrev(Entry<T> *e) { Entry<T>::prev = (Entry<T> *)e; }
};

// A generic doubly-linked list container class. If created with a constant
// size, uses an array instead. Tracks a "current" entry similar to an iterator.
template <class T> class LinkedList {
public:
  // Constructs a linked list, by default using a dynamic size.
  LinkedList(int maxSize = INVALID_VALUE);
  // A virtual destructor, to support inheritance.
  virtual ~LinkedList();
  // Deletes all existing entries and resets the list to its initial state.
  virtual void Reset();

  // Appends a new element to the end of the list.
  virtual void InsrtElmnt(T *elmnt);
  // Removes the provided element. The list must be dynamically sized.
  virtual void RmvElmnt(const T *const elmnt);
  // Removes the last element of the list. The list must be dynamically sized.
  virtual void RmvLastElmnt();

  // Returns the number of elements currently in the list.
  virtual int GetElmntCnt() const;
  // Returns the first/top/head element. Does not affect the "current"
  // element.
  virtual T *GetHead() const;
  // Returns the last/bottom/tail element. Does not affect the "current"
  // element.
  virtual T *GetTail() const;

  // Returns the first/top/head element and sets the "current" element to it.
  virtual T *GetFrstElmnt();
  // Returns the last/bottom/tail element and sets the "current" element to
  // it.
  virtual T *GetLastElmnt();
  // Returns the element following the last retrieved one and sets the
  // "current" element to it.
  virtual T *GetNxtElmnt();
  // Returns the element preceding the last retrieved one and sets the
  // "current" element to it.
  virtual T *GetPrevElmnt();
  // Resets the "current" element (iterator) state.
  virtual void ResetIterator();
  // Removes the "current" element from the list.
  virtual void RmvCrntElmnt();

  // Searches for an element in the list. Returns true if it is found.
  virtual bool FindElmnt(const T *const element) const;
  // Searches for an element in the list and records the number of times it.
  // is found in hitCnt. Returns true if the element is found at least once.
  virtual bool FindElmnt(const T *const element, int &hitCnt) const;

protected:
  int maxSize_;
  Entry<T> *allocEntries_;
  int crntAllocIndx_;
  Entry<T> *topEntry_, *bottomEntry_, *rtrvEntry_;
  int elmntCnt_;
  bool itrtrReset_;
  bool wasTopRmvd_;
  bool wasBottomRmvd_;

  // Appends an element to the bottom/end/tail of the list.
  virtual void AppendEntry_(Entry<T> *newEntry);
  // Removes a given entry from the list. If free = true, deletes it via
  // FreeEntry_().
  virtual void RmvEntry_(Entry<T> *entry, bool free = true);
  // Resets all state to default values. Warning: does not free memory!
  virtual void Init_();
  // Deletes an entry object in dynamically-sized lists.
  virtual void FreeEntry_(Entry<T> *entry);
  // Creates a new entry, by allocating memory in dynamically-sized lists or
  // using previously allocated memory in fixed-sized lists.
  virtual Entry<T> *AllocEntry_(T *element);
  // Allocates all entries for a fixed-sized list.
  virtual void AllocEntries_();
};

// A queue class that provides a helper head extraction method.
template <class T> class Queue : public LinkedList<T> {
public:
  Queue(int maxSize = INVALID_VALUE) : LinkedList<T>(maxSize) {}
  // Extracts the head of the list.
  virtual T *ExtractElmnt();
};

// A stack class that provides a helper head extraction method.
template <class T> class Stack : public LinkedList<T> {
public:
  Stack(int maxSize = INVALID_VALUE) : LinkedList<T>(maxSize) {}
  // Extracts the head of the list.
  virtual T *ExtractElmnt();
};

// A priority list (queue) class with a configurable value and key types.
template <class T, class K = unsigned long>
class PriorityList : public LinkedList<T> {
public:
  // Constructs a priority list, by default using a dynamic size.
  inline PriorityList(int maxSize = INVALID_VALUE);
  // Constructs a priority list, by default using a dynamic size.
  ~PriorityList() {
    if (LinkedList<T>::maxSize_ != INVALID_VALUE) {
      delete[] allocKeyEntries_;
    }
  }

  // Insert a new element by automatically finding its place in the list.
  // If allowDplct is false, the element will not be inserted if another
  // element with the same key exists.
  KeyedEntry<T, K> *InsrtElmnt(T *elmnt, K key, bool allowDplct);
  // Disable the version from LinkedList.
  void InsrtElmnt(T *) { Logger::Fatal("Unimplemented."); }
  // Updates an entry's key and moves it to its correct place.
  void BoostEntry(KeyedEntry<T, K> *entry, K newKey);
  // Gets the next element in the list, based on the "current" element.
  // Returns NULL when the end of the list has been reached. If key is
  // provided, it is filled with the key of the retrieved element.
  T *GetNxtPriorityElmnt();
  T *GetNxtPriorityElmnt(K &key);
  // Copies all the data from another list. The existing list must be empty.
  // Also insert the entries into an array if it one is passed.
  void CopyList(PriorityList<T, K> const *const otherLst,
                KeyedEntry<T, unsigned long> **keyedEntries_ = nullptr);

protected:
  KeyedEntry<T, K> *allocKeyEntries_;

  // Creates and returns a keyed entry. For dynamically-sized lists, new
  // memory is allocated. For fixed-size lists, existing memory is used.
  KeyedEntry<T, K> *AllocEntry_(T *elmnt, K key);
  // Disable the version from LinkedList.
  Entry<T> *AllocEntry_(T *) {
    Logger::Fatal("Unimplemented.");
    return NULL;
  }
  // Allocates all the keyed entries in a fixed-size list.
  void AllocEntries_();
  // Inserts entry before next.
  virtual void InsrtEntry_(KeyedEntry<T, K> *entry, KeyedEntry<T, K> *next);
};

template <class T> inline LinkedList<T>::LinkedList(int maxSize) {
  Init_();
  maxSize_ = maxSize;

  if (maxSize_ == INVALID_VALUE) {
    allocEntries_ = NULL;
  } else {
    AllocEntries_();
  }
}

template <class T> LinkedList<T>::~LinkedList() {
  Reset();

  if (maxSize_ != INVALID_VALUE) {
    delete[] allocEntries_;
  }
}

template <class T> inline void LinkedList<T>::Reset() {
  Entry<T> *nextEntry;

  if (maxSize_ == INVALID_VALUE) {
    for (Entry<T> *crntEntry = topEntry_; crntEntry != NULL;
         crntEntry = nextEntry) {
      nextEntry = crntEntry->GetNext();
      FreeEntry_(crntEntry);
    }
  }

  Init_();
}

template <class T> void LinkedList<T>::InsrtElmnt(T *elmnt) {
  Entry<T> *newEntry;

  newEntry = AllocEntry_(elmnt);
  AppendEntry_(newEntry);
}

template <class T> void LinkedList<T>::RmvElmnt(const T *const elmnt) {
  assert(LinkedList<T>::maxSize_ == INVALID_VALUE);

  Entry<T> *crntEntry, *prevEntry = NULL;

  for (crntEntry = topEntry_; crntEntry != NULL;
       prevEntry = crntEntry, crntEntry = crntEntry->GetNext()) {
    if (crntEntry->element == elmnt) {
      // Found.
      if (crntEntry == topEntry_) {
        topEntry_ = crntEntry->GetNext();
      }

      if (crntEntry == bottomEntry_) {
        bottomEntry_ = prevEntry;

        if (bottomEntry_ != NULL) {
          bottomEntry_->SetNext(NULL);
        }
      }

      if (prevEntry != NULL) {
        prevEntry->SetNext(crntEntry->GetNext());
      }

      FreeEntry_(crntEntry);
      elmntCnt_--;
      return;
    }
  }

  Logger::Fatal("Invalid linked list removal.");
}

template <class T> void LinkedList<T>::RmvLastElmnt() {
  assert(maxSize_ == INVALID_VALUE);

  Entry<T> *rmvdEntry = bottomEntry_;
  assert(bottomEntry_ != NULL);
  bottomEntry_ = bottomEntry_->GetPrev();
  assert(elmntCnt_ > 0);
  elmntCnt_--;

  if (elmntCnt_ == 0) {
    assert(bottomEntry_ == NULL);
    topEntry_ = NULL;
  } else {
    assert(bottomEntry_ != NULL);
    bottomEntry_->SetNext(NULL);
  }

  FreeEntry_(rmvdEntry);
}

template <class T> inline int LinkedList<T>::GetElmntCnt() const {
  return elmntCnt_;
}

template <class T> inline T *LinkedList<T>::GetHead() const {
  return topEntry_ == NULL ? NULL : topEntry_->element;
}

template <class T> inline T *LinkedList<T>::GetTail() const {
  return bottomEntry_ == NULL ? NULL : bottomEntry_->element;
}

template <class T> inline T *LinkedList<T>::GetFrstElmnt() {
  wasTopRmvd_ = false;
  wasBottomRmvd_ = false;
  rtrvEntry_ = topEntry_;
  return rtrvEntry_ == NULL ? NULL : rtrvEntry_->element;
}

template <class T> inline T *LinkedList<T>::GetLastElmnt() {
  rtrvEntry_ = bottomEntry_;
  return rtrvEntry_ == NULL ? NULL : rtrvEntry_->element;
}

template <class T> inline T *LinkedList<T>::GetNxtElmnt() {
  if (wasTopRmvd_) {
    rtrvEntry_ = topEntry_;
  } else {
    rtrvEntry_ = rtrvEntry_->GetNext();
  }

  if (wasBottomRmvd_) {
    rtrvEntry_ = NULL;
  }

  wasTopRmvd_ = false;
  wasBottomRmvd_ = false;
  T *elmnt = rtrvEntry_ == NULL ? NULL : rtrvEntry_->element;
  return elmnt;
}

template <class T> inline T *LinkedList<T>::GetPrevElmnt() {
  rtrvEntry_ = rtrvEntry_->GetPrev();
  return rtrvEntry_ == NULL ? NULL : rtrvEntry_->element;
}

template <class T> inline void LinkedList<T>::ResetIterator() {
  itrtrReset_ = true;
  rtrvEntry_ = NULL;
  wasTopRmvd_ = false;
  wasBottomRmvd_ = false;
}

template <class T>
bool LinkedList<T>::FindElmnt(const T *const element, int &hitCnt) const {
  Entry<T> *crntEntry;
  hitCnt = 0;
  for (crntEntry = topEntry_; crntEntry != NULL;
       crntEntry = crntEntry->GetNext()) {
    if (crntEntry->element == element)
      hitCnt++;
  }

  return hitCnt > 0 ? true : false;
}

template <class T> bool LinkedList<T>::FindElmnt(const T *const element) const {
  int hitCnt;
  return FindElmnt(element, hitCnt);
}

template <class T> inline void LinkedList<T>::RmvCrntElmnt() {
  assert(rtrvEntry_ != NULL);
  wasTopRmvd_ = rtrvEntry_ == topEntry_;
  wasBottomRmvd_ = rtrvEntry_ == bottomEntry_;
  Entry<T> *prevEntry = rtrvEntry_->GetPrev();
  RmvEntry_(rtrvEntry_);
  rtrvEntry_ = prevEntry;
}

template <class T> void LinkedList<T>::AppendEntry_(Entry<T> *newEntry) {
  if (bottomEntry_ == NULL) {
    topEntry_ = newEntry;
  } else {
    bottomEntry_->SetNext(newEntry);
  }

  newEntry->SetPrev(bottomEntry_);
  newEntry->SetNext(NULL);
  bottomEntry_ = newEntry;
  elmntCnt_++;
}

template <class T> void LinkedList<T>::RmvEntry_(Entry<T> *entry, bool free) {
  assert(maxSize_ == INVALID_VALUE);
  assert(LinkedList<T>::elmntCnt_ > 0);

  Entry<T> *nextEntry = entry->GetNext();
  Entry<T> *prevEntry = entry->GetPrev();

  // Update the top entry pointer if the entry to insert is the top entry.
  if (prevEntry == NULL) {
    assert(entry == topEntry_);
    topEntry_ = nextEntry;
  } else {
    prevEntry->SetNext(nextEntry);
  }

  // Update the bottom entry pointer if the entry to insert is the bottom entry.
  if (nextEntry == NULL) {
    assert(entry == bottomEntry_);
    bottomEntry_ = prevEntry;
  } else {
    nextEntry->SetPrev(prevEntry);
  }

  if (free)
    FreeEntry_(entry);

  elmntCnt_--;
}

template <class T> void LinkedList<T>::FreeEntry_(Entry<T> *entry) {
  if (maxSize_ == INVALID_VALUE) {
    delete entry;
  } else {
    assert(crntAllocIndx_ >= 1);
    crntAllocIndx_--;
  }
}

template <class T> inline void LinkedList<T>::Init_() {
  topEntry_ = bottomEntry_ = rtrvEntry_ = NULL;
  elmntCnt_ = 0;
  itrtrReset_ = true;
  wasTopRmvd_ = false;
  wasBottomRmvd_ = false;
  crntAllocIndx_ = 0;
}

template <class T> Entry<T> *LinkedList<T>::AllocEntry_(T *element) {
  Entry<T> *entry;

  if (maxSize_ == INVALID_VALUE) {
    entry = new Entry<T>();
  } else {
    assert(crntAllocIndx_ < maxSize_);
    entry = allocEntries_ + crntAllocIndx_;
    crntAllocIndx_++;
  }

  entry->element = element;
  return entry;
}

template <class T> void LinkedList<T>::AllocEntries_() {
  assert(maxSize_ != INVALID_VALUE);
  allocEntries_ = new Entry<T>[maxSize_];
  crntAllocIndx_ = 0;
}

template <class T> inline T *Queue<T>::ExtractElmnt() {
  // assert(LinkedList<T>::maxSize_ == INVALID_VALUE);
  if (LinkedList<T>::topEntry_ == NULL)
    return NULL;

  Entry<T> *headEntry = LinkedList<T>::topEntry_;
  T *headElmnt = headEntry->element;

  if (LinkedList<T>::bottomEntry_ == LinkedList<T>::topEntry_) {
    LinkedList<T>::bottomEntry_ = NULL;
  }

  LinkedList<T>::topEntry_ = headEntry->GetNext();

  if (LinkedList<T>::topEntry_ != NULL) {
    LinkedList<T>::topEntry_->SetPrev(NULL);
  }

  LinkedList<T>::elmntCnt_--;
  FreeEntry_(headEntry);
  return headElmnt;
}

template <class T> inline T *Stack<T>::ExtractElmnt() {
  // assert(LinkedList<T>::maxSize_ == INVALID_VALUE);
  if (LinkedList<T>::bottomEntry_ == NULL)
    return NULL;

  Entry<T> *trgtEntry = LinkedList<T>::bottomEntry_;
  T *trgtElmnt = trgtEntry->element;

  if (LinkedList<T>::bottomEntry_ == LinkedList<T>::topEntry_) {
    LinkedList<T>::topEntry_ = NULL;
  }

  LinkedList<T>::bottomEntry_ = LinkedList<T>::bottomEntry_->GetPrev();

  if (LinkedList<T>::bottomEntry_ != NULL) {
    LinkedList<T>::bottomEntry_->SetNext(NULL);
  }

  LinkedList<T>::elmntCnt_--;
  LinkedList<T>::FreeEntry_(trgtEntry);
  return trgtElmnt;
}

template <class T, class K>
PriorityList<T, K>::PriorityList(int maxSize) : LinkedList<T>(maxSize) {
  if (LinkedList<T>::maxSize_ != INVALID_VALUE) {
    delete[] LinkedList<T>::allocEntries_;
    LinkedList<T>::allocEntries_ = new Entry<T>[0];
    AllocEntries_();
  } else {
    allocKeyEntries_ = NULL;
  }
}

template <class T, class K>
KeyedEntry<T, K> *PriorityList<T, K>::InsrtElmnt(T *elmnt, K key,
                                                 bool allowDplct) {
  KeyedEntry<T, K> *crnt;
  KeyedEntry<T, K> *next = NULL;
  bool foundDplct = false;

  for (crnt = (KeyedEntry<T, K> *)LinkedList<T>::bottomEntry_; crnt != NULL;
       crnt = crnt->GetPrev()) {
    if (crnt->key >= key) {
      foundDplct = (crnt->key == key);
      break;
    }
    next = crnt;
  }

  if (!allowDplct && foundDplct)
    return crnt;

  KeyedEntry<T, K> *newEntry = AllocEntry_(elmnt, key);
  InsrtEntry_(newEntry, next);
  LinkedList<T>::itrtrReset_ = true;
  return newEntry;
}

template <class T, class K>
inline T *PriorityList<T, K>::GetNxtPriorityElmnt() {
  assert(LinkedList<T>::itrtrReset_ || LinkedList<T>::rtrvEntry_ != NULL);

  if (LinkedList<T>::itrtrReset_) {
    LinkedList<T>::rtrvEntry_ = LinkedList<T>::topEntry_;
  } else {
    LinkedList<T>::rtrvEntry_ = LinkedList<T>::rtrvEntry_->GetNext();
  }

  LinkedList<T>::itrtrReset_ = false;

  if (LinkedList<T>::rtrvEntry_ == NULL) {
    return NULL;
  } else {
    return LinkedList<T>::rtrvEntry_->element;
  }
}

template <class T, class K>
inline T *PriorityList<T, K>::GetNxtPriorityElmnt(K &key) {
  assert(LinkedList<T>::itrtrReset_ || LinkedList<T>::rtrvEntry_ != NULL);
  if (LinkedList<T>::itrtrReset_) {
    LinkedList<T>::rtrvEntry_ = LinkedList<T>::topEntry_;
  } else {
    LinkedList<T>::rtrvEntry_ = LinkedList<T>::rtrvEntry_->GetNext();
  }

  LinkedList<T>::itrtrReset_ = false;

  if (LinkedList<T>::rtrvEntry_ == NULL) {
    return NULL;
  } else {
    key = ((KeyedEntry<T, K> *)LinkedList<T>::rtrvEntry_)->key;
    return LinkedList<T>::rtrvEntry_->element;
  }
}

//(Vlad) added functionality to decrease priority
// used for decreasing priority of clusterable instrs
// when leaving a cluster
template <class T, class K>
void PriorityList<T, K>::BoostEntry(KeyedEntry<T, K> *entry, K newKey) {
  KeyedEntry<T, K> *crnt;
  KeyedEntry<T, K> *next = entry->GetNext();
  KeyedEntry<T, K> *prev = entry->GetPrev();

  assert(LinkedList<T>::topEntry_ != NULL);

  if (entry->key < newKey) // behave normally
  {
    entry->key = newKey;

    // If it is already at the top, or its previous still has a larger key,
    // then the entry is already in place and no boosting is needed
    if (entry == LinkedList<T>::topEntry_ || prev->key >= newKey)
      return;

    prev = NULL;

    for (crnt = entry->GetPrev(); crnt != NULL; crnt = crnt->GetPrev()) {
      if (crnt->key >= newKey) {
        assert(crnt != entry);
        assert(crnt != entry->GetPrev());
        prev = crnt;
        break;
      }
    }

    if (prev == NULL) {
      next = (KeyedEntry<T, K> *)LinkedList<T>::topEntry_;
    } else {
      next = prev->GetNext();
      assert(next != NULL);
    }

    assert(next != entry->GetNext());
    LinkedList<T>::RmvEntry_(entry, false);
    InsrtEntry_(entry, next);
  } else // move entry down on priority list
  {
    entry->key = newKey;

    // if it is at the bottom or next entry still has a smaller key,
    // then the entry is already in place
    if (entry == LinkedList<T>::bottomEntry_ || next->key <= newKey)
      return;

    for (crnt = entry->GetNext(); crnt != NULL; crnt = crnt->GetNext()) {
      if (crnt->key <= newKey) {
        next = crnt;
        break;
      }
    }

    LinkedList<T>::RmvEntry_(entry, false);
    InsrtEntry_(entry, next);
  }

  this->itrtrReset_ = true;
}

template <class T, class K>
void PriorityList<T, K>::CopyList(
    PriorityList<T, K> const *const otherLst,
    KeyedEntry<T, unsigned long> **keyedEntries_) {
  assert(LinkedList<T>::elmntCnt_ == 0);

  for (KeyedEntry<T, K> *entry = (KeyedEntry<T, K> *)otherLst->topEntry_;
       entry != NULL; entry = entry->GetNext()) {
    T *elmnt = entry->element;
    K key = entry->key;
    KeyedEntry<T, K> *newEntry = AllocEntry_(elmnt, key);
    LinkedList<T>::AppendEntry_(newEntry);
    if (keyedEntries_)
      keyedEntries_[entry->element->GetNum()] = newEntry;

    if (entry == otherLst->rtrvEntry_) {
      LinkedList<T>::rtrvEntry_ = newEntry;
    }
  }

  LinkedList<T>::itrtrReset_ = otherLst->itrtrReset_;
}

template <class T, class K>
KeyedEntry<T, K> *PriorityList<T, K>::AllocEntry_(T *element, K key) {
  KeyedEntry<T, K> *newEntry;

  if (LinkedList<T>::maxSize_ == INVALID_VALUE) {
    newEntry = new KeyedEntry<T, K>(element, key);
  } else {
    assert(LinkedList<T>::crntAllocIndx_ < LinkedList<T>::maxSize_);
    newEntry = allocKeyEntries_ + LinkedList<T>::crntAllocIndx_;
    newEntry->element = element;
    newEntry->key = key;
    LinkedList<T>::crntAllocIndx_++;
  }

  return newEntry;
}

template <class T, class K> void PriorityList<T, K>::AllocEntries_() {
  allocKeyEntries_ = new KeyedEntry<T, K>[LinkedList<T>::maxSize_];
  LinkedList<T>::crntAllocIndx_ = 0;
}

template <class T, class K>
void PriorityList<T, K>::InsrtEntry_(KeyedEntry<T, K> *entry,
                                     KeyedEntry<T, K> *next) {
  KeyedEntry<T, K> *prev;
  if (next == NULL) {
    prev = (KeyedEntry<T, K> *)LinkedList<T>::bottomEntry_;
  } else {
    prev = (KeyedEntry<T, K> *)next->GetPrev();
  }

  // Update the top entry pointer if the entry to insert is the top entry.
  if (prev == NULL) {
    LinkedList<T>::topEntry_ = entry;
  } else {
    prev->SetNext(entry);
  }
  // Update the bottom entry pointer if the entry to insert is the bottom entry.
  if (next == NULL) {
    LinkedList<T>::bottomEntry_ = entry;
  } else {
    next->SetPrev(entry);
  }

  entry->SetNext(next);
  entry->SetPrev(prev);
  LinkedList<T>::elmntCnt_++;
}

} // namespace opt_sched
} // namespace llvm

#endif
