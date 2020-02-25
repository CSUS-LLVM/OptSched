/*******************************************************************************
Description:  Defines a generic hash table template class.
Author:       Ghassan Shobaki
Created:      Oct. 1997
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_HASH_TABLE_H
#define OPTSCHED_GENERIC_HASH_TABLE_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/mem_mngr.h"
#include <cstring>
#include <limits>

namespace llvm {
namespace opt_sched {

typedef unsigned int UDT_HASHVAL;
typedef unsigned long UDT_HASHKEY;
typedef unsigned int UDT_HASHTBL_CPCTY;

const int DFLT_HASHTBL_SIZE = 4048;
const int MAX_HASHBITS = 22;
const UDT_HASHTBL_CPCTY DFLT_HASHTBL_CPCTY =
    std::numeric_limits<UDT_HASHTBL_CPCTY>::max();

// This is a container class for the class to be stored into the hash table.
// It is used to implement a linked list of objects for each hashValue.
template <class T> class HashTblEntry {
public:
  HashTblEntry(T *elmnt = NULL, UDT_HASHVAL hashVal = 0);
  virtual ~HashTblEntry() {}
  virtual void Clean() {}

  HashTblEntry *GetNxt() const;
  void SetNxt(HashTblEntry *newEntry);
  HashTblEntry *GetPrev() const;
  void SetPrev(HashTblEntry *newEntry);

  T *GetElmnt() const;
  virtual UDT_HASHVAL GetHashVal() = 0;

protected:
  T *elmnt_;
  HashTblEntry *nxt_;
  HashTblEntry *prev_;
  // The hash value.
  UDT_HASHVAL hashVal_;

  void Init_();
  void Construct_(T *elmnt, UDT_HASHVAL hashVal);
};

// A hash table entry when the key is a binary number of a fixed length
template <class T> class BinHashTblEntry : public HashTblEntry<T> {
public:
  BinHashTblEntry(UDT_HASHKEY key, T *_elmnt, UDT_HASHVAL hashVal);
  BinHashTblEntry();
  ~BinHashTblEntry() {}
  void Construct(UDT_HASHKEY key, T *_elmnt, UDT_HASHVAL hashVal);
  void Clean() {}

  UDT_HASHKEY GetKey();
  UDT_HASHVAL GetHashVal();

private:
  UDT_HASHKEY key_; // A binary key value

  void Init_();
};

// A hash table entry when the key is a string of an arbitrary length
template <class T> class StrHashTblEntry : public HashTblEntry<T> {
public:
  StrHashTblEntry(const char *name, T *_elmnt, UDT_HASHVAL hashVal,
                  UDT_HASHTBL_CPCTY indx = 0);
  ~StrHashTblEntry();
  void Construct(const char *name, T *elmnt, UDT_HASHVAL hashVal,
                 UDT_HASHTBL_CPCTY indx);
  void Clean();

  T *GetElmnt(const char *srchName);
  T *GetElmnt(const char *srchName, UDT_HASHTBL_CPCTY &indx);
  bool IsThis(const char *srchName);
  UDT_HASHTBL_CPCTY GetIndx();
  UDT_HASHVAL GetHashVal();

private:
  char *name_; // an ASCII string name for the entry
  UDT_HASHTBL_CPCTY indx_;

  void Init_();
};

template <class T> class HashTable {
public:
  HashTable(UDT_HASHVAL size = DFLT_HASHTBL_SIZE,
            UDT_HASHTBL_CPCTY maxEntryCnt = DFLT_HASHTBL_CPCTY);
  virtual ~HashTable();
  bool IsConstructed();

  void RemoveEntry(HashTblEntry<T> *entry, bool del = false,
                   MemAlloc<HashTblEntry<T>> *entryAlctr = NULL);

  UDT_HASHTBL_CPCTY GetEntryCnt() { return entryCnt_; }
  UDT_HASHTBL_CPCTY GetPpultdBktCnt() { return ppultdBktCnt_; }
  UDT_HASHTBL_CPCTY GetMaxListSize() { return maxListSize_; }

  // Clear the table by deleting all entries, and if (del is set to true),
  // delete the element themselves as well.
  // After this call the table will be empty
  virtual void Clear(bool del, MemAlloc<BinHashTblEntry<T>> *entryAlctr = NULL);

  void GetFullList(LinkedList<T> *lst);

protected:
  UDT_HASHVAL tblSize_;
  UDT_HASHTBL_CPCTY entryCnt_;
  UDT_HASHTBL_CPCTY maxEntryCnt_;
  UDT_HASHTBL_CPCTY ppultdBktCnt_;
  UDT_HASHTBL_CPCTY maxListSize_;

  // array of heads of linked lists
  HashTblEntry<T> **topEntry_;
  // array of last-linked entries
  HashTblEntry<T> **lastEntry_;
  // array of linked-list sizes
  UDT_HASHTBL_CPCTY *entryCnts_;

  UDT_HASHVAL maxHash_;
  bool isCnstrctd_;
  bool isExtrnlAlctr_;

  // Search linearly for the maximum non-empty hash value
  void FindNewMaxHash_();

  // Given a hash value, search linearly for the next non-empty hash value
  UDT_HASHVAL FindNextHash_(UDT_HASHVAL crntHash);

  void AddNewEntry_(HashTblEntry<T> *newEntry, UDT_HASHVAL hashVal);

  void Init_();
};

template <class T> class BinHashTable : public HashTable<T> {
public:
  BinHashTable(int16_t keyBitCnt, int16_t hashBitCnt, bool isExtrnlAlctr,
               UDT_HASHTBL_CPCTY maxEntryCnt = DFLT_HASHTBL_CPCTY);
  ~BinHashTable() {}

  void Clear(bool del, MemAlloc<BinHashTblEntry<T>> *entryAlctr = NULL);

  HashTblEntry<T> *InsertElement(UDT_HASHKEY key, T *elmnt,
                                 MemAlloc<BinHashTblEntry<T>> *entryAlctr);

  UDT_HASHVAL HashKey(UDT_HASHKEY key);
  UDT_HASHTBL_CPCTY GetListSize(const UDT_HASHKEY key);

  T *GetFirstMatch(const UDT_HASHKEY key, bool skipCollision = true);
  // If the last call to GetFirstMatch() or GetNextMatch() has returned NULL,
  // then GetNextMatch() should not be called again.
  T *GetNextMatch(bool skipCollision = true);
  T *GetLastMatch(const UDT_HASHKEY key, bool skipCollision = true);
  // If the last call to GetLastMatch() or GetPrevMatch() has returned NULL,
  // then GetPrevMatch() should not be called again.
  T *GetPrevMatch(bool skipCollision = true);
  //----------------------------------------------------------------------
  // The following pair of methods work only under perfect hashing
  //,i.e., when hashBitCnt_=keyBitCnt_

  // Get the entry with the maximum key value
  HashTblEntry<T> *GetMaxEntry();

  // Given an entry, get the entry with next higher priority
  // By calling this method iteratively, entries can be retrieved one by one
  // in priority order
  HashTblEntry<T> *GetNextEntry(HashTblEntry<T> *);
  //----------------------------------------------------------------------

  // Extract the element with the maximum key value
  T *ExtractMax();

  // Undo the last ExtractMax
  void RestoreLastMax();

  void DeleteEntries(UDT_HASHKEY key, LinkedList<T> *elmntLst);
  bool DeleteEntry(UDT_HASHKEY key, T *elmnt,
                   MemAlloc<HashTblEntry<T>> *entryAlctr = NULL);
  bool DeleteEntry(UDT_HASHVAL hashVal, T *elmnt,
                   MemAlloc<HashTblEntry<T>> *entryAlctr = NULL);

  // Reinsert a previously removed entry into the exact original location
  void ReInsertEntry(HashTblEntry<T> *entry);

private:
  // The size of the key in bits
  uint16_t keyBitCnt_;

  // The number of MS bits to be used as a hash value indexing into the table,
  // and thus detremining the table size
  uint16_t hashBitCnt_;

  UDT_HASHKEY maxKey_;

  UDT_HASHKEY keyMask_;

  // The number of least sig. bits to discard from the key (by right shifting)
  // in order to form the hash value that will be used as an index.
  int16_t hashRShft_;

  // These two fields save information about the last max. element that
  // has been extracted, so that it can be restored to undo the extraction
  UDT_HASHVAL lastMaxHash_;
  HashTblEntry<T> *lastMaxEntry_;

  // A pointer to the current entry to be used by the search-by-key iterators
  HashTblEntry<T> *srchPtr_;
  // The key value currently being searched
  UDT_HASHKEY srchKey_;

  void Init_();
  void CmputConsts_();
  void FindNextMatch_();
  void FindPrevMatch_();
  void UpdtLastMax_(HashTblEntry<T> *maxEntry, UDT_HASHVAL maxHash);
};

template <class T> class StrHashTable : public HashTable<T> {
public:
  StrHashTable(UDT_HASHVAL size = DFLT_HASHTBL_SIZE, bool useIndx = false,
               UDT_HASHTBL_CPCTY maxEntryCnt = DFLT_HASHTBL_CPCTY);
  ~StrHashTable();

  FUNC_RESULT InsertElement(const char *name, T *elmnt);

  T *FindElement(const char *srchName);
  T *GetElement(const UDT_HASHTBL_CPCTY indx);

private:
  bool useIndx_;
  StrHashTblEntry<T> **indxdTbl_; // array indexed by entry code

  // For String-Based Hash Table
  // A very simple hashing function
  UDT_HASHVAL HashString_(const char *string);
};

template <class T>
inline HashTblEntry<T>::HashTblEntry(T *elmnt, UDT_HASHVAL hashVal) {
  Construct_(elmnt, hashVal);
}

template <class T> inline void HashTblEntry<T>::Init_() {
  nxt_ = NULL;
  prev_ = NULL;
}

template <class T>
inline void HashTblEntry<T>::Construct_(T *elmnt, UDT_HASHVAL hashVal) {
  Init_();
  elmnt_ = elmnt;
  hashVal_ = hashVal;
}

template <class T> HashTblEntry<T> *HashTblEntry<T>::GetNxt() const {
  return nxt_;
}

template <class T> inline void HashTblEntry<T>::SetNxt(HashTblEntry *newEntry) {
  nxt_ = newEntry;
}

template <class T> HashTblEntry<T> *HashTblEntry<T>::GetPrev() const {
  return prev_;
}

template <class T>
inline void HashTblEntry<T>::SetPrev(HashTblEntry *newEntry) {
  prev_ = newEntry;
}

template <class T> inline T *HashTblEntry<T>::GetElmnt() const {
  return elmnt_;
}

template <class T> inline BinHashTblEntry<T>::BinHashTblEntry() { Init_(); }

template <class T>
inline BinHashTblEntry<T>::BinHashTblEntry(UDT_HASHKEY key, T *elmnt,
                                           UDT_HASHVAL hashVal) {
  Construct(key, elmnt, hashVal);
}

template <class T> inline void BinHashTblEntry<T>::Init_() {
  HashTblEntry<T>::Init_();
  key_ = 0;
}

template <class T>
inline void BinHashTblEntry<T>::Construct(UDT_HASHKEY key, T *elmnt,
                                          UDT_HASHVAL hashVal) {
  Init_();
  HashTblEntry<T>::Construct_(elmnt, hashVal);
  key_ = key;
}

template <class T> inline UDT_HASHKEY BinHashTblEntry<T>::GetKey() {
  return key_;
}

template <class T> inline UDT_HASHVAL BinHashTblEntry<T>::GetHashVal() {
  return HashTblEntry<T>::hashVal_;
}

template <class T>
inline StrHashTblEntry<T>::StrHashTblEntry(const char *name, T *elmnt,
                                           UDT_HASHVAL hashVal,
                                           UDT_HASHTBL_CPCTY indx) {
  Construct(name, elmnt, hashVal, indx);
}

template <class T> inline StrHashTblEntry<T>::~StrHashTblEntry() {
  if (name_ != NULL)
    delete[] name_;
}

template <class T> inline void StrHashTblEntry<T>::Init_() {
  name_ = NULL;
  indx_ = 0;
}

template <class T>
inline void StrHashTblEntry<T>::Construct(const char *name, T *elmnt,
                                          UDT_HASHVAL hashVal,
                                          UDT_HASHTBL_CPCTY indx) {
  Init_();
  HashTblEntry<T>::Construct_(elmnt, hashVal);
  name_ = new char[strlen(name) + 1];

  strcpy(name_, name);
  indx_ = indx;
}

template <class T> inline void StrHashTblEntry<T>::Clean() {
  if (name_ != NULL)
    delete[] name_;
}

template <class T> inline UDT_HASHVAL StrHashTblEntry<T>::GetHashVal() {
  return HashTblEntry<T>::hashVal_;
}

template <class T>
inline T *StrHashTblEntry<T>::GetElmnt(const char *srchName) {
  return strcmp(srchName, name_) == 0 ? this->elmnt_ : NULL;
}

template <class T>
inline T *StrHashTblEntry<T>::GetElmnt(const char *srchName,
                                       UDT_HASHTBL_CPCTY &indx) {
  indx = indx_;
  return GetElmnt(srchName);
}

template <class T>
inline bool StrHashTblEntry<T>::IsThis(const char *srchName) {
  return strcmp(srchName, name_) == 0;
}

template <class T> inline UDT_HASHTBL_CPCTY StrHashTblEntry<T>::GetIndx() {
  return indx_;
}

template <class T>
HashTable<T>::HashTable(UDT_HASHVAL size, UDT_HASHTBL_CPCTY maxEntryCnt) {
  isCnstrctd_ = false;
  topEntry_ = NULL;
  lastEntry_ = NULL;
  entryCnts_ = NULL;
  tblSize_ = size;
  maxEntryCnt_ = maxEntryCnt;

  topEntry_ = new HashTblEntry<T> *[tblSize_];
  lastEntry_ = new HashTblEntry<T> *[tblSize_];
  entryCnts_ = new UDT_HASHTBL_CPCTY[tblSize_];

  UDT_HASHVAL i;

  for (i = 0; i < tblSize_; i++) {
    topEntry_[i] = NULL;
    lastEntry_[i] = NULL;
    entryCnts_[i] = 0;
  }

  entryCnt_ = 0;
  ppultdBktCnt_ = 0;
  maxListSize_ = 0;
  maxHash_ = 0;
  isExtrnlAlctr_ = false;
  isCnstrctd_ = true;
}

template <class T> inline bool HashTable<T>::IsConstructed() {
  return isCnstrctd_;
}

template <class T>
void HashTable<T>::Clear(bool del, MemAlloc<BinHashTblEntry<T>> *entryAlctr) {
  UDT_HASHVAL i;
  HashTblEntry<T> *crntEntry;
  HashTblEntry<T> *nxtEntry;
  assert(isCnstrctd_);

  if (entryCnt_ == 0) {
    return;
  }

  for (i = 0; i <= maxHash_; i++) {
    for (crntEntry = topEntry_[i]; crntEntry != NULL; crntEntry = nxtEntry) {
      nxtEntry = crntEntry->GetNxt();

      if (del) {
        delete crntEntry->GetElmnt();
      }

      if (isExtrnlAlctr_) {
        assert(entryAlctr != NULL);
        crntEntry->Clean();
        // Under the assumption that the entire allocator will be freed right
        // after this, we do not need to free this object and put it in a
        // potentially huge linked list. Actually, that was found to cause a
        // serious memory over-allocation problem. [GOS 3.25.03]
        // entryAlctr->FreeObject((BinHashTblEntry<T>*)crntEntry);
      } else {
        delete crntEntry;
      }
    }

    topEntry_[i] = NULL;
    lastEntry_[i] = NULL;
    entryCnts_[i] = 0;
  }

  entryCnt_ = 0;
  ppultdBktCnt_ = 0;
  maxListSize_ = 0;
  maxHash_ = 0;
}

template <class T> HashTable<T>::~HashTable() {
  if (isCnstrctd_)
    Clear(false);
  if (topEntry_)
    delete[] topEntry_;
  if (lastEntry_)
    delete[] lastEntry_;
  if (entryCnts_)
    delete[] entryCnts_;
}

template <class T>
void HashTable<T>::AddNewEntry_(HashTblEntry<T> *newEntry,
                                UDT_HASHVAL hashVal) {
  assert(hashVal < tblSize_);
  newEntry->SetPrev(lastEntry_[hashVal]);

  if (lastEntry_[hashVal] == NULL) {
    topEntry_[hashVal] = newEntry;
  } else {
    lastEntry_[hashVal]->SetNxt(newEntry);
  }

  lastEntry_[hashVal] = newEntry;
  entryCnts_[hashVal]++;

  if (entryCnts_[hashVal] > maxListSize_) {
    maxListSize_ = entryCnts_[hashVal];
  }

  if (entryCnts_[hashVal] == 1) {
    ppultdBktCnt_++;
  }

  entryCnt_++;
}

template <class T> void HashTable<T>::FindNewMaxHash_() {
  maxHash_ = FindNextHash_(maxHash_);
}

template <class T>
void HashTable<T>::RemoveEntry(HashTblEntry<T> *entry, bool del,
                               MemAlloc<HashTblEntry<T>> *entryAlctr) {
  assert(entryCnt_ > 0);

  UDT_HASHVAL hashVal = entry->GetHashVal();
  HashTblEntry<T> *nxtEntry = entry->GetNxt();
  HashTblEntry<T> *prevEntry = entry->GetPrev();

  // Update the top entry pointer if the entry to remove is the top entry.
  if (prevEntry == NULL) {
    topEntry_[hashVal] = nxtEntry;
  } else {
    prevEntry->SetNxt(nxtEntry);
  }

  // Update the bottom entry pointer if the entry to remove is the bottom entry.
  if (nxtEntry == NULL) {
    lastEntry_[hashVal] = prevEntry;
  } else {
    nxtEntry->SetPrev(prevEntry);
  }

  entryCnts_[hashVal]--;
  entryCnt_--;

  if (hashVal == maxHash_ && entryCnts_[hashVal] == 0) {
    FindNewMaxHash_();
  }

  if (del) {
    if (isExtrnlAlctr_) {
      assert(entryAlctr != NULL);
      entry->Clean();
      entryAlctr->FreeObject(entry);
    } else {
      delete entry;
    }
  }
}

template <class T>
UDT_HASHVAL HashTable<T>::FindNextHash_(UDT_HASHVAL crntHash) {
  if (entryCnt_ == 0)
    return 0;

  assert(crntHash > 0);

  for (UDT_HASHVAL nxtHash = crntHash - 1; nxtHash != 0; nxtHash--) {
    if (entryCnts_[nxtHash] > 0)
      return nxtHash;
  }

  return 0;
}

template <class T> void HashTable<T>::GetFullList(LinkedList<T> *lst) {
  for (UDT_HASHVAL crntHash = maxHash_; crntHash >= 0 && crntHash <= maxHash_;
       crntHash--) {
    if (entryCnts_[crntHash] > 0) {
      for (HashTblEntry<T> *crntEntry = topEntry_[crntHash]; crntEntry != NULL;
           crntEntry = crntEntry->GetNxt()) {
        lst->InsrtElmnt(crntEntry->GetElmnt());
      }
    }
  }

  assert(lst->GetElmntCnt() == entryCnt_);
}

template <class T>
BinHashTable<T>::BinHashTable(int16_t keyBitCnt, int16_t hashBitCnt,
                              bool isExtrnlAlctr, UDT_HASHTBL_CPCTY maxEntryCnt)
    : HashTable<T>(1 + (UDT_HASHVAL)(((int64_t)(1) << hashBitCnt) - 1),
                   maxEntryCnt) {
  keyBitCnt_ = keyBitCnt;
  hashBitCnt_ = hashBitCnt;
  HashTable<T>::isExtrnlAlctr_ = isExtrnlAlctr;
  CmputConsts_();
  Init_();
}

template <class T>
void BinHashTable<T>::Clear(bool del,
                            MemAlloc<BinHashTblEntry<T>> *entryAlctr) {
  HashTable<T>::Clear(del, entryAlctr);
  Init_();
}

template <class T> void BinHashTable<T>::Init_() {
  srchPtr_ = NULL;
  srchKey_ = 0;

  maxKey_ = 0;
  // TODO(max): Fix this signed -> unsigned conversion as it overflows.
  lastMaxHash_ = (UDT_HASHVAL)INVALID_VALUE;
  lastMaxEntry_ = NULL;
}

template <class T> inline void BinHashTable<T>::CmputConsts_() {
  hashRShft_ = keyBitCnt_ - hashBitCnt_;
  assert(keyBitCnt_ < (8 * sizeof(UDT_HASHKEY)));
  keyMask_ = (((UDT_HASHKEY)1) << keyBitCnt_) - 1;
}

template <class T>
inline UDT_HASHVAL BinHashTable<T>::HashKey(UDT_HASHKEY key) {
  if (keyBitCnt_ == hashBitCnt_)
    return (UDT_HASHVAL)key;
  return ((UDT_HASHVAL)key >> hashRShft_);
}

template <class T>
HashTblEntry<T> *
BinHashTable<T>::InsertElement(const UDT_HASHKEY key, T *elmnt,
                               MemAlloc<BinHashTblEntry<T>> *allocator) {
  if (this->entryCnt_ == this->maxEntryCnt_)
    return NULL;

  UDT_HASHVAL hashVal = HashKey(key);
  assert(hashVal < HashTable<T>::tblSize_);

  BinHashTblEntry<T> *newEntry;
  if (this->isExtrnlAlctr_) {
    assert(allocator != NULL);
    newEntry = allocator->GetObject();
    newEntry->Construct(key, elmnt, hashVal);
  } else {
    newEntry = new BinHashTblEntry<T>(key, elmnt, hashVal);
  }

  HashTable<T>::AddNewEntry_(newEntry, hashVal);

  if (key > maxKey_)
    maxKey_ = key;

  if (hashVal > this->maxHash_) {
    this->maxHash_ = hashVal;
  }

  return newEntry;
}

template <class T> void BinHashTable<T>::ReInsertEntry(HashTblEntry<T> *entry) {
  UDT_HASHVAL hashVal = entry->GetHashVal();
  UDT_HASHKEY key = entry->GetKey();
  assert(hashVal < HashTable<T>::tblSize_);

  HashTblEntry<T> *prev = entry->GetPrev();
  HashTblEntry<T> *nxt = entry->GetNxt();

  if (nxt == NULL) {
    this->lastEntry_[hashVal] = entry;
  } else {
    nxt->SetPrev(entry);
  }

  if (prev == NULL) {
    this->topEntry_[hashVal] = entry;
  } else {
    prev->SetNxt(entry);
  }

  if (key > maxKey_) {
    this->maxKey_ = key;
  }

  if (hashVal > this->maxHash_) {
    this->maxHash_ = hashVal;
  }

  this->entryCnts_[hashVal]++;
  this->entryCnt_++;
}

template <class T> T *BinHashTable<T>::ExtractMax() {
  T *elmnt;
  HashTblEntry<T> *maxEntry;

  if (this->entryCnt_ == 0)
    return NULL;

  if (keyBitCnt_ == hashBitCnt_) {
    // If all bits of the key are used as a hash value, then all entries at a
    // given table index have the same key value and we just get any one of
    // them. The top would be the easiest one.
    maxEntry = this->topEntry_[this->maxHash_];
  } else {
    // We have to conduct a linear search for the entry with the maximum key.
    HashTblEntry<T> *crntEnt;
    UDT_HASHKEY maxKey;
    maxEntry = this->topEntry_[this->maxHash_];
    maxKey = maxEntry->GetKey();

    for (crntEnt = maxEntry->GetNxt(); crntEnt != NULL;
         crntEnt = crntEnt->GetNxt()) {
      if (crntEnt->GetKey() > maxKey) {
        maxEntry = crntEnt;
        maxKey = crntEnt->GetKey();
      }
    }
  }

  RemoveEntry(maxEntry);
  UpdtLastMax_(maxEntry, this->maxHash_);
  elmnt = maxEntry->GetElmnt();

  return elmnt;
}

template <class T> HashTblEntry<T> *BinHashTable<T>::GetMaxEntry() {
  if (this->entryCnt_ == 0) {
    return NULL;
  }

  assert(keyBitCnt_ == hashBitCnt_);
  // If all bits of the key are used as a hash value, then all entries at a
  // given table index have the same key value and we just get any one of them.
  return this->topEntry_[this->maxHash_];
}

template <class T>
HashTblEntry<T> *BinHashTable<T>::GetNextEntry(HashTblEntry<T> *crntEntry) {
  assert(HashTable<T>::entryCnt_ != 0);
  assert(keyBitCnt_ == hashBitCnt_);
  HashTblEntry<T> *nxtEntry = crntEntry->GetNxt();
  UDT_HASHVAL hashVal = crntEntry->GetHashVal();

  if (nxtEntry == NULL && hashVal > 0) {
    hashVal = this->FindNextHash_(hashVal);
    nxtEntry = this->topEntry_[hashVal];
  }

  return nxtEntry;
}

template <class T>
void BinHashTable<T>::UpdtLastMax_(HashTblEntry<T> *maxEntry,
                                   UDT_HASHVAL maxHash) {
  if (lastMaxEntry_ != NULL)
    delete lastMaxEntry_;
  lastMaxEntry_ = maxEntry;
  lastMaxHash_ = maxHash;
}

template <class T> void BinHashTable<T>::RestoreLastMax() {
  ReInsertEntry(lastMaxEntry_);

  maxKey_ = lastMaxEntry_->GetKey();
  this->maxHash_ = this->lastMaxHash_;
}

template <class T>
UDT_HASHTBL_CPCTY BinHashTable<T>::GetListSize(const UDT_HASHKEY key) {
  UDT_HASHVAL srchHash = HashKey(key);
  return this->entryCnts_[srchHash];
}

template <class T>
T *BinHashTable<T>::GetFirstMatch(const UDT_HASHKEY key, bool skipCollision) {
  if (this->entryCnt_ == 0)
    return NULL;

  srchKey_ = key;
  UDT_HASHVAL srchHash = HashKey(srchKey_);
  srchPtr_ = this->topEntry_[srchHash];

  if (skipCollision)
    FindNextMatch_();

  return srchPtr_ == NULL ? NULL : srchPtr_->GetElmnt();
}

template <class T> T *BinHashTable<T>::GetNextMatch(bool skipCollision) {
  assert(srchPtr_ != NULL);
  srchPtr_ = srchPtr_->GetNxt();

  if (skipCollision)
    FindNextMatch_();

  return srchPtr_ == NULL ? NULL : srchPtr_->GetElmnt();
}

template <class T> void BinHashTable<T>::FindNextMatch_() {
  for (; srchPtr_ != NULL; srchPtr_ = srchPtr_->GetNxt()) {
    if (((BinHashTblEntry<T> *)srchPtr_)->GetKey() == srchKey_)
      return;
  }
}

template <class T>
T *BinHashTable<T>::GetLastMatch(const UDT_HASHKEY key, bool skipCollision) {
  if (this->entryCnt_ == 0)
    return NULL;

  srchKey_ = key;
  UDT_HASHVAL srchHash = HashKey(srchKey_);
  srchPtr_ = this->lastEntry_[srchHash];

  if (skipCollision)
    FindPrevMatch_();

  return srchPtr_ == NULL ? NULL : srchPtr_->GetElmnt();
}

template <class T> T *BinHashTable<T>::GetPrevMatch(bool skipCollision) {
  assert(srchPtr_ != NULL);
  srchPtr_ = srchPtr_->GetPrev();

  if (skipCollision)
    FindPrevMatch_();

  return srchPtr_ == NULL ? NULL : srchPtr_->GetElmnt();
}

template <class T> void BinHashTable<T>::FindPrevMatch_() {
  for (; srchPtr_ != NULL; srchPtr_ = srchPtr_->GetPrev()) {
    if (((BinHashTblEntry<T> *)srchPtr_)->GetKey() == srchKey_)
      return;
  }
}

template <class T>
void BinHashTable<T>::DeleteEntries(UDT_HASHKEY key, LinkedList<T> *elmntLst) {
  UDT_HASHVAL hashVal = this->HashKey_(key);

  for (T *elmnt = elmntLst->GetFrstElmnt(); elmnt != NULL;
       elmnt = elmntLst->GetNxtElmnt()) {
    DeleteEntry(hashVal, elmnt);
  }
}

template <class T>
bool BinHashTable<T>::DeleteEntry(UDT_HASHVAL hashVal, T *elmnt,
                                  MemAlloc<HashTblEntry<T>> *entryAlctr) {
  if (hashVal < 0 || hashVal > this->maxHash_)
    return false;

  for (HashTblEntry<T> *entry = this->topEntry_[hashVal]; entry != NULL;
       entry = entry->GetNxt()) {
    if (entry->GetElmnt() == elmnt) {
      RemoveEntry(entry, true, entryAlctr);
      return true;
    }
  }

  return false;
}

template <class T>
bool BinHashTable<T>::DeleteEntry(UDT_HASHKEY key, T *elmnt,
                                  MemAlloc<HashTblEntry<T>> *entryAlctr) {
  UDT_HASHVAL hashVal = this->HashKey_(key);
  return DeleteEntry(hashVal, elmnt, entryAlctr);
}

template <class T>
StrHashTable<T>::StrHashTable(UDT_HASHVAL size, bool useIndx,
                              UDT_HASHTBL_CPCTY maxEntryCnt)
    : HashTable<T>(size, maxEntryCnt) {
  indxdTbl_ = NULL;
  useIndx_ = useIndx;

  if (useIndx_) {
    UDT_HASHTBL_CPCTY i;
    indxdTbl_ = new StrHashTblEntry<T> *[this->maxEntryCnt_];

    for (i = 0; i < this->maxEntryCnt_; i++) {
      indxdTbl_[i] = NULL;
    }
  }
}

template <class T> StrHashTable<T>::~StrHashTable() {
  if (indxdTbl_)
    delete[] indxdTbl_;
}

template <class T> UDT_HASHVAL StrHashTable<T>::HashString_(const char *strng) {
  int i = 0;
  UDT_HASHVAL hashVal = 0;

  while (strng[i]) {
    hashVal += strng[i++] - '0';
  }

  hashVal %= this->tblSize_;
  return hashVal;
}

template <class T> T *StrHashTable<T>::FindElement(const char *srchName) {
  UDT_HASHVAL hashVal = HashString_(srchName);
  T *elmnt;

  for (HashTblEntry<T> *crnt = this->topEntry_[hashVal]; crnt != NULL;
       crnt = crnt->GetNxt()) {
    if ((elmnt = ((StrHashTblEntry<T> *)crnt)->GetElmnt(srchName)) != NULL) {
      return elmnt;
    }
  }

  return NULL; // Not found.
}

template <class T>
T *StrHashTable<T>::GetElement(const UDT_HASHTBL_CPCTY indx) {
  if (!useIndx_)
    return NULL;
  return indx >= this->entryCnt_ ? NULL : indxdTbl_[indx]->GetElmnt();
}

template <class T>
FUNC_RESULT StrHashTable<T>::InsertElement(const char *name, T *elmnt) {
  StrHashTblEntry<T> *newEntry;
  UDT_HASHVAL hashVal = HashString_(name);

  if (this->entryCnt_ == this->maxEntryCnt_)
    return RES_FAIL;

  assert(!HashTable<T>::isExtrnlAlctr_);
  newEntry = new StrHashTblEntry<T>(name, elmnt, hashVal);

  AddNewEntry_(newEntry, hashVal);

  if (hashVal > this->maxHash_) {
    this->maxHash_ = hashVal;
  }

  return RES_SUCCESS;
}

} // namespace opt_sched
} // namespace llvm

#endif
