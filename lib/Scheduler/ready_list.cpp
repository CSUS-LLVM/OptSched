#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/utilities.h"

#include <iostream>

using namespace llvm::opt_sched;

// pre-compute region info
void KeysHelper::initForRegion(DataDepGraph *DDG) {

  uint16_t CurrentOffset = 0, CurrentWidth = 0;

  uint64_t MaxKVs[MAX_SCHED_PRIRTS] = { 0 };

  // Calculate the number of bits needed to hold the maximum value of each
  // priority scheme
  for (int I = 0; I < Priorities.cnt; ++I) {
    LISTSCHED_HEURISTIC Heur = Priorities.vctr[I];
    uint64_t MaxV = 0;
    switch (Heur) {
    case LSH_CP:
    case LSH_CPR:
      MaxV = DDG->GetRootInst()->GetCrntLwrBound(DIR_BKWRD);
      break;

    case LSH_LUC:
    case LSH_UC:
      MaxV = DDG->GetMaxUseCnt();
      break;

    case LSH_NID:
    case LSH_LLVM:
      MaxV = DDG->GetInstCnt() - 1;
      break;

    case LSH_ISO:
      MaxV = DDG->GetMaxFileSchedOrder();
      break;

    case LSH_SC:
      MaxV = DDG->GetMaxScsrCnt();
      break;

    case LSH_LS:
      MaxV = DDG->GetMaxLtncySum();
      break;
    } // end switch

    // Track the size of the key and the width and location of our values
    CurrentWidth = Utilities::clcltBitsNeededToHoldNum(MaxV);
    Entries[Heur] = PriorityEntry{CurrentWidth, CurrentOffset};
    MaxKVs[Heur] = MaxV;
    CurrentOffset += CurrentWidth;
  }   // end for

  // check to see if the key can fit in our type
  assert(CurrentOffset <= 8 * sizeof(HeurType));

  // set the key size value to the final offset of the key
  KeysSz = CurrentOffset;

  //set maximumvalues needed to compute keys
  MaxNID = MaxKVs[LSH_NID];
  MaxISO = MaxKVs[LSH_ISO];

  // mark the object as initialized
  WasInitialized = true;

  // set the max value using the values compute key
  MaxValue = computeKey(MaxKVs);
}

// compute key
HeurType KeysHelper::computeKey(SchedInstruction *Inst, bool IncludeDynamic) const {
  assert(WasInitialized);

  HeurType Key= 0;
  for (int I = 0; I < Priorities.cnt; ++I) {
    LISTSCHED_HEURISTIC Heur = Priorities.vctr[I];
    HeurType PriorityValue = 0;
    switch (Heur) {
    case LSH_CP:
    case LSH_CPR:
      PriorityValue = Inst->GetCrtclPath(DIR_BKWRD);
      break;

    case LSH_LUC:
      PriorityValue = IncludeDynamic ? Inst->CmputLastUseCnt() : 0;
      break;

    case LSH_UC:
      PriorityValue = Inst->NumUses();
      break;

    case LSH_NID:
    case LSH_LLVM:
      PriorityValue = MaxNID - Inst->GetNodeID();
      break;

    case LSH_ISO:
      PriorityValue = MaxISO - Inst->GetFileSchedOrder();
      break;

    case LSH_SC:
      PriorityValue = Inst->GetScsrCnt();
      break;

    case LSH_LS:
      PriorityValue = Inst->GetLtncySum();
      break;
    }

    Key <<= Entries[Heur].Width;
    Key |= PriorityValue;
  }
  return Key;
}

HeurType KeysHelper::computeKey(const uint64_t *Values) const {
  assert(WasInitialized);

  HeurType Key = 0;

  for (int I = 0; I < Priorities.cnt; ++I) {
    LISTSCHED_HEURISTIC Heur = Priorities.vctr[I];
    Key <<= Entries[Heur].Width;
    Key |= Values[Heur];
  }

  return Key;
}

ReadyList::ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts) {
  prirts_ = prirts;

  // Initialize an array of KeyedEntry if a dynamic heuristic is used. This
  // enable fast updating for dynamic heuristics.
  if (prirts_.isDynmc) {
    keyedEntries_.resize(dataDepGraph->GetInstCnt());
  }

  // Initialize the KeyHelper
  KHelper = KeysHelper(prirts);
  KHelper.initForRegion(dataDepGraph);

  // if we have an luc in the Priorities then lets store some info about it
  // to improve efficiency
  PriorityEntry LUCEntry = KHelper.getPriorityEntry(LSH_LUC);
  if (LUCEntry.Width) {
    useCntBits_ = LUCEntry.Width;
    LUCOffset = LUCEntry.Offset;
  }

#ifdef IS_DEBUG_READY_LIST2
  Logger::Info("The ready list key size is %d bits", KHelper->getKeySizeInBits());
#endif

}

ReadyList::~ReadyList() { Reset(); }

void ReadyList::Reset() {
  prirtyLst_.Reset();
  latestSubLst_.Reset();
}

void ReadyList::CopyList(ReadyList *otherList) {
  assert(prirtyLst_.GetElmntCnt() == 0);
  assert(latestSubLst_.GetElmntCnt() == 0);
  assert(otherList != NULL);

  // Copy the ready list and create the array of keyed entries. If a dynamic
  // heuristic is not used then the second parameter should be an empty array.
  prirtyLst_.CopyList(&otherList->prirtyLst_, keyedEntries_);
}

unsigned long ReadyList::CmputKey_(SchedInstruction *inst, bool isUpdate,
                                   bool &changed) {
  int16_t OldLastUseCnt, NewLastUseCnt;

  // if we have an LUC Priority then we need to save the oldLUC
  OldLastUseCnt = inst->GetLastUseCnt();

  HeurType Key = KHelper.computeKey(inst, /*IncludeDynamic*/ true);

  //check if the luc value changed
  HeurType Mask = (0x01 << useCntBits_) - 1;
  HeurType LUCVal = (Key >> LUCOffset) & Mask;
  NewLastUseCnt = (int16_t) LUCVal;
  //set changed if the compute is not an update or the luc was changed
  changed = !isUpdate || OldLastUseCnt != NewLastUseCnt;

  return Key;
}

void ReadyList::AddLatestSubLists(LinkedList<SchedInstruction> *lst1,
                                  LinkedList<SchedInstruction> *lst2) {
  assert(latestSubLst_.GetElmntCnt() == 0);
  if (lst1 != NULL)
    AddLatestSubList_(lst1);
  if (lst2 != NULL)
    AddLatestSubList_(lst2);
  prirtyLst_.ResetIterator();
}

void ReadyList::Print(std::ostream &out) {
  out << "Ready List: ";
  for (const auto *crntInst = prirtyLst_.GetFrstElmnt(); crntInst != NULL;
       crntInst = prirtyLst_.GetNxtElmnt()) {
    out << " " << crntInst->GetNum();
  }
  out << '\n';

  prirtyLst_.ResetIterator();
}

void ReadyList::AddLatestSubList_(LinkedList<SchedInstruction> *lst) {
  assert(lst != NULL);

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "Adding to the ready list: ";
#endif

  // Start iterating from the bottom of the list to access the most recent
  // instructions first.
  for (SchedInstruction *crntInst = lst->GetLastElmnt(); crntInst != NULL;
       crntInst = lst->GetPrevElmnt()) {
    // Once an instruction that is already in the ready list has been
    // encountered, this instruction and all the ones above it must be in the
    // ready list already.
    if (crntInst->IsInReadyList())
      break;
    AddInst(crntInst);
#ifdef IS_DEBUG_READY_LIST2
    Logger::GetLogStream() << crntInst->GetNum() << ", ";
#endif
    crntInst->PutInReadyList();
    latestSubLst_.InsrtElmnt(crntInst);
  }

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "\n";
#endif
}

void ReadyList::RemoveLatestSubList() {
#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "Removing from the ready list: ";
#endif

  for (SchedInstruction *inst = latestSubLst_.GetFrstElmnt(); inst != NULL;
       inst = latestSubLst_.GetNxtElmnt()) {
    assert(inst->IsInReadyList());
    inst->RemoveFromReadyList();
#ifdef IS_DEBUG_READY_LIST2
    Logger::GetLogStream() << inst->GetNum() << ", ";
#endif
  }

#ifdef IS_DEBUG_READY_LIST2
  Logger::GetLogStream() << "\n";
#endif
}

void ReadyList::ResetIterator() { prirtyLst_.ResetIterator(); }

void ReadyList::AddInst(SchedInstruction *inst) {
  bool changed;
  unsigned long key = CmputKey_(inst, false, changed);
  assert(changed == true);
  KeyedEntry<SchedInstruction, unsigned long> *entry =
      prirtyLst_.InsrtElmnt(inst, key, true);
  InstCount instNum = inst->GetNum();
  if (prirts_.isDynmc)
    keyedEntries_[instNum] = entry;
}

void ReadyList::AddList(LinkedList<SchedInstruction> *lst) {
  SchedInstruction *crntInst;

  if (lst != NULL)
    for (crntInst = lst->GetFrstElmnt(); crntInst != NULL;
         crntInst = lst->GetNxtElmnt()) {
      AddInst(crntInst);
    }

  prirtyLst_.ResetIterator();
}

InstCount ReadyList::GetInstCnt() const { return prirtyLst_.GetElmntCnt(); }

SchedInstruction *ReadyList::GetNextPriorityInst() {
  return prirtyLst_.GetNxtPriorityElmnt();
}

SchedInstruction *ReadyList::GetNextPriorityInst(unsigned long &key) {
  return prirtyLst_.GetNxtPriorityElmnt(key);
}

void ReadyList::UpdatePriorities() {
  assert(prirts_.isDynmc);

  SchedInstruction *inst;
  bool instChanged = false;
  for (inst = prirtyLst_.GetFrstElmnt(); inst != NULL;
       inst = prirtyLst_.GetNxtElmnt()) {
    unsigned long key = CmputKey_(inst, true, instChanged);
    if (instChanged) {
      prirtyLst_.BoostEntry(keyedEntries_[inst->GetNum()], key);
    }
  }
}

void ReadyList::RemoveNextPriorityInst() { prirtyLst_.RmvCrntElmnt(); }

bool ReadyList::FindInst(SchedInstruction *inst, int &hitCnt) {
  return prirtyLst_.FindElmnt(inst, hitCnt);
}

unsigned long ReadyList::MaxPriority() { return KHelper.getMaxValue(); }
