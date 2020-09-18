#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/utilities.h"

using namespace llvm::opt_sched;

ReadyList::ReadyList(DataDepGraph *dataDepGraph, SchedPriorities prirts) {
  prirts_ = prirts;
  int i;
  uint16_t totKeyBits = 0;

  // Initialize an array of KeyedEntry if a dynamic heuristic is used. This
  // enable fast updating for dynamic heuristics.
  if (prirts_.isDynmc) {
    keyedEntries_.resize(dataDepGraph->GetInstCnt());
  }

  useCntBits_ = crtclPathBits_ = scsrCntBits_ = ltncySumBits_ = nodeID_Bits_ =
      inptSchedOrderBits_ = 0;

  // Calculate the number of bits needed to hold the maximum value of each
  // priority scheme
  for (i = 0; i < prirts.cnt; i++) {
    switch (prirts.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      maxCrtclPath_ = dataDepGraph->GetRootInst()->GetCrntLwrBound(DIR_BKWRD);
      crtclPathBits_ = Utilities::clcltBitsNeededToHoldNum(maxCrtclPath_);
      totKeyBits += crtclPathBits_;
      break;

    case LSH_LUC:
      for (int j = 0; j < dataDepGraph->GetInstCnt(); j++) {
        keyedEntries_[j] = NULL;
      }
      maxUseCnt_ = dataDepGraph->GetMaxUseCnt();
      useCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxUseCnt_);
      totKeyBits += useCntBits_;
      break;

    case LSH_UC:
      maxUseCnt_ = dataDepGraph->GetMaxUseCnt();
      useCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxUseCnt_);
      totKeyBits += useCntBits_;
      break;

    case LSH_NID:
    case LSH_LLVM:
      maxNodeID_ = dataDepGraph->GetInstCnt() - 1;
      nodeID_Bits_ = Utilities::clcltBitsNeededToHoldNum(maxNodeID_);
      totKeyBits += nodeID_Bits_;
      break;

    case LSH_ISO:
      maxInptSchedOrder_ = dataDepGraph->GetMaxFileSchedOrder();
      inptSchedOrderBits_ =
          Utilities::clcltBitsNeededToHoldNum(maxInptSchedOrder_);
      totKeyBits += inptSchedOrderBits_;
      break;

    case LSH_SC:
      maxScsrCnt_ = dataDepGraph->GetMaxScsrCnt();
      scsrCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxScsrCnt_);
      totKeyBits += scsrCntBits_;
      break;

    case LSH_LS:
      maxLtncySum_ = dataDepGraph->GetMaxLtncySum();
      ltncySumBits_ = Utilities::clcltBitsNeededToHoldNum(maxLtncySum_);
      totKeyBits += ltncySumBits_;
      break;
    } // end switch
  }   // end for

  assert(totKeyBits <= 8 * sizeof(unsigned long));

#ifdef IS_DEBUG_READY_LIST2
  Logger::Info("The ready list key size is %d bits", totKeyBits);
#endif

  int16_t keySize = 0;
  maxPriority_ = 0;
  for (i = 0; i < prirts_.cnt; i++) {
    switch (prirts_.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      AddPrirtyToKey_(maxPriority_, keySize, crtclPathBits_, maxCrtclPath_,
                      maxCrtclPath_);
      break;
    case LSH_LUC:
    case LSH_UC:
      AddPrirtyToKey_(maxPriority_, keySize, useCntBits_, maxUseCnt_,
                      maxUseCnt_);
      break;
    case LSH_NID:
    case LSH_LLVM:
      AddPrirtyToKey_(maxPriority_, keySize, nodeID_Bits_, maxNodeID_,
                      maxNodeID_);
      break;
    case LSH_ISO:
      AddPrirtyToKey_(maxPriority_, keySize, inptSchedOrderBits_,
                      maxInptSchedOrder_, maxInptSchedOrder_);
      break;
    case LSH_SC:
      AddPrirtyToKey_(maxPriority_, keySize, scsrCntBits_, maxScsrCnt_,
                      maxScsrCnt_);
      break;
    case LSH_LS:
      AddPrirtyToKey_(maxPriority_, keySize, ltncySumBits_, maxLtncySum_,
                      maxLtncySum_);
      break;
    }
  }
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
  unsigned long key = 0;
  int16_t keySize = 0;
  int i;
  int16_t oldLastUseCnt, newLastUseCnt;
  changed = true;
  if (isUpdate)
    changed = false;

  for (i = 0; i < prirts_.cnt; i++) {
    switch (prirts_.vctr[i]) {
    case LSH_CP:
    case LSH_CPR:
      AddPrirtyToKey_(key, keySize, crtclPathBits_,
                      inst->GetCrtclPath(DIR_BKWRD), maxCrtclPath_);
      break;

    case LSH_LUC:
      oldLastUseCnt = inst->GetLastUseCnt();
      newLastUseCnt = inst->CmputLastUseCnt();
      if (newLastUseCnt != oldLastUseCnt)
        changed = true;

      AddPrirtyToKey_(key, keySize, useCntBits_, newLastUseCnt, maxUseCnt_);
      break;

    case LSH_UC:
      AddPrirtyToKey_(key, keySize, useCntBits_, inst->NumUses(), maxUseCnt_);
      break;

    case LSH_NID:
    case LSH_LLVM:
      AddPrirtyToKey_(key, keySize, nodeID_Bits_,
                      maxNodeID_ - inst->GetNodeID(), maxNodeID_);
      break;

    case LSH_ISO:
      AddPrirtyToKey_(key, keySize, inptSchedOrderBits_,
                      maxInptSchedOrder_ - inst->GetFileSchedOrder(),
                      maxInptSchedOrder_);
      break;

    case LSH_SC:
      AddPrirtyToKey_(key, keySize, scsrCntBits_, inst->GetScsrCnt(),
                      maxScsrCnt_);
      break;

    case LSH_LS:
      AddPrirtyToKey_(key, keySize, ltncySumBits_, inst->GetLtncySum(),
                      maxLtncySum_);
      break;
    }
  }
  return key;
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

void ReadyList::AddPrirtyToKey_(unsigned long &key, int16_t &keySize,
                                int16_t bitCnt, unsigned long val,
                                unsigned long maxVal) {
  assert(val <= maxVal);
  if (keySize > 0)
    key <<= bitCnt;
  key |= val;
  keySize += bitCnt;
}

unsigned long ReadyList::MaxPriority() { return maxPriority_; }
