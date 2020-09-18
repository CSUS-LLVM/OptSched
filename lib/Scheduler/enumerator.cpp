#include "opt-sched/Scheduler/enumerator.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/hist_table.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/utilities.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>

using namespace llvm::opt_sched;

EnumTreeNode::EnumTreeNode() {
  isCnstrctd_ = false;
  isClean_ = true;
  rdyLst_ = NULL;
}
/*****************************************************************************/

EnumTreeNode::~EnumTreeNode() {
  assert(isCnstrctd_ || isClean_);
  assert(isCnstrctd_ || rdyLst_ == NULL);

  if (isCnstrctd_) {
    assert(frwrdLwrBounds_ != NULL);
    delete[] frwrdLwrBounds_;

    assert(exmndInsts_ != NULL);
    for (ExaminedInst *exmndInst = exmndInsts_->GetFrstElmnt();
         exmndInst != NULL; exmndInst = exmndInsts_->GetNxtElmnt()) {
      delete exmndInst;
    }
    exmndInsts_->Reset();
    delete exmndInsts_;

    assert(chldrn_ != NULL);
    delete chldrn_;

    if (rdyLst_ != NULL)
      delete rdyLst_;
    if (rsrvSlots_ != NULL)
      delete[] rsrvSlots_;
  } else {
    assert(isClean_);
  }
}
/*****************************************************************************/

void EnumTreeNode::Init_() {
  assert(isClean_);
  brnchCnt_ = 0;
  crntBrnchNum_ = 0;
  fsblBrnchCnt_ = 0;
  legalInstCnt_ = 0;
  hstry_ = NULL;
  rdyLst_ = NULL;
  dmntdNode_ = NULL;
  isArchivd_ = false;
  isFsbl_ = true;
  isLngthFsbl_ = true;
  lngthFsblBrnchCnt_ = 0;
  isLeaf_ = false;
  cost_ = INVALID_VALUE;
  costLwrBound_ = INVALID_VALUE;
  crntCycleBlkd_ = false;
  rsrvSlots_ = NULL;
  totalCostIsActualCost_ = false;
  totalCost_ = -1;
  suffix_.clear();
}
/*****************************************************************************/

void EnumTreeNode::Construct(EnumTreeNode *prevNode, SchedInstruction *inst,
                             Enumerator *enumrtr) {
  if (isCnstrctd_) {
    if (isClean_ == false) {
      Clean();
    }
  }

  Init_();

  prevNode_ = prevNode;
  inst_ = inst;
  enumrtr_ = enumrtr;
  time_ = prevNode_ == NULL ? 0 : prevNode_->time_ + 1;

  InstCount instCnt = enumrtr_->totInstCnt_;

  if (isCnstrctd_ == false) {
    exmndInsts_ = new LinkedList<ExaminedInst>(instCnt);
    chldrn_ = new LinkedList<HistEnumTreeNode>(instCnt);
    frwrdLwrBounds_ = new InstCount[instCnt];
  }

  if (enumrtr_->IsHistDom()) {
    CreateTmpHstry_();
  }

  FormPrtilSchedSig_();

  dmntdNode_ = NULL;

  isCnstrctd_ = true;
  isClean_ = false;
}
/*****************************************************************************/

void EnumTreeNode::Reset() {
  assert(isCnstrctd_);

  if (rdyLst_ != NULL) {
    rdyLst_->Reset();
  }

  if (exmndInsts_ != NULL) {
    for (ExaminedInst *exmndInst = exmndInsts_->GetFrstElmnt();
         exmndInst != NULL; exmndInst = exmndInsts_->GetNxtElmnt()) {
      delete exmndInst;
    }
    exmndInsts_->Reset();
  }

  if (chldrn_ != NULL) {
    chldrn_->Reset();
  }

  suffix_.clear();
}
/*****************************************************************************/

void EnumTreeNode::Clean() {
  assert(isCnstrctd_);
  Reset();

  if (rdyLst_ != NULL) {
    delete rdyLst_;
    rdyLst_ = NULL;
  }

  if (rsrvSlots_ != NULL) {
    delete[] rsrvSlots_;
    rsrvSlots_ = NULL;
  }

  isClean_ = true;
}
/*****************************************************************************/

void EnumTreeNode::FormPrtilSchedSig_() {
  SchedInstruction *inst = inst_;
  EnumTreeNode *prevNode = prevNode_;

  if (prevNode != NULL) {
    prtilSchedSig_ = prevNode->GetSig();
  } else { // if this is the root node
    prtilSchedSig_ = 0;
  }

  if (inst != NULL) {
    InstSignature instSig = inst->GetSig();
    prtilSchedSig_ ^= instSig;
  }
}
/*****************************************************************************/

void EnumTreeNode::SetLwrBounds() { SetLwrBounds(DIR_FRWRD); }
/*****************************************************************************/

void EnumTreeNode::SetLwrBounds(DIRECTION dir) {
  assert(dir == DIR_FRWRD);
  InstCount *&nodeLwrBounds = frwrdLwrBounds_;
  assert(nodeLwrBounds != NULL);
  DataDepGraph *dataDepGraph = enumrtr_->dataDepGraph_;
  dataDepGraph->GetCrntLwrBounds(dir, nodeLwrBounds);
}
/*****************************************************************************/

void EnumTreeNode::SetRsrvSlots(int16_t rsrvSlotCnt, ReserveSlot *rsrvSlots) {
  assert(rsrvSlots_ == NULL);
  rsrvSlots_ = NULL;

  if (rsrvSlotCnt == 0) { // If no unpipelined instrs are scheduled
    return;
  }

  int issuRate = enumrtr_->machMdl_->GetIssueRate();

  rsrvSlots_ = new ReserveSlot[issuRate];

  for (int i = 0; i < issuRate; i++) {
    rsrvSlots_[i].strtCycle = rsrvSlots[i].strtCycle;
    rsrvSlots_[i].endCycle = rsrvSlots[i].endCycle;
  }
}
/*****************************************************************************/

bool EnumTreeNode::DoesPartialSchedMatch(EnumTreeNode *othr) {
  EnumTreeNode *thisNode, *othrNode;

  if (othr->time_ != time_) {
    return false;
  }

  for (thisNode = this, othrNode = othr;
       thisNode->IsRoot() != true && othrNode->IsRoot() != true;
       thisNode = thisNode->GetParent(), othrNode = othrNode->GetParent()) {
    if (thisNode->GetInst() != othrNode->GetInst()) {
      return false;
    }
  }

  return true;
}
/****************************************************************************/

void EnumTreeNode::PrntPartialSched(std::ostream &out) {
  out << "\nPartial sched. at time " << time_ << ": ";

  for (EnumTreeNode *node = this; node->IsRoot() != true;
       node = node->GetParent()) {
    out << node->GetInstNum() << ", ";
  }
}
/*****************************************************************************/

void EnumTreeNode::NewBranchExmnd(SchedInstruction *inst, bool isLegal,
                                  bool isNodeDmntd, bool wasRlxInfsbl,
                                  bool isBrnchFsbl, DIRECTION dir,
                                  bool isLngthFsbl) {
  if (inst != NULL) {
    InstCount deadline = inst->GetCrntDeadline();
    InstCount cycleNum = enumrtr_->GetCycleNumFrmTime_(time_ + 1);
    InstCount slotNum = enumrtr_->GetSlotNumFrmTime_(time_ + 1);

    if (dir == DIR_FRWRD && cycleNum == deadline &&
        slotNum == enumrtr_->issuRate_ - 1) {
      // If that was the last issue slot in the instruction's deadline
      // then this instruction has just missed its deadline
      // and we don't need to consider this tree node any further
      isFsbl_ = false;
    }

    if (isLegal) {
      legalInstCnt_++;

      if (enumrtr_->prune_.nodeSup) {
        if (!isNodeDmntd) {
          ExaminedInst *exmndInst;
          exmndInst =
              new ExaminedInst(inst, wasRlxInfsbl, enumrtr_->dirctTightndLst_);
          exmndInsts_->InsrtElmnt(exmndInst);
        }
      }
    }
  }

  if (isLngthFsbl == false) {
    lngthFsblBrnchCnt_--;

    if (lngthFsblBrnchCnt_ == 0) {
      isLngthFsbl_ = false;
    }
  }

  crntBrnchNum_++;

  if (isBrnchFsbl == false) {
    ChildInfsbl();
  }
}
/*****************************************************************************/

void EnumTreeNode::SetBranchCnt(InstCount rdyLstSize, bool isLeaf) {
  assert(isLeaf == false || rdyLstSize == 0);
  isLeaf_ = isLeaf;

  if (isLeaf_) {
    isLngthFsbl_ = true;
  }

  brnchCnt_ = rdyLstSize + 1;
  isEmpty_ = rdyLstSize == 0;

  if (isLeaf_) {
    brnchCnt_ = 0;
  }

  fsblBrnchCnt_ = brnchCnt_;
  lngthFsblBrnchCnt_ = brnchCnt_;
}
/*****************************************************************************/

bool EnumTreeNode::ChkInstRdndncy(SchedInstruction *, int) {
  // Since we are optimizing spill cost, different permutations of the
  // same set of instructions within a certain cycle may have different
  // spill costs
  return false;
}
/*****************************************************************************/

bool EnumTreeNode::IsNxtSlotStall() {
  if (IsNxtCycleNew_() == false) {
    // If a stall has been scheduled in the current cycle then all slots in
    // this cycle must be stalls
    if (inst_ == NULL && time_ > 0) {
      return true;
    }
  }
  return false;
}
/*****************************************************************************/

bool EnumTreeNode::WasSprirNodeExmnd(SchedInstruction *cnddtInst) {
  if (cnddtInst == NULL)
    return false;

  for (ExaminedInst *exmndInst = exmndInsts_->GetFrstElmnt(); exmndInst != NULL;
       exmndInst = exmndInsts_->GetNxtElmnt()) {
    SchedInstruction *inst = exmndInst->GetInst();
    assert(inst != cnddtInst);

    if (inst->GetIssueType() == cnddtInst->GetIssueType() &&
        inst->BlocksCycle() == cnddtInst->BlocksCycle() &&
        inst->IsPipelined() == cnddtInst->IsPipelined()) {
      if (cnddtInst->IsScsrDmntd(inst)) {
        return true;
      } else {
#ifdef IS_DEBUG
        assert(!cnddtInst->IsScsrEquvlnt(inst));
#ifdef IS_DEBUG_NODEDOM
        if (inst->IsScsrDmntd(cnddtInst)) {
          stats::negativeNodeDominationHits++;
        }
#endif
#endif
      }
    }
  }

  return false;
}
/*****************************************************************************/

bool EnumTreeNode::WasRsrcDmnntNodeExmnd(SchedInstruction *cnddtInst) {
  if (cnddtInst == NULL) {
    return false;
  }

  SchedInstruction *inst;
  ExaminedInst *exmndInst;

  for (exmndInst = exmndInsts_->GetFrstElmnt(); exmndInst != NULL;
       exmndInst = exmndInsts_->GetNxtElmnt()) {
    inst = exmndInst->GetInst();
    assert(inst != cnddtInst);

    if (inst->GetIssueType() == cnddtInst->GetIssueType()) {
      if (exmndInst->wasRlxInfsbl()) {
        if (exmndInst->IsRsrcDmntd(cnddtInst)) {
          return true;
        }
      }
    }
  }
  return false;
}
/*****************************************************************************/

bool EnumTreeNode::IsBranchDominated(SchedInstruction *cnddtInst) {
  // Check if the given instruction can be feasibly replaced by a previously
  // examined instruction, which was found to be infeasible, thus proving by
  // contradiction that the given instruction is infeasible for this slot
  ExaminedInst *exmndInst = exmndInsts_->GetFrstElmnt();
  if (exmndInst == NULL)
    return false;

  SchedInstruction *inst = exmndInst->GetInst();
  assert(inst->IsSchduld() == false);

  if (cnddtInst->GetIssueType() != inst->GetIssueType())
    return false;

  InstCount deadline = inst->GetCrntDeadline();

  // If one of the successors of the given instruction will get delayed if
  // this instruction was replaced by the examined instruction
  // then the swapping won't be possible and the domination checking fails.
  if (cnddtInst->ProbeScsrsCrntLwrBounds(deadline))
    return false;

  return true;
}
/*****************************************************************************/

void EnumTreeNode::Archive() {
  assert(isArchivd_ == false);

  if (enumrtr_->IsCostEnum()) {
    hstry_->SetCostInfo(this, false, enumrtr_);
  }

  isArchivd_ = true;
}
/**************************************************************************/

EnumTreeNode::ExaminedInst::ExaminedInst(SchedInstruction *inst,
                                         bool wasRlxInfsbl,
                                         LinkedList<SchedInstruction> *) {
  inst_ = inst;
  wasRlxInfsbl_ = wasRlxInfsbl;
  tightndScsrs_ = NULL;
}
/****************************************************************************/

EnumTreeNode::ExaminedInst::~ExaminedInst() {
  if (tightndScsrs_ != NULL) {
    for (TightndInst *inst = tightndScsrs_->GetFrstElmnt(); inst != NULL;
         inst = tightndScsrs_->GetNxtElmnt()) {
      delete inst;
    }
    tightndScsrs_->Reset();
    delete tightndScsrs_;
  }
}
/****************************************************************************/

/****************************************************************************/
/****************************************************************************/

Enumerator::Enumerator(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                       InstCount schedUprBound, int16_t sigHashSize,
                       SchedPriorities prirts, Pruning PruningStrategy,
                       bool SchedForRPOnly, bool enblStallEnum,
                       Milliseconds timeout, InstCount preFxdInstCnt,
                       SchedInstruction *preFxdInsts[])
    : ConstrainedScheduler(dataDepGraph, machMdl, schedUprBound) {
  memAllocBlkSize_ = (int)timeout / TIMEOUT_TO_MEMBLOCK_RATIO;
  assert(preFxdInstCnt >= 0);

  if (memAllocBlkSize_ > MAX_MEMBLOCK_SIZE) {
    memAllocBlkSize_ = MAX_MEMBLOCK_SIZE;
  }

  if (memAllocBlkSize_ == 0) {
    memAllocBlkSize_ = 1;
  }

  isCnstrctd_ = false;
  rdyLst_ = NULL;
  prirts_ = prirts;
  prune_ = PruningStrategy;
  SchedForRPOnly_ = SchedForRPOnly;
  enblStallEnum_ = enblStallEnum;

  isEarlySubProbDom_ = true;

  rlxdSchdulr_ = new RJ_RelaxedScheduler(dataDepGraph, machMdl,
                                         schedUprBound_ + SCHED_UB_EXTRA,
                                         DIR_FRWRD, RST_DYNMC, INVALID_VALUE);

  for (int16_t i = 0; i < issuTypeCnt_; i++) {
    neededSlots_[i] = instCntPerIssuType_[i];
#ifdef IS_DEBUG_ISSUE_TYPES
    Logger::Info("#of insts. of type %d is %d", i, instCntPerIssuType_[i]);
#endif
  }

  dataDepGraph_->EnableBackTracking();

  maxNodeCnt_ = 0;
  createdNodeCnt_ = 0;
  exmndNodeCnt_ = 0;
  fxdInstCnt_ = 0;
  minUnschduldTplgclOrdr_ = 0;
  backTrackCnt_ = 0;
  fsblSchedCnt_ = 0;
  imprvmntCnt_ = 0;
  prevTrgtLngth_ = INVALID_VALUE;
  rgn_ = NULL;

  int16_t sigSize = 8 * sizeof(InstSignature) - 1;

  Milliseconds histTableInitTime = Utilities::GetProcessorTime();

  exmndSubProbs_ = NULL;

  if (IsHistDom()) {
    exmndSubProbs_ =
        new BinHashTable<HistEnumTreeNode>(sigSize, sigHashSize, true);
  }

  histTableInitTime = Utilities::GetProcessorTime() - histTableInitTime;
  stats::historyTableInitializationTime.Record(histTableInitTime);

  tightndLst_ = NULL;
  bkwrdTightndLst_ = NULL;
  dirctTightndLst_ = NULL;
  fxdLst_ = NULL;

  tightndLst_ = new LinkedList<SchedInstruction>(totInstCnt_);
  fxdLst_ = new LinkedList<SchedInstruction>(totInstCnt_);
  dirctTightndLst_ = new LinkedList<SchedInstruction>(totInstCnt_);
  bkwrdTightndLst_ = new LinkedList<SchedInstruction>(totInstCnt_);
  tmpLwrBounds_ = new InstCount[totInstCnt_];

  SetInstSigs_();
  iterNum_ = 0;
  preFxdInstCnt_ = preFxdInstCnt;
  preFxdInsts_ = preFxdInsts;

  isCnstrctd_ = true;
}
/****************************************************************************/

Enumerator::~Enumerator() {
  delete exmndSubProbs_;

  for (InstCount i = 0; i < schedUprBound_; i++) {
    if (frstRdyLstPerCycle_[i] != NULL) {
      delete frstRdyLstPerCycle_[i];
      frstRdyLstPerCycle_[i] = NULL;
    }
  }

  delete tightndLst_;
  delete dirctTightndLst_;
  delete fxdLst_;
  delete bkwrdTightndLst_;
  delete[] tmpLwrBounds_;
  tmpHstryNode_->Clean();
  delete tmpHstryNode_;
}
/****************************************************************************/

void Enumerator::SetupAllocators_() {
  int memAllocBlkSize = memAllocBlkSize_;
  int lastInstsEntryCnt = issuRate_ * (dataDepGraph_->GetMaxLtncy());
  int maxNodeCnt = issuRate_ * schedUprBound_ + 1;

  nodeAlctr_ = new EnumTreeNodeAlloc(maxNodeCnt);

  if (IsHistDom()) {
    hashTblEntryAlctr_ =
        new MemAlloc<BinHashTblEntry<HistEnumTreeNode>>(memAllocBlkSize);

    bitVctr1_ = new BitVector(totInstCnt_);
    bitVctr2_ = new BitVector(totInstCnt_);

    lastInsts_ = new SchedInstruction *[lastInstsEntryCnt];
    othrLastInsts_ = new SchedInstruction *[totInstCnt_];
  }
}
/****************************************************************************/

void Enumerator::ResetAllocators_() {
  nodeAlctr_->Reset();

  if (IsHistDom())
    hashTblEntryAlctr_->Reset();
}
/****************************************************************************/

void Enumerator::FreeAllocators_() {
  delete nodeAlctr_;
  nodeAlctr_ = NULL;
  delete rlxdSchdulr_;

  if (IsHistDom()) {
    delete hashTblEntryAlctr_;
    hashTblEntryAlctr_ = NULL;
    delete bitVctr1_;
    delete bitVctr2_;
    delete[] lastInsts_;
    delete[] othrLastInsts_;
  }
}
/****************************************************************************/

void Enumerator::Reset() {
  if (IsHistDom()) {
    exmndSubProbs_->Clear(false, hashTblEntryAlctr_);
  }

  ResetAllocators_();

  for (InstCount i = 0; i < schedUprBound_; i++) {
    if (frstRdyLstPerCycle_[i] != NULL) {
      frstRdyLstPerCycle_[i]->Reset();
    }
  }

  fxdLst_->Reset();
  tightndLst_->Reset();
  dirctTightndLst_->Reset();
  bkwrdTightndLst_->Reset();
  dataDepGraph_->SetSttcLwrBounds();
}
/****************************************************************************/

bool Enumerator::Initialize_(InstSchedule *sched, InstCount trgtLngth) {
  assert(trgtLngth <= schedUprBound_);
  trgtSchedLngth_ = trgtLngth;
  fsblSchedCnt_ = 0;
  imprvmntCnt_ = 0;
  crntSched_ = sched;
  minUnschduldTplgclOrdr_ = 0;
  backTrackCnt_ = 0;
  iterNum_++;

  if (ConstrainedScheduler::Initialize_(trgtSchedLngth_, fxdLst_) == false) {
    return false;
  }

  rlxdSchdulr_->Initialize(false);

  if (preFxdInstCnt_ > 0) {
    if (InitPreFxdInsts_() == false) {
      return false;
    }

    dataDepGraph_->SetDynmcLwrBounds();
  }

  if (FixInsts_(NULL) == false) {
    return false;
  }

  // For each issue slot the total number of options is equal to the total
  // number of instructions plus the option of scheduling a stall
  // This establishes an upper bound on the number of tree nodes
  // InstCount maxSlotCnt = trgtSchedLngth_ * issuRate_;
  maxNodeCnt_ = 0;

  int i;

  for (i = 0; i < issuTypeCnt_; i++) {
    avlblSlots_[i] = slotsPerTypePerCycle_[i] * trgtSchedLngth_;

    if (avlblSlots_[i] < neededSlots_[i]) {
#ifdef IS_DEBUG_FLOW
      Logger::Info("Length %d is infeasible; %d slots of type %d are needed.",
                   trgtLngth, neededSlots_[i], i);
#endif
      return false;
    }
  }

  rlxdSchdulr_->SetupPrirtyLst();

  createdNodeCnt_ = 0;
  fxdInstCnt_ = 0;
  rdyLst_ = NULL;
  CreateRootNode_();
  crntNode_ = rootNode_;
  ClearState_();
  return true;
}
/*****************************************************************************/

bool Enumerator::InitPreFxdInsts_() {
  for (InstCount i = 0; i < preFxdInstCnt_; i++) {
    bool fsbl = preFxdInsts_[i]->ApplyPreFxng(tightndLst_, fxdLst_);
    if (!fsbl)
      return false;
  }
  return true;
}
/*****************************************************************************/

void Enumerator::SetInstSigs_() {
  InstCount i;
  int16_t bitsForInstNum = Utilities::clcltBitsNeededToHoldNum(totInstCnt_ - 1);

  for (i = 0; i < totInstCnt_; i++) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(i);
    InstSignature sig = RandomGen::GetRand32();

    // ensure it is not zero
    if (sig == 0) {
      sig += 1;
    }

    // left shift by the number of bits needed to encode the instruction number
    sig <<= bitsForInstNum;

    // now, place the instruction number in the least significant bits
    sig |= i;

    //    sig &= 0x7fffffffffffffff;
    sig &= 0x7fffffff;

    assert(sig != 0);

    inst->SetSig(sig);
  }
}
/*****************************************************************************/

void Enumerator::CreateRootNode_() {
  rootNode_ = nodeAlctr_->Alloc(NULL, NULL, this);
  CreateNewRdyLst_();
  rootNode_->SetRdyLst(rdyLst_);
  rootNode_->SetLwrBounds(DIR_FRWRD);
  assert(rsrvSlotCnt_ == 0);
  rootNode_->SetRsrvSlots(rsrvSlotCnt_, rsrvSlots_);
  InitNewNode_(rootNode_);
  CmtLwrBoundTightnng_();
}
/*****************************************************************************/

namespace {

// TODO: Add unit tests to replace this style of debug code.
#if defined(IS_DEBUG_SUFFIX_SCHED)

void CheckHistNodeMatches(EnumTreeNode *const node,
                          HistEnumTreeNode *const histNode,
                          const char *loc = "CheckHistNodeMatches") {
  auto histPrefix = histNode->GetPrefix();
  auto currPrefix = [&]() {
    std::vector<InstCount> prefix;
    for (auto n = node; n != nullptr; n = n->GetParent()) {
      if (n->GetInstNum() != SCHD_STALL)
        prefix.push_back(n->GetInstNum());
    }
    return prefix;
  }();
  std::sort(histPrefix.begin(), histPrefix.end());
  std::sort(currPrefix.begin(), currPrefix.end());
  if (histPrefix.size() != currPrefix.size()) {
    printVector(histPrefix, "HistPrefix");
    printVector(currPrefix, "CurrPrefix");
    Logger::Fatal(
        "%s: Hist prefix size %llu doesn't match current prefix %llu!", loc,
        histPrefix.size(), currPrefix.size());
  }
  if (histPrefix != currPrefix) {
    printVector(histPrefix, "HistPrefix");
    printVector(currPrefix, "CurrPrefix");
    Logger::Fatal("%s: Hist prefix and current prefix are not permutations of "
                  "each other!",
                  loc, histPrefix.size(), currPrefix.size());
  }
}

void PrintSchedule(InstSchedule *const sched,
                   Logger::LOG_LEVEL level = Logger::INFO) {
  InstCount cycle, slot;
  std::stringstream s;
  for (auto inst = sched->GetFrstInst(cycle, slot); inst != INVALID_VALUE;
       inst = sched->GetNxtInst(cycle, slot)) {
    s << inst << ' ';
  }
  Logger::Log(level, false, "Schedule: %s", s.str().c_str());
}

#endif // IS_DEBUG_SUFFIX_SCHED

void AppendAndCheckSuffixSchedules(
    HistEnumTreeNode *const matchingHistNodeWithSuffix, SchedRegion *const rgn_,
    InstSchedule *const crntSched_, InstCount trgtSchedLngth_,
    LengthCostEnumerator *const thisAsLengthCostEnum,
    EnumTreeNode *const crntNode_, DataDepGraph *const dataDepGraph_) {
  assert(matchingHistNodeWithSuffix != nullptr && "Hist node is null");
  assert(matchingHistNodeWithSuffix->GetSuffix() != nullptr &&
         "Hist node suffix is null");
  assert(matchingHistNodeWithSuffix->GetSuffix()->size() > 0 &&
         "Hist node suffix size is zero");
  // For each matching history node, concatenate the suffix with the
  // current schedule and check to see if it's better than the best
  // schedule found so far.
  auto concatSched = std::unique_ptr<InstSchedule>(rgn_->AllocNewSched_());
  // Get the prefix.
  concatSched->Copy(crntSched_);

#if defined(IS_DEBUG_SUFFIX_SCHED)
  {
    auto prefix = matchingHistNodeWithSuffix->GetPrefix();
    if (prefix.size() != crntSched_->GetCrntLngth()) {
      PrintSchedule(crntSched_, Logger::ERROR);
      std::stringstream s;
      for (auto j : prefix)
        s << j << ' ';
      Logger::Error("Prefix: %s", s.str().c_str());
      s.str("");
      for (auto j : *matchingHistNodeWithSuffix->GetSuffix())
        s << (j == nullptr ? SCHD_STALL : j->GetNum()) << ' ';
      Logger::Error("SUffix: %s", s.str().c_str());
      Logger::Fatal(
          "Hist node prefix size %llu doesn't match current sched length %d!",
          prefix.size(), crntSched_->GetCrntLngth());
    }
  }
#endif

  // Concatenate the suffix.
  for (auto inst : *matchingHistNodeWithSuffix->GetSuffix())
    concatSched->AppendInst((inst == nullptr) ? SCHD_STALL : inst->GetNum());

    // Update and check.

#if defined(IS_DEBUG_SUFFIX_SCHED)
  if (concatSched->GetCrntLngth() != trgtSchedLngth_) {
    PrintSchedule(concatSched.get(), Logger::ERROR);
    PrintSchedule(crntSched_, Logger::ERROR);
    std::stringstream s;
    auto prefix = matchingHistNodeWithSuffix->GetPrefix();
    for (auto j : prefix)
      s << j << ' ';
    Logger::Error("Prefix: %s", s.str().c_str());
    s.str("");
    for (auto j : *matchingHistNodeWithSuffix->GetSuffix())
      s << (j == nullptr ? SCHD_STALL : j->GetNum()) << ' ';
    Logger::Error("SUffix: %s", s.str().c_str());
    Logger::Fatal("Suffix Scheduling: Concatenated schedule length %d "
                  "does not meet target length %d!",
                  concatSched->GetCrntLngth(), trgtSchedLngth_);
  }
#endif
  auto oldCost = thisAsLengthCostEnum->GetBestCost();
  auto newCost = rgn_->UpdtOptmlSched(concatSched.get(), thisAsLengthCostEnum);
#if defined(IS_DEBUG_SUFFIX_SCHED)
  Logger::Info("Found a concatenated schedule with node instruction %d",
               crntNode_->GetInstNum());
#endif
  if (newCost < oldCost) {
#if defined(IS_DEBUG_SUFFIX_SCHED)
    Logger::Info("Suffix Scheduling: Concatenated schedule has better "
                 "cost %d than best schedule %d!",
                 newCost, oldCost);
#endif
    // Don't forget to update the total cost and suffix for this node,
    // because we intentionally backtrack without visiting its
    // children.
    crntNode_->SetTotalCost(newCost);
    crntNode_->SetTotalCostIsActualCost(true);
    if (newCost == 0) {
      Logger::Info(
          "Suffix Scheduling: ***GOOD*** Schedule of cost 0 was found!");
    }
  } else {
#if defined(IS_DEBUG_SUFFIX_SCHED)
    Logger::Info("Suffix scheduling: Concatenated schedule does not have "
                 "better cost %d than best schedule %d.",
                 newCost, oldCost);
#endif
  }

  // Before backtracking, reset the SchedRegion state to where it was before
  // concatenation.
  rgn_->InitForSchdulng();
  InstCount cycleNum, slotNum;
  for (auto instNum = crntSched_->GetFrstInst(cycleNum, slotNum);
       instNum != INVALID_VALUE;
       instNum = crntSched_->GetNxtInst(cycleNum, slotNum)) {
    rgn_->SchdulInst(dataDepGraph_->GetInstByIndx(instNum), cycleNum, slotNum,
                     false);
  }
}
} // namespace

FUNC_RESULT Enumerator::FindFeasibleSchedule_(InstSchedule *sched,
                                              InstCount trgtLngth,
                                              Milliseconds deadline) {
  EnumTreeNode *nxtNode = NULL;
  bool allNodesExplrd = false;
  bool foundFsblBrnch = false;
  bool isCrntNodeFsbl = true;
  bool isTimeout = false;

  if (!isCnstrctd_)
    return RES_ERROR;

  assert(trgtLngth <= schedUprBound_);

  if (Initialize_(sched, trgtLngth) == false) {
    return RES_FAIL;
  }

#ifdef IS_DEBUG_NODES
  uint64_t prevNodeCnt = exmndNodeCnt_;
#endif

  while (!(allNodesExplrd || WasObjctvMet_())) {
    if (deadline != INVALID_VALUE && Utilities::GetProcessorTime() > deadline) {
      isTimeout = true;
      break;
    }

    mostRecentMatchingHistNode_ = nullptr;

    if (isCrntNodeFsbl) {
      foundFsblBrnch = FindNxtFsblBrnch_(nxtNode);
    } else {
      foundFsblBrnch = false;
    }

    if (foundFsblBrnch) {
      // (Chris): It's possible that the node we just determined to be feasible
      // dominates a history node with a suffix schedule. If this is the case,
      // then instead of continuing the search, we should generate schedules by
      // concatenating the best known suffix.

      StepFrwrd_(nxtNode);

      // Find matching history nodes with suffixes.
      auto matchingHistNodesWithSuffix = mostRecentMatchingHistNode_;

      // If there are no such matches, continue the search. Else,
      // generate concatenated schedules.
      if (!IsHistDom() || matchingHistNodesWithSuffix == nullptr) {
        // If a branch from the current node that leads to a feasible node has
        // been found, move on down the tree to that feasible node.
        isCrntNodeFsbl = true;
      } else {
        assert(this->IsCostEnum() && "Not a LengthCostEnum instance!");
        crntNode_->GetHistory()->SetSuffix(
            matchingHistNodesWithSuffix->GetSuffix());
        AppendAndCheckSuffixSchedules(matchingHistNodesWithSuffix, rgn_,
                                      crntSched_, trgtSchedLngth_,
                                      static_cast<LengthCostEnumerator *>(this),
                                      crntNode_, dataDepGraph_);
        isCrntNodeFsbl = BackTrack_();
      }
    } else {
      // All branches from the current node have been explored, and no more
      // branches that lead to feasible nodes have been found.
      if (crntNode_ == rootNode_) {
        allNodesExplrd = true;
      } else {
        isCrntNodeFsbl = BackTrack_();
      }
    }

#ifdef IS_DEBUG_FLOW
    crntNode_->PrntPartialSched(Logger::GetLogStream());
#endif
#ifdef IS_DEBUG
// Logger::PeriodicLog();
#endif
  }

#ifdef IS_DEBUG_NODES
  uint64_t crntNodeCnt = exmndNodeCnt_ - prevNodeCnt;
  stats::nodesPerLength.Record(crntNodeCnt);
#endif

  if (isTimeout)
    return RES_TIMEOUT;
  // Logger::Info("\nEnumeration at length %d done\n", trgtLngth);
  return fsblSchedCnt_ > 0 ? RES_SUCCESS : RES_FAIL;
}
/****************************************************************************/

bool Enumerator::FindNxtFsblBrnch_(EnumTreeNode *&newNode) {
  InstCount i;
  bool isEmptyNode;
  SchedInstruction *inst;
  InstCount brnchCnt = crntNode_->GetBranchCnt(isEmptyNode);
  InstCount crntBrnchNum = crntNode_->GetCrntBranchNum();
  bool isNodeDmntd, isRlxInfsbl;
  bool enumStall = false;
  bool isLngthFsbl = true;

#if defined(IS_DEBUG) || defined(IS_DEBUG_READY_LIST)
  InstCount rdyInstCnt = rdyLst_->GetInstCnt();
  assert(crntNode_->IsLeaf() || (brnchCnt != rdyInstCnt) ? 1 : rdyInstCnt);
#endif

#ifdef IS_DEBUG_READY_LIST
  Logger::Info("Ready List Size is %d", rdyInstCnt);
  // Warning! That will reset the instruction iterator!
  // rdyLst_->Print(Logger::GetLogStream());

  stats::maxReadyListSize.SetMax(rdyInstCnt);
#endif

  if (crntBrnchNum == 0 && SchedForRPOnly_)
    crntNode_->SetFoundInstWithUse(IsUseInRdyLst_());

  for (i = crntBrnchNum; i < brnchCnt && crntNode_->IsFeasible(); i++) {
#ifdef IS_DEBUG_FLOW
    Logger::Info("Probing branch %d out of %d", i, brnchCnt);
#endif

    if (i == brnchCnt - 1) {
      // then we only have the option of scheduling a stall
      assert(isEmptyNode == false || brnchCnt == 1);
      inst = NULL;
      enumStall = EnumStall_();

      if (isEmptyNode || crntNode_->GetLegalInstCnt() == 0 || enumStall) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
        stats::stalls++;
#endif
      } else {
        crntNode_->NewBranchExmnd(inst, false, false, false, false, DIR_FRWRD,
                                  false);
        continue;
      }
    } else {
      inst = rdyLst_->GetNextPriorityInst();
      assert(inst != NULL);
      bool isLegal = ChkInstLglty_(inst);
      isLngthFsbl = isLegal;

      if (isLegal == false || crntNode_->ChkInstRdndncy(inst, i)) {
#ifdef IS_DEBUG_FLOW
        Logger::Info("Inst %d is illegal or redundant in cyc%d/slt%d",
                     inst->GetNum(), crntCycleNum_, crntSlotNum_);
#endif
        exmndNodeCnt_++;
        crntNode_->NewBranchExmnd(inst, false, false, false, false, DIR_FRWRD,
                                  isLngthFsbl);
        continue;
      }
    }

    exmndNodeCnt_++;

#ifdef IS_DEBUG_INFSBLTY_TESTS
    stats::feasibilityTests++;
#endif
    isNodeDmntd = isRlxInfsbl = false;
    isLngthFsbl = true;

    if (ProbeBranch_(inst, newNode, isNodeDmntd, isRlxInfsbl, isLngthFsbl)) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
      stats::feasibilityHits++;
#endif
      return true;
    } else {
      RestoreCrntState_(inst, newNode);
      crntNode_->NewBranchExmnd(inst, true, isNodeDmntd, isRlxInfsbl, false,
                                DIR_FRWRD, isLngthFsbl);
    }
  }

  return false; // No feasible branch has been found at the current node
}
/*****************************************************************************/

bool Enumerator::ProbeBranch_(SchedInstruction *inst, EnumTreeNode *&newNode,
                              bool &isNodeDmntd, bool &isRlxInfsbl,
                              bool &isLngthFsbl) {
  bool fsbl;
  newNode = NULL;
  isLngthFsbl = false;

  assert(IsStateClear_());
  assert(inst == NULL || inst->IsSchduld() == false);

#ifdef IS_DEBUG_FLOW
  InstCount instNum = inst == NULL ? -2 : inst->GetNum();
  Logger::Info("Probing inst %d in cycle %d / slot %d", instNum, crntCycleNum_,
               crntSlotNum_);
#endif

  // If this instruction is prefixed, it cannot be scheduled earlier than its
  // prefixed cycle
  if (inst != NULL)
    if (inst->GetPreFxdCycle() != INVALID_VALUE)
      if (inst->GetPreFxdCycle() != crntCycleNum_) {
        return false;
      }

  if (inst != NULL) {
    if (inst->GetCrntLwrBound(DIR_FRWRD) > crntCycleNum_) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
      stats::forwardLBInfeasibilityHits++;
#endif
      return false;
    }

    if (inst->GetCrntDeadline() < crntCycleNum_) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
      stats::backwardLBInfeasibilityHits++;
#endif
      return false;
    }
  }

  // If we are scheduling for register pressure only, and this branch
  // defines a register but does not use any, we can prune this branch
  // if another instruction in the ready list does use a register.
  if (SchedForRPOnly_) {
    if (inst != NULL && crntNode_->FoundInstWithUse() &&
        inst->GetAdjustedUseCnt() == 0 && !dataDepGraph_->DoesFeedUser(inst))
      return false;
  }

  if (prune_.nodeSup) {
    if (inst != NULL)
      if (crntNode_->WasSprirNodeExmnd(inst)) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
        stats::nodeSuperiorityInfeasibilityHits++;
#endif
        isNodeDmntd = true;
        return false;
      }
  }

  if (inst != NULL) {
    inst->Schedule(crntCycleNum_, crntSlotNum_);
    DoRsrvSlots_(inst);
    state_.instSchduld = true;
  }

  fsbl = ProbeIssuSlotFsblty_(inst);
  state_.issuSlotsProbed = true;

  if (!fsbl) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
    stats::slotCountInfeasibilityHits++;
#endif
    return false;
  }

  fsbl = TightnLwrBounds_(inst);
  state_.lwrBoundsTightnd = true;

  if (fsbl == false) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
    stats::rangeTighteningInfeasibilityHits++;
#endif
    return false;
  }

  state_.instFxd = true;

  newNode = nodeAlctr_->Alloc(crntNode_, inst, this);
  newNode->SetLwrBounds(DIR_FRWRD);
  newNode->SetRsrvSlots(rsrvSlotCnt_, rsrvSlots_);

  // If a node (sub-problem) that dominates the candidate node (sub-problem)
  // has been examined already and found infeasible
  if (prune_.histDom) {
    if (isEarlySubProbDom_)
      if (WasDmnntSubProbExmnd_(inst, newNode)) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
        stats::historyDominationInfeasibilityHits++;
#endif
        return false;
      }
  }

  // Try to find a relaxed schedule for the unscheduled instructions
  if (prune_.rlxd) {
    fsbl = RlxdSchdul_(newNode);
    state_.rlxSchduld = true;

    if (fsbl == false) {
#ifdef IS_DEBUG_INFSBLTY_TESTS
      stats::relaxedSchedulingInfeasibilityHits++;
#endif
      isRlxInfsbl = true;

      return false;
    }
  }

  isLngthFsbl = true;
  assert(newNode != NULL);
  return true;
}
/****************************************************************************/

bool Enumerator::ProbeIssuSlotFsblty_(SchedInstruction *inst) {
  bool endOfCycle = crntSlotNum_ == issuRate_ - 1;
  IssueType issuType = inst == NULL ? ISSU_STALL : inst->GetIssueType();

  if (issuType != ISSU_STALL) {
    assert(avlblSlotsInCrntCycle_[issuType] > 0);
    assert(avlblSlots_[issuType] > 0);
    avlblSlotsInCrntCycle_[issuType]--;
    avlblSlots_[issuType]--;
    neededSlots_[issuType]--;
    assert(avlblSlots_[issuType] >= neededSlots_[issuType]);
  }

  int16_t i;

  for (i = 0; i < issuTypeCnt_; i++) {
    // Due to stalls in the cycle that has just completed, the available
    // slots for each type that did not get filled can never be used.
    // This could not have been decremented when the stalls were
    // scheduled, because it was not clear then which type was affected
    // by each stall
    if (endOfCycle) {
      avlblSlots_[i] -= avlblSlotsInCrntCycle_[i];
      avlblSlotsInCrntCycle_[i] = 0;
    }

    if (avlblSlots_[i] < neededSlots_[i]) {
      return false;
    }
  }

  return true;
}
/*****************************************************************************/

void Enumerator::RestoreCrntState_(SchedInstruction *inst,
                                   EnumTreeNode *newNode) {
  if (newNode != NULL) {
    if (newNode->IsArchived() == false) {
      nodeAlctr_->Free(newNode);
    }
  }

  if (state_.lwrBoundsTightnd) {
    UnTightnLwrBounds_(inst);
  }

  if (state_.instSchduld) {
    assert(inst != NULL);
    UndoRsrvSlots_(inst);
    inst->UnSchedule();
  }

  if (state_.issuSlotsProbed) {
    crntNode_->GetSlotAvlblty(avlblSlots_, avlblSlotsInCrntCycle_);

    if (inst != NULL) {
      IssueType issuType = inst->GetIssueType();
      neededSlots_[issuType]++;
    }
  }

  ClearState_();
}
/*****************************************************************************/

void Enumerator::StepFrwrd_(EnumTreeNode *&newNode) {
  SchedInstruction *instToSchdul = newNode->GetInst();
  InstCount instNumToSchdul;

  CreateNewRdyLst_();
  // Let the new node inherit its parent's ready list before we update it
  newNode->SetRdyLst(rdyLst_);

  if (instToSchdul == NULL) {
    instNumToSchdul = SCHD_STALL;
  } else {
    instNumToSchdul = instToSchdul->GetNum();
    SchdulInst_(instToSchdul, crntCycleNum_);
    rdyLst_->RemoveNextPriorityInst();

    if (instToSchdul->GetTplgclOrdr() == minUnschduldTplgclOrdr_) {
      minUnschduldTplgclOrdr_++;
    }
  }

  crntSched_->AppendInst(instNumToSchdul);

  MovToNxtSlot_(instToSchdul);
  assert(crntCycleNum_ <= trgtSchedLngth_);

  if (crntSlotNum_ == 0) {
    InitNewCycle_();
  }

  InitNewNode_(newNode);

#ifdef IS_DEBUG_FLOW
  Logger::Info("Stepping forward from node %lld to node %lld by scheduling "
               "inst. #%d in cycle #%d. CostLB=%d",
               crntNode_->GetParent()->GetNum(), crntNode_->GetNum(),
               instNumToSchdul, crntCycleNum_, crntNode_->GetCostLwrBound());
#endif

  CmtLwrBoundTightnng_();
  ClearState_();
}
/*****************************************************************************/

void Enumerator::InitNewNode_(EnumTreeNode *newNode) {
  crntNode_ = newNode;

  crntNode_->SetCrntCycleBlkd(isCrntCycleBlkd_);
  crntNode_->SetRealSlotNum(crntRealSlotNum_);

  if (IsHistDom()) {
    crntNode_->CreateHistory();
    assert(crntNode_->GetHistory() != tmpHstryNode_);
  }

  crntNode_->SetSlotAvlblty(avlblSlots_, avlblSlotsInCrntCycle_);

  UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
  bool isLeaf = schduldInstCnt_ == totInstCnt_;

  crntNode_->SetBranchCnt(rdyLst_->GetInstCnt(), isLeaf);

  createdNodeCnt_++;
  crntNode_->SetNum(createdNodeCnt_);
}
/*****************************************************************************/
namespace {
void SetTotalCostsAndSuffixes(EnumTreeNode *const currentNode,
                              EnumTreeNode *const parentNode,
                              const InstCount targetLength,
                              const bool suffixConcatenationEnabled) {
  // (Chris): Before archiving, set the total cost info of this node. If it's a
  // leaf node, then the total cost is the current cost. If it's an inner node,
  // then the total cost either has already been set (if one of its children had
  // a real cost), or hasn't been set, which means the total cost right now is
  // the dynamic lower bound of this node.

  if (currentNode->IsLeaf()) {
#if defined(IS_DEBUG_ARCHIVE)
    Logger::Info("Leaf node total cost %d", currentNode->GetCost());
#endif
    currentNode->SetTotalCost(currentNode->GetCost());
    currentNode->SetTotalCostIsActualCost(true);
  } else {
    if (!currentNode->GetTotalCostIsActualCost() &&
        (currentNode->GetTotalCost() == -1 ||
         currentNode->GetCostLwrBound() < currentNode->GetTotalCost())) {
#if defined(IS_DEBUG_ARCHIVE)
      Logger::Info("Inner node doesn't have a real cost yet. Setting total "
                   "cost to dynamic lower bound %d",
                   currentNode->GetCostLwrBound());
#endif
      currentNode->SetTotalCost(currentNode->GetCostLwrBound());
    }
  }

#if defined(IS_DEBUG_SUFFIX_SCHED)
  if (currentNode->GetTotalCostIsActualCost() &&
      currentNode->GetTotalCost() == -1) {
    Logger::Fatal("Actual cost was not set even though its flag was!");
  }
#endif

  // (Chris): If this node has an actual cost associated with the best schedule,
  // we want to propagate it backward only if this node's cost is less than the
  // parent node's cost.
  std::vector<SchedInstruction *> parentSuffix;
  if (parentNode != nullptr) {
    if (currentNode->GetTotalCostIsActualCost()) {
      if (suffixConcatenationEnabled &&
          (currentNode->IsLeaf() ||
           (!currentNode->IsLeaf() && currentNode->GetSuffix().size() > 0))) {
        parentSuffix.reserve(currentNode->GetSuffix().size() + 1);
        parentSuffix.push_back(currentNode->GetInst());
        parentSuffix.insert(parentSuffix.end(),
                            currentNode->GetSuffix().begin(),
                            currentNode->GetSuffix().end());
      }
      if (!parentNode->GetTotalCostIsActualCost()) {
#if defined(IS_DEBUG_ARCHIVE)
        Logger::Info("Current node has a real cost, but its parent doesn't. "
                     "Settings parent's total cost to %d",
                     currentNode->GetTotalCost());
#endif
        parentNode->SetTotalCost(currentNode->GetTotalCost());
        parentNode->SetTotalCostIsActualCost(true);
        parentNode->SetSuffix(std::move(parentSuffix));
      } else if (currentNode->GetTotalCost() < parentNode->GetTotalCost()) {
#if defined(IS_DEBUG_ARCHIVE)
        Logger::Info(
            "Current node has a real cost (%d), and so does parent. (%d)",
            currentNode->GetTotalCost(), parentNode->GetTotalCost());
#endif
        parentNode->SetTotalCost(currentNode->GetTotalCost());
        parentNode->SetSuffix(std::move(parentSuffix));
      }
    }
  }

// (Chris): Ensure that the prefix and the suffix of the current node contain
// no common instructions. This can be compiled out once the code is working.
#if defined(IS_DEBUG_SUFFIX_SCHED)
  if (suffixConcatenationEnabled) {

    void printVector(const std::vector<InstCount> &v, const char *label) {
      std::stringstream s;
      for (auto i : v)
        s << i << ' ';
      Logger::Info("%s: %s", label, s.str().c_str());
    }

    std::vector<InstCount> prefix;
    for (auto n = currentNode; n != nullptr; n = n->GetParent()) {
      if (n->GetInstNum() != SCHD_STALL)
        prefix.push_back(n->GetInstNum());
    }
    auto sortedPrefix = prefix;
    std::sort(sortedPrefix.begin(), sortedPrefix.end());

    std::vector<InstCount> suffix;
    for (auto i : currentNode->GetSuffix()) {
      suffix.push_back(i->GetNum());
    }
    auto sortedSuffix = suffix;
    std::sort(sortedSuffix.begin(), sortedSuffix.end());

    std::vector<InstCount> intersection;
    std::set_intersection(sortedPrefix.begin(), sortedPrefix.end(),
                          sortedSuffix.begin(), sortedSuffix.end(),
                          std::back_inserter(intersection));

    auto printVector = [](const std::vector<InstCount> &v, const char *prefix) {
      std::stringstream s;
      for (auto i : v)
        s << i << ' ';
      Logger::Error("SetTotalCostsAndSuffixes: %s: %s", prefix,
                    s.str().c_str());
    };
    if (intersection.size() != 0) {
      printVector(prefix, "prefix");
      printVector(suffix, "suffix");
      printVector(intersection, "intersection");
      Logger::Error("SetTotalCostsAndSuffixes: Error occurred when archiving "
                    "node with InstNum %d",
                    currentNode->GetInstNum());
      Logger::Fatal(
          "Prefix schedule and suffix schedule contain common instructions!");
    }
    if (suffix.size() > 0 && suffix.size() + prefix.size() != targetLength) {
      printVector(prefix, "prefix");
      printVector(suffix, "suffix");
      Logger::Fatal("Sum of suffix (%llu) and prefix (%llu) sizes doesn't "
                    "match target length %d!",
                    suffix.size(), prefix.size(), targetLength);
    }
    CheckHistNodeMatches(currentNode, currentNode->GetHistory(),
                         "SetTotalCostsAndSuffixes: CheckHistNodeMatches");
  }
#endif
}
} // end anonymous namespace

bool Enumerator::BackTrack_() {
  bool fsbl = true;
  SchedInstruction *inst = crntNode_->GetInst();
  EnumTreeNode *trgtNode = crntNode_->GetParent();

  rdyLst_->RemoveLatestSubList();

  if (IsHistDom()) {
    assert(!crntNode_->IsArchived());
    HistEnumTreeNode *crntHstry = crntNode_->GetHistory();
    exmndSubProbs_->InsertElement(crntNode_->GetSig(), crntHstry,
                                  hashTblEntryAlctr_);
    SetTotalCostsAndSuffixes(crntNode_, trgtNode, trgtSchedLngth_,
                             prune_.useSuffixConcatenation);
    crntNode_->Archive();
  } else {
    assert(crntNode_->IsArchived() == false);
  }

  nodeAlctr_->Free(crntNode_);

  EnumTreeNode *prevNode = crntNode_;
  crntNode_ = trgtNode;
  rdyLst_ = crntNode_->GetRdyLst();
  assert(rdyLst_ != NULL);

  MovToPrevSlot_(crntNode_->GetRealSlotNum());

  trgtNode->NewBranchExmnd(inst, true, false, false, crntNode_->IsFeasible(),
                           DIR_BKWRD, prevNode->IsLngthFsbl());

#ifdef IS_DEBUG_FLOW
  InstCount instNum = inst == NULL ? SCHD_STALL : inst->GetNum();
  Logger::Info("Backtracking from node %lld to node %lld by unscheduling inst. "
               "#%d in cycle #%d. CostLB=%d",
               prevNode->GetNum(), trgtNode->GetNum(), instNum, crntCycleNum_,
               trgtNode->GetCostLwrBound());
#endif

  crntNode_->GetSlotAvlblty(avlblSlots_, avlblSlotsInCrntCycle_);
  isCrntCycleBlkd_ = crntNode_->GetCrntCycleBlkd();

  if (inst != NULL) {
    IssueType issuType = inst->GetIssueType();
    neededSlots_[issuType]++;
  }

  crntSched_->RemoveLastInst();
  RestoreCrntLwrBounds_(inst);

  if (inst != NULL) {
    // int hitCnt;
    // assert(rdyLst_->FindInst(inst, hitCnt) && hitCnt == 1);
    assert(inst->IsInReadyList());

    UndoRsrvSlots_(inst);
    UnSchdulInst_(inst);
    inst->UnSchedule();

    if (inst->GetTplgclOrdr() == minUnschduldTplgclOrdr_ - 1) {
      minUnschduldTplgclOrdr_--;
    }
  }

  backTrackCnt_++;
  return fsbl;
}
/*****************************************************************************/

bool Enumerator::WasDmnntSubProbExmnd_(SchedInstruction *,
                                       EnumTreeNode *&newNode) {
#ifdef IS_DEBUG_SPD
  stats::signatureDominationTests++;
#endif
  HistEnumTreeNode *exNode;
  int listSize = exmndSubProbs_->GetListSize(newNode->GetSig());
  int trvrsdListSize = 0;
  stats::historyListSize.Record(listSize);
  mostRecentMatchingHistNode_ = nullptr;
  bool mostRecentMatchWasSet = false;

  for (exNode = exmndSubProbs_->GetLastMatch(newNode->GetSig()); exNode != NULL;
       exNode = exmndSubProbs_->GetPrevMatch()) {
    trvrsdListSize++;
#ifdef IS_DEBUG_SPD
    stats::signatureMatches++;
#endif

    if (exNode->DoesMatch(newNode, this)) {
      if (!mostRecentMatchWasSet) {
        mostRecentMatchingHistNode_ =
            (exNode->GetSuffix() != nullptr) ? exNode : nullptr;
        mostRecentMatchWasSet = true;
      }
      if (exNode->DoesDominate(newNode, this)) {

#ifdef IS_DEBUG_SPD
        Logger::Info("Node %d is dominated. Partial scheds:",
                     newNode->GetNum());
        Logger::Info("Current node:");
        newNode->PrntPartialSched(Logger::GetLogStream());
        Logger::Info("Hist node:");
        exNode->PrntPartialSched(Logger::GetLogStream());
#endif

        nodeAlctr_->Free(newNode);
        newNode = NULL;
#ifdef IS_DEBUG_SPD
        stats::positiveDominationHits++;
        stats::traversedHistoryListSize.Record(trvrsdListSize);
        stats::historyDominationPosition.Record(trvrsdListSize);
        stats::historyDominationPositionToListSize.Record(
            (trvrsdListSize * 100) / listSize);
#endif
        return true;
      } else {
#ifdef IS_DEBUG_SPD
        stats::signatureAliases++;
#endif
      }
    }
  }

  stats::traversedHistoryListSize.Record(trvrsdListSize);
  return false;
}
/****************************************************************************/

bool Enumerator::TightnLwrBounds_(SchedInstruction *newInst) {
  SchedInstruction *inst;
  InstCount newLwrBound = 0;
  InstCount nxtAvlblCycle[MAX_ISSUTYPE_CNT];
  bool fsbl;
  InstCount i;

  assert(fxdLst_->GetElmntCnt() == 0);
  assert(tightndLst_->GetElmntCnt() == 0);

  for (i = 0; i < issuTypeCnt_; i++) {
    // If this slot is filled with a stall then all subsequent slots are
    // going to be filled with stalls
    if (newInst == NULL) {
      nxtAvlblCycle[i] = crntCycleNum_ + 1;
    } else {
      // If the last slot for this type has been taken in this cycle
      // then an inst. of this type cannot issue any earlier than the
      // next cycle
      nxtAvlblCycle[i] =
          avlblSlotsInCrntCycle_[i] == 0 ? crntCycleNum_ + 1 : crntCycleNum_;
    }
  }

  for (i = minUnschduldTplgclOrdr_; i < totInstCnt_; i++) {
    inst = dataDepGraph_->GetInstByTplgclOrdr(i);
    assert(inst != newInst ||
           inst->GetCrntLwrBound(DIR_FRWRD) == crntCycleNum_);

    if (inst->IsSchduld() == false) {
      IssueType issuType = inst->GetIssueType();
      newLwrBound = nxtAvlblCycle[issuType];

      if (newLwrBound > inst->GetCrntLwrBound(DIR_FRWRD)) {
#ifdef IS_DEBUG_FLOW
        Logger::Info("Tightening LB of inst %d from %d to %d", inst->GetNum(),
                     inst->GetCrntLwrBound(DIR_FRWRD), newLwrBound);
#endif
        fsbl = inst->TightnLwrBoundRcrsvly(DIR_FRWRD, newLwrBound, tightndLst_,
                                           fxdLst_, false);

        if (fsbl == false) {
          return false;
        }
      }

      assert(inst->GetCrntLwrBound(DIR_FRWRD) >= newLwrBound);

      if (inst->GetCrntLwrBound(DIR_FRWRD) > inst->GetCrntDeadline()) {
        return false;
      }
    }
  }

  for (inst = tightndLst_->GetFrstElmnt(); inst != NULL;
       inst = tightndLst_->GetNxtElmnt()) {
    dataDepGraph_->SetCrntFrwrdLwrBound(inst);
  }

  return FixInsts_(newInst);
}
/****************************************************************************/

void Enumerator::UnTightnLwrBounds_(SchedInstruction *newInst) {
  UnFixInsts_(newInst);

  SchedInstruction *inst;

  for (inst = tightndLst_->GetFrstElmnt(); inst != NULL;
       inst = tightndLst_->GetNxtElmnt()) {
    inst->UnTightnLwrBounds();
    dataDepGraph_->SetCrntFrwrdLwrBound(inst);
    assert(inst->IsFxd() == false);
  }

  tightndLst_->Reset();
  dirctTightndLst_->Reset();
}
/*****************************************************************************/

void Enumerator::CmtLwrBoundTightnng_() {
  SchedInstruction *inst;

  for (inst = tightndLst_->GetFrstElmnt(); inst != NULL;
       inst = tightndLst_->GetNxtElmnt()) {
    inst->CmtLwrBoundTightnng();
  }

  tightndLst_->Reset();
  dirctTightndLst_->Reset();
  CmtInstFxng_();
}
/*****************************************************************************/

bool Enumerator::FixInsts_(SchedInstruction *newInst) {
  bool fsbl = true;

  bool newInstFxd = false;

  fxdInstCnt_ = 0;

  for (SchedInstruction *inst = fxdLst_->GetFrstElmnt(); inst != NULL;
       inst = fxdLst_->GetNxtElmnt()) {
    assert(inst->IsFxd());
    assert(inst->IsSchduld() == false || inst == newInst);
    fsbl = rlxdSchdulr_->FixInst(inst, inst->GetFxdCycle());

    if (inst == newInst) {
      newInstFxd = true;
      assert(inst->GetFxdCycle() == crntCycleNum_);
    }

    if (fsbl == false) {
#ifdef IS_DEBUG_FLOW
      Logger::Info("Can't fix inst %d in cycle %d", inst->GetNum(),
                   inst->GetFxdCycle());
#endif
      break;
    }

    fxdInstCnt_++;
  }

  if (fsbl)
    if (!newInstFxd && newInst != NULL) {
      if (newInst->IsFxd() == false)
      // We need to fix the new inst. only if it has not been fixed before
      {
        fsbl = rlxdSchdulr_->FixInst(newInst, crntCycleNum_);

        if (fsbl) {
          fxdLst_->InsrtElmnt(newInst);
          fxdInstCnt_++;
        }
      }
    }

  return fsbl;
}
/*****************************************************************************/

void Enumerator::UnFixInsts_(SchedInstruction *newInst) {
  InstCount unfxdInstCnt = 0;
  SchedInstruction *inst;

  for (inst = fxdLst_->GetFrstElmnt(), unfxdInstCnt = 0;
       inst != NULL && unfxdInstCnt < fxdInstCnt_;
       inst = fxdLst_->GetNxtElmnt(), unfxdInstCnt++) {
    assert(inst->IsFxd() || inst == newInst);
    InstCount cycle = inst == newInst ? crntCycleNum_ : inst->GetFxdCycle();
    rlxdSchdulr_->UnFixInst(inst, cycle);
  }

  assert(unfxdInstCnt == fxdInstCnt_);
  fxdLst_->Reset();
  fxdInstCnt_ = 0;
}
/*****************************************************************************/

void Enumerator::CmtInstFxng_() {
  fxdLst_->Reset();
  fxdInstCnt_ = 0;
}
/*****************************************************************************/

void Enumerator::RestoreCrntLwrBounds_(SchedInstruction *unschduldInst) {
  InstCount *frwrdLwrBounds = crntNode_->GetLwrBounds(DIR_FRWRD);
  bool unschduldInstDone = false;

  for (InstCount i = 0; i < totInstCnt_; i++) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(i);
    InstCount fxdCycle = 0;
    bool preFxd = inst->IsFxd();

    if (preFxd) {
      fxdCycle = inst->GetFxdCycle();
    }

    inst->SetCrntLwrBound(DIR_FRWRD, frwrdLwrBounds[i]);
    dataDepGraph_->SetCrntFrwrdLwrBound(inst);
    bool postFxd = inst->IsFxd();

    if (preFxd && !postFxd) { // if got untightened and unfixed
      rlxdSchdulr_->UnFixInst(inst, fxdCycle);

      if (inst == unschduldInst) {
        unschduldInstDone = true;
      }
    }
  }

  if (unschduldInst != NULL && !unschduldInstDone) {
    // Assume that the instruction has not been unscheduled yet
    // i.e. lower bound restoration occurs before unscheduling
    assert(unschduldInst->IsSchduld());

    if (unschduldInst->IsFxd() == false)
    // only if the untightening got it unfixed
    {
      rlxdSchdulr_->UnFixInst(unschduldInst, unschduldInst->GetSchedCycle());
    }
  }
}
/*****************************************************************************/

bool Enumerator::RlxdSchdul_(EnumTreeNode *newNode) {
  assert(newNode != NULL);
  LinkedList<SchedInstruction> *rsrcFxdLst = new LinkedList<SchedInstruction>;

  bool fsbl =
      rlxdSchdulr_->SchdulAndChkFsblty(crntCycleNum_, trgtSchedLngth_ - 1);

  for (SchedInstruction *inst = rsrcFxdLst->GetFrstElmnt(); inst != NULL;
       inst = rsrcFxdLst->GetNxtElmnt()) {
    assert(inst->IsSchduld() == false);
    fsbl = rlxdSchdulr_->FixInst(inst, inst->GetCrntLwrBound(DIR_FRWRD));

    if (fsbl == false) {
      return false;
    }

    fxdLst_->InsrtElmnt(inst);
    fxdInstCnt_++;
#ifdef IS_DEBUG_FIX
    Logger::Info("%d [%d], ", inst->GetNum(), inst->GetFxdCycle());
#endif
  }

  assert(rsrcFxdLst->GetElmntCnt() == 0);
  rsrcFxdLst->Reset();
  delete rsrcFxdLst;
  return fsbl;
}
/*****************************************************************************/

bool Enumerator::IsUseInRdyLst_() {
  assert(rdyLst_ != NULL);
  bool isEmptyNode = false;
  InstCount brnchCnt = crntNode_->GetBranchCnt(isEmptyNode);
  SchedInstruction *inst;
  bool foundUse = false;

#ifdef IS_DEBUG_RP_ONLY
  Logger::Info("Looking for a use in the ready list with nodes:");
  for (int i = 0; i < brnchCnt - 1; i++) {
    inst = rdyLst_->GetNextPriorityInst();
    assert(inst != NULL);
    Logger::Info("#%d:%d", i, inst->GetNum());
  }
  rdyLst_->ResetIterator();
#endif

  for (int i = 0; i < brnchCnt - 1; i++) {
    inst = rdyLst_->GetNextPriorityInst();
    assert(inst != NULL);
    if (inst->GetAdjustedUseCnt() != 0 || dataDepGraph_->DoesFeedUser(inst)) {
      foundUse = true;
#ifdef IS_DEBUG_RP_ONLY
      Logger::Info("Inst %d uses a register", inst->GetNum());
#endif
      break;
    }
#ifdef IS_DEBUG_RP_ONLY
    Logger::Info("Inst %d does not use a register", inst->GetNum());
#endif
  }

  rdyLst_->ResetIterator();
  return foundUse;
}
/*****************************************************************************/

void Enumerator::PrintLog_() {
  Logger::Info("--------------------------------------------------\n");

  Logger::Info("Total nodes examined: %lld\n", GetNodeCnt());
  Logger::Info("History table includes %d entries.\n",
               exmndSubProbs_->GetEntryCnt());
  Logger::GetLogStream() << stats::historyEntriesPerIteration;
  Logger::Info("--------------------------------------------------\n");
}
/*****************************************************************************/

bool Enumerator::EnumStall_() { return enblStallEnum_; }
/*****************************************************************************/

LengthEnumerator::LengthEnumerator(
    DataDepGraph *dataDepGraph, MachineModel *machMdl, InstCount schedUprBound,
    int16_t sigHashSize, SchedPriorities prirts, Pruning PruningStrategy,
    bool SchedForRPOnly, bool enblStallEnum, Milliseconds timeout,
    InstCount preFxdInstCnt, SchedInstruction *preFxdInsts[])
    : Enumerator(dataDepGraph, machMdl, schedUprBound, sigHashSize, prirts,
                 PruningStrategy, SchedForRPOnly, enblStallEnum, timeout,
                 preFxdInstCnt, preFxdInsts) {
  SetupAllocators_();
  tmpHstryNode_ = new HistEnumTreeNode;
}
/*****************************************************************************/

LengthEnumerator::~LengthEnumerator() {
  Reset();
  FreeAllocators_();
}
/*****************************************************************************/

void LengthEnumerator::SetupAllocators_() {
  int memAllocBlkSize = memAllocBlkSize_;

  Enumerator::SetupAllocators_();

  if (IsHistDom()) {
    histNodeAlctr_ = new MemAlloc<HistEnumTreeNode>(memAllocBlkSize);
  }
}
/****************************************************************************/

void LengthEnumerator::ResetAllocators_() {
  Enumerator::ResetAllocators_();
  if (IsHistDom())
    histNodeAlctr_->Reset();
}
/****************************************************************************/

void LengthEnumerator::FreeAllocators_() {
  Enumerator::FreeAllocators_();

  if (IsHistDom()) {
    delete histNodeAlctr_;
    histNodeAlctr_ = NULL;
  }
}
/****************************************************************************/

bool LengthEnumerator::IsCostEnum() { return false; }
/*****************************************************************************/

FUNC_RESULT LengthEnumerator::FindFeasibleSchedule(InstSchedule *sched,
                                                   InstCount trgtLngth,
                                                   Milliseconds deadline) {
  return FindFeasibleSchedule_(sched, trgtLngth, deadline);
}
/*****************************************************************************/

void LengthEnumerator::Reset() { Enumerator::Reset(); }
/*****************************************************************************/

bool LengthEnumerator::WasObjctvMet_() {
  bool wasSlonFound = WasSolnFound_();

  return wasSlonFound;
}
/*****************************************************************************/

HistEnumTreeNode *LengthEnumerator::AllocHistNode_(EnumTreeNode *node) {
  HistEnumTreeNode *histNode = histNodeAlctr_->GetObject();
  histNode->Construct(node, false);
  return histNode;
}
/*****************************************************************************/

HistEnumTreeNode *LengthEnumerator::AllocTempHistNode_(EnumTreeNode *node) {
  HistEnumTreeNode *histNode = tmpHstryNode_;
  histNode->Construct(node, true);
  return histNode;
}
/*****************************************************************************/

void LengthEnumerator::FreeHistNode_(HistEnumTreeNode *histNode) {
  histNode->Clean();
  histNodeAlctr_->FreeObject(histNode);
}
/*****************************************************************************/

LengthCostEnumerator::LengthCostEnumerator(
    DataDepGraph *dataDepGraph, MachineModel *machMdl, InstCount schedUprBound,
    int16_t sigHashSize, SchedPriorities prirts, Pruning PruningStrategy,
    bool SchedForRPOnly, bool enblStallEnum, Milliseconds timeout,
    SPILL_COST_FUNCTION spillCostFunc, InstCount preFxdInstCnt,
    SchedInstruction *preFxdInsts[])
    : Enumerator(dataDepGraph, machMdl, schedUprBound, sigHashSize, prirts,
                 PruningStrategy, SchedForRPOnly, enblStallEnum, timeout,
                 preFxdInstCnt, preFxdInsts) {
  SetupAllocators_();

  costChkCnt_ = 0;
  costPruneCnt_ = 0;
  isEarlySubProbDom_ = false;
  costLwrBound_ = 0;
  spillCostFunc_ = spillCostFunc;
  tmpHstryNode_ = new CostHistEnumTreeNode;
}
/*****************************************************************************/

LengthCostEnumerator::~LengthCostEnumerator() {
  Reset();
  FreeAllocators_();
}
/*****************************************************************************/

void LengthCostEnumerator::SetupAllocators_() {
  int memAllocBlkSize = memAllocBlkSize_;

  Enumerator::SetupAllocators_();

  if (IsHistDom()) {
    histNodeAlctr_ = new MemAlloc<CostHistEnumTreeNode>(memAllocBlkSize);
  }
}
/****************************************************************************/

void LengthCostEnumerator::ResetAllocators_() {
  Enumerator::ResetAllocators_();
  if (IsHistDom())
    histNodeAlctr_->Reset();
}
/****************************************************************************/

void LengthCostEnumerator::FreeAllocators_() {
  Enumerator::FreeAllocators_();

  if (IsHistDom()) {
    delete histNodeAlctr_;
    histNodeAlctr_ = NULL;
  }
}
/****************************************************************************/

bool LengthCostEnumerator::IsCostEnum() { return true; }
/*****************************************************************************/

void LengthCostEnumerator::Reset() { Enumerator::Reset(); }
/*****************************************************************************/

bool LengthCostEnumerator::Initialize_(InstSchedule *preSched,
                                       InstCount trgtLngth) {
  bool fsbl = Enumerator::Initialize_(preSched, trgtLngth);

  if (fsbl == false) {
    return false;
  }

  costChkCnt_ = 0;
  costPruneCnt_ = 0;
  return true;
}
/*****************************************************************************/

FUNC_RESULT LengthCostEnumerator::FindFeasibleSchedule(InstSchedule *sched,
                                                       InstCount trgtLngth,
                                                       SchedRegion *rgn,
                                                       int costLwrBound,
                                                       Milliseconds deadline) {
  rgn_ = rgn;
  costLwrBound_ = costLwrBound;
  FUNC_RESULT rslt = FindFeasibleSchedule_(sched, trgtLngth, deadline);

#ifdef IS_DEBUG_TRACE_ENUM
  stats::costChecksPerLength.Record(costChkCnt_);
  stats::costPruningsPerLength.Record(costPruneCnt_);
  stats::feasibleSchedulesPerLength.Record(fsblSchedCnt_);
  stats::improvementsPerLength.Record(imprvmntCnt_);
#endif

  return rslt;
}
/*****************************************************************************/

bool LengthCostEnumerator::WasObjctvMet_() {
  assert(GetBestCost_() >= 0);

  if (WasSolnFound_() == false) {
    return false;
  }

  InstCount crntCost = GetBestCost_();

  InstCount newCost = rgn_->UpdtOptmlSched(crntSched_, this);
  assert(newCost <= GetBestCost_());

  if (newCost < crntCost) {
    imprvmntCnt_++;
  }

  return newCost == costLwrBound_;
}
/*****************************************************************************/

bool LengthCostEnumerator::ProbeBranch_(SchedInstruction *inst,
                                        EnumTreeNode *&newNode,
                                        bool &isNodeDmntd, bool &isRlxInfsbl,
                                        bool &isLngthFsbl) {
  bool isFsbl = true;

  isFsbl = Enumerator::ProbeBranch_(inst, newNode, isNodeDmntd, isRlxInfsbl,
                                    isLngthFsbl);

  if (isFsbl == false) {
    assert(isLngthFsbl == false);
    isLngthFsbl = false;
    return false;
  }

  isLngthFsbl = true;

  isFsbl = ChkCostFsblty_(inst, newNode);

  if (isFsbl == false) {
    return false;
  }

  if (IsHistDom()) {
    assert(newNode != NULL);
    EnumTreeNode *parent = newNode->GetParent();

    if (WasDmnntSubProbExmnd_(inst, newNode)) {
#ifdef IS_DEBUG_FLOW
      Logger::Info("History domination\n\n");
#endif

#ifdef IS_DEBUG_INFSBLTY_TESTS
      stats::historyDominationInfeasibilityHits++;
#endif
      rgn_->UnschdulInst(inst, crntCycleNum_, crntSlotNum_, parent);

      return false;
    }
  }

  return true;
}
/*****************************************************************************/

bool LengthCostEnumerator::ChkCostFsblty_(SchedInstruction *inst,
                                          EnumTreeNode *&newNode) {
  bool isFsbl = true;

  costChkCnt_++;

  rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);

  if (prune_.spillCost) {
    isFsbl = rgn_->ChkCostFsblty(trgtSchedLngth_, newNode);

    if (!isFsbl) {
      costPruneCnt_++;
#ifdef IS_DEBUG_FLOW
      Logger::Info("Detected cost infeasibility of inst %d in cycle %d",
                   inst == NULL ? -2 : inst->GetNum(), crntCycleNum_);
#endif
      rgn_->UnschdulInst(inst, crntCycleNum_, crntSlotNum_,
                         newNode->GetParent());
    }
  }

  return isFsbl;
}
/*****************************************************************************/

bool LengthCostEnumerator::BackTrack_() {
  SchedInstruction *inst = crntNode_->GetInst();

  rgn_->UnschdulInst(inst, crntCycleNum_, crntSlotNum_, crntNode_->GetParent());

  bool fsbl = Enumerator::BackTrack_();

  if (prune_.spillCost) {
    if (fsbl) {
      assert(crntNode_->GetCostLwrBound() >= 0);
      fsbl = crntNode_->GetCostLwrBound() < GetBestCost_();
    }
  }

  return fsbl;
}
/*****************************************************************************/

InstCount LengthCostEnumerator::GetBestCost_() { return rgn_->GetBestCost(); }
/*****************************************************************************/

void LengthCostEnumerator::CreateRootNode_() {
  rootNode_ = nodeAlctr_->Alloc(NULL, NULL, this);
  CreateNewRdyLst_();
  rootNode_->SetRdyLst(rdyLst_);
  rootNode_->SetLwrBounds(DIR_FRWRD);

  assert(rsrvSlotCnt_ == 0);
  rootNode_->SetRsrvSlots(rsrvSlotCnt_, rsrvSlots_);

  rgn_->SetSttcLwrBounds(rootNode_);

  rootNode_->SetCost(0);
  rootNode_->SetCostLwrBound(0);

  InitNewNode_(rootNode_);
  CmtLwrBoundTightnng_();
}
/*****************************************************************************/

bool LengthCostEnumerator::EnumStall_() {
  // Logger::Info("enblStallEnum_ = %d", enblStallEnum_);
  if (!enblStallEnum_)
    return false;
  if (crntNode_->IsNxtSlotStall())
    return true;
  if (crntNode_ == rootNode_)
    return false;
  if (dataDepGraph_->IncludesUnpipelined())
    return true;
  //  return false;
  return true;
}
/*****************************************************************************/

void LengthCostEnumerator::InitNewNode_(EnumTreeNode *newNode) {
  Enumerator::InitNewNode_(newNode);
}
/*****************************************************************************/

HistEnumTreeNode *LengthCostEnumerator::AllocHistNode_(EnumTreeNode *node) {
  CostHistEnumTreeNode *histNode = histNodeAlctr_->GetObject();
  histNode->Construct(node, false);
  return histNode;
}
/*****************************************************************************/

HistEnumTreeNode *LengthCostEnumerator::AllocTempHistNode_(EnumTreeNode *node) {
  HistEnumTreeNode *histNode = tmpHstryNode_;
  histNode->Construct(node, true);
  return histNode;
}
/*****************************************************************************/

void LengthCostEnumerator::FreeHistNode_(HistEnumTreeNode *histNode) {
  histNode->Clean();
  histNodeAlctr_->FreeObject((CostHistEnumTreeNode *)histNode);
}
/*****************************************************************************/
