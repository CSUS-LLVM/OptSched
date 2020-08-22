#include "opt-sched/Scheduler/hist_table.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/utilities.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

using namespace llvm::opt_sched;

HistEnumTreeNode::HistEnumTreeNode() { rsrvSlots_ = NULL; }

HistEnumTreeNode::~HistEnumTreeNode() {
  if (rsrvSlots_)
    delete[] rsrvSlots_;
}

void HistEnumTreeNode::Construct(EnumTreeNode *node, bool) {
  prevNode_ = node->prevNode_ == NULL ? NULL : node->prevNode_->hstry_;
  assert(prevNode_ != this);

  time_ = node->time_;
  inst_ = node->inst_;

#ifdef IS_DEBUG
  isCnstrctd_ = true;
#endif

  crntCycleBlkd_ = node->crntCycleBlkd_;
  suffix_ = nullptr;
  SetRsrvSlots_(node);
}

void HistEnumTreeNode::SetRsrvSlots_(EnumTreeNode *node) {
  if (rsrvSlots_ != NULL)
    delete[] rsrvSlots_;
  rsrvSlots_ = NULL;

  // If no unpipelined instrs are scheduled.
  if (node->rsrvSlots_ == NULL)
    return;

  int issuRate = node->enumrtr_->machMdl_->GetIssueRate();

  rsrvSlots_ = new ReserveSlot[issuRate];

  for (int i = 0; i < issuRate; i++) {
    rsrvSlots_[i].strtCycle = node->rsrvSlots_[i].strtCycle;
    rsrvSlots_[i].endCycle = node->rsrvSlots_[i].endCycle;
  }
}

void HistEnumTreeNode::Init_() {
  time_ = 0;
  inst_ = NULL;
  prevNode_ = NULL;
#ifdef IS_DEBUG
  isCnstrctd_ = false;
#endif
  crntCycleBlkd_ = false;
  rsrvSlots_ = NULL;
}

void HistEnumTreeNode::Clean() {
  if (rsrvSlots_) {
    delete[] rsrvSlots_;
    rsrvSlots_ = NULL;
  }
}

InstCount HistEnumTreeNode::SetLastInsts_(SchedInstruction *lastInsts[],
                                          InstCount thisTime,
                                          InstCount minTimeToExmn) {
  assert(minTimeToExmn >= 1);
  assert(lastInsts != NULL);

  HistEnumTreeNode *crntNode;
  InstCount indx;
  InstCount time;
  for (crntNode = this, time = thisTime, indx = 0; time >= minTimeToExmn;
       crntNode = crntNode->prevNode_, indx++, time--) {
    // Trace back the predecessors of the node to examine all the
    // instructions in its partial schedule
    assert(crntNode->prevNode_ != NULL);
    assert(crntNode->GetTime() == thisTime - indx);
    SchedInstruction *inst = crntNode->inst_;
    assert(indx < (thisTime - minTimeToExmn + 1));
    lastInsts[indx] = inst;
  }

  return indx;
}

void HistEnumTreeNode::SetInstsSchduld_(BitVector *instsSchduld) {
  instsSchduld->Reset();
  HistEnumTreeNode *crntNode;

  for (crntNode = this; crntNode != NULL; crntNode = crntNode->prevNode_) {
    SchedInstruction *inst = crntNode->inst_;

    if (inst != NULL) {
      assert(!instsSchduld->GetBit(inst->GetNum()));
      instsSchduld->SetBit(inst->GetNum());
    }
  }
}

void HistEnumTreeNode::SetLwrBounds_(InstCount lwrBounds[],
                                     SchedInstruction *lastInsts[],
                                     InstCount thisTime,
                                     InstCount minTimeToExmn,
                                     Enumerator *enumrtr) {
  InstCount instCnt = enumrtr->totInstCnt_;

  for (InstCount i = 0; i < instCnt; i++) {
    lwrBounds[i] = 0;
  }

  InstCount entryCnt = SetLastInsts_(lastInsts, thisTime, minTimeToExmn);

  for (InstCount indx = 0; indx < entryCnt; indx++) {
    InstCount time = thisTime - indx;
    InstCount cycleNum = enumrtr->GetCycleNumFrmTime_(time);
    SchedInstruction *inst = lastInsts[indx];

    // If an instruction is scheduled after its static lower bound then its
    // successors will potentially be pushed down and should be checked.
    if (inst != NULL && cycleNum > inst->GetLwrBound(DIR_FRWRD)) {
      UDT_GLABEL ltncy;
      DependenceType depType;

      // Examine all the unscheduled successors of this instruction
      // to see if any of them is pushed down.
      for (SchedInstruction *scsr = inst->GetFrstScsr(NULL, &ltncy, &depType);
           scsr != NULL; scsr = inst->GetNxtScsr(NULL, &ltncy, &depType)) {
        if (scsr->IsSchduld() == false) {
          InstCount num = scsr->GetNum();
          InstCount thisBound = cycleNum + ltncy;
          if (thisBound > lwrBounds[num])
            lwrBounds[num] = thisBound;
        }
      }
    }
  }
}

InstCount HistEnumTreeNode::GetMinTimeToExmn_(InstCount nodeTime,
                                              Enumerator *enumrtr) {
  int issuRate = enumrtr->issuRate_;
  DataDepGraph *dataDepGraph = enumrtr->dataDepGraph_;
  UDT_GLABEL maxLtncy = dataDepGraph->GetMaxLtncy();
  InstCount crntCycleNum = enumrtr->GetCycleNumFrmTime_(nodeTime);
  InstCount nxtCycleNum = crntCycleNum + 1;
  InstCount minCycleNumToExmn = std::max(nxtCycleNum - maxLtncy, 0);
  InstCount minTimeToExmn = minCycleNumToExmn * issuRate + 1;
  return minTimeToExmn;
}

bool HistEnumTreeNode::DoesDominate_(EnumTreeNode *node,
                                     HistEnumTreeNode *othrHstry,
                                     ENUMTREE_NODEMODE mode,
                                     Enumerator *enumrtr, InstCount shft) {
  InstCount indx, time;
  InstCount *othrLwrBounds = NULL;
  InstCount thisTime, othrTime;
  SchedInstruction **lastInsts = enumrtr->lastInsts_;
  SchedInstruction **othrLastInsts = enumrtr->othrLastInsts_;
  InstCount *instsPerType = enumrtr->histInstsPerType_;
  InstCount *nxtAvlblCycles = enumrtr->histNxtAvlblCycles_;
  bool othrCrntCycleBlkd;

  assert(othrHstry != this);
  thisTime = GetTime();

  if (mode == ETN_ACTIVE) {
    assert(node != NULL && othrHstry == NULL);
    othrLwrBounds = node->frwrdLwrBounds_;
    othrTime = node->GetTime();
    othrCrntCycleBlkd = node->crntCycleBlkd_;
  } else {
    assert(mode == ETN_HISTORY);
    assert(othrHstry != NULL && node == NULL && enumrtr != NULL);
    othrLwrBounds = enumrtr->tmpLwrBounds_;
    othrTime = othrHstry->GetTime();
    othrCrntCycleBlkd = othrHstry->crntCycleBlkd_;
  }

  // We cannot make a decision about domination if the candidate dominant
  // node lies deeper in the enumeration tree than the node in question.
  if (thisTime > othrTime)
    return false;
    //  if (thisTime != othrTime) return false;

#ifdef IS_DEBUG_SPD
  if (thisTime < othrTime)
    stats::subsetMatches++;
#endif

  if (othrCrntCycleBlkd != crntCycleBlkd_)
    return false;

  if (rsrvSlots_ != NULL) {
    if (node->rsrvSlots_ == NULL)
      return false;

    int issuRate = node->enumrtr_->machMdl_->GetIssueRate();
    for (int i = 0; i < issuRate; i++) {
      if (rsrvSlots_[i].strtCycle != INVALID_VALUE) {
        if (node->rsrvSlots_[i].strtCycle == INVALID_VALUE ||
            rsrvSlots_[i].endCycle > node->rsrvSlots_[i].endCycle)
          return false;
      }
    }
  }

  InstCount entryCnt;
  InstCount minTimeToExmn = GetMinTimeToExmn_(thisTime, enumrtr);

  entryCnt = SetLastInsts_(lastInsts, thisTime, minTimeToExmn);
  assert(entryCnt == thisTime - minTimeToExmn + 1);

  assert(lastInsts != NULL);
  bool isAbslutDmnnt = true;

  if (othrHstry != NULL) {
    othrHstry->SetLwrBounds_(othrLwrBounds, othrLastInsts, othrTime,
                             minTimeToExmn, enumrtr);
  }

  CmputNxtAvlblCycles_(enumrtr, instsPerType, nxtAvlblCycles);

  for (indx = 0; indx < entryCnt; indx++) {
    time = thisTime - indx;
    InstCount cycleNum = enumrtr->GetCycleNumFrmTime_(time);
    SchedInstruction *inst = lastInsts[indx];

    // If an inst. is scheduled after its static lower bound then its
    // successors will potentially be pushed down and should be checked.
    if (inst != NULL && (cycleNum > inst->GetLwrBound(DIR_FRWRD) || shft > 0)) {
      UDT_GLABEL ltncy;
      DependenceType depType;

      // Examine all the unscheduled successors of this instruction to see if
      // any of them is pushed down.
      for (SchedInstruction *scsr = inst->GetFrstScsr(NULL, &ltncy, &depType);
           scsr != NULL; scsr = inst->GetNxtScsr(NULL, &ltncy, &depType)) {
        if (scsr->IsSchduld() == false) {
          InstCount nxtAvlblCycle = nxtAvlblCycles[scsr->GetIssueType()];
          InstCount num = scsr->GetNum();
          InstCount thisBound = cycleNum + ltncy;
          thisBound = std::max(thisBound, nxtAvlblCycle);
          InstCount sttcBound = scsr->GetLwrBound(DIR_FRWRD);
          InstCount normBound = std::max(sttcBound, nxtAvlblCycle);

          if (thisBound > normBound || shft > 0) {
            isAbslutDmnnt = false;
            InstCount othrBound = othrLwrBounds[num];

            if ((thisBound + shft) > othrBound)
              return false;
          }
        }
      }
    } else {
      // If this inst. is scheduled at its static lower bound it cannot
      // push down any successors. Therefore it will be safe to skip it in
      // future tests
      lastInsts[indx] = NULL;
    }
  }

  // If this node is an absolute dominant that dominates any matching node.
  if (isAbslutDmnnt)
    stats::absoluteDominationHits++;

  // PrntPartialSched(enumrtr);
  return true;
}

void HistEnumTreeNode::CmputNxtAvlblCycles_(Enumerator *enumrtr,
                                            InstCount instsPerType[],
                                            InstCount nxtAvlblCycles[]) {
  InstCount thisTime = GetTime();
  InstCount crntCycle = enumrtr->GetCycleNumFrmTime_(thisTime);
  HistEnumTreeNode *crntNode;
  InstCount time;
  InstCount cycleNum = crntCycle;

  MachineModel *machMdl = enumrtr->machMdl_;
  int issuTypeCnt = machMdl->GetIssueTypeCnt();

  for (int i = 0; i < issuTypeCnt; i++) {
    instsPerType[i] = 0;
    nxtAvlblCycles[i] = crntCycle;
  }

  for (crntNode = this, time = thisTime;
       crntNode != NULL && cycleNum == crntCycle;
       crntNode = crntNode->prevNode_, time--) {
    assert(crntNode->prevNode_ != NULL);
    SchedInstruction *inst = crntNode->inst_;
    cycleNum = enumrtr->GetCycleNumFrmTime_(time);

    if (inst == NULL)
      continue;

    IssueType issuType = inst->GetIssueType();
    assert(issuType < issuTypeCnt);
    instsPerType[issuType]++;

    if (instsPerType[issuType] == machMdl->GetSlotsPerCycle(issuType)) {
      nxtAvlblCycles[issuType] = crntCycle + 1;
    }
  }
}

bool HistEnumTreeNode::DoesDominate(EnumTreeNode *node, Enumerator *enumrtr) {
#ifdef IS_DEBUG
  assert(isCnstrctd_);
#endif
  InstCount shft = 0;
  return DoesDominate_(node, NULL, ETN_ACTIVE, enumrtr, shft);
}

void HistEnumTreeNode::PrntPartialSched(std::ostream &out) {
  out << "\nPartial sched. at time " << GetTime() << " (add=" << (void *)this
      << "): ";

  for (HistEnumTreeNode *node = this; node != NULL; node = node->GetParent()) {
    InstCount instNum =
        (node->inst_ == NULL) ? SCHD_STALL : node->inst_->GetNum();
    out << instNum << ' ';
  }
}

bool HistEnumTreeNode::CompPartialScheds(HistEnumTreeNode *othrHist) {
  InstCount thisTime = GetTime();
  InstCount othrTime = othrHist->GetTime();

  if (thisTime != othrTime)
    return false;

  for (HistEnumTreeNode *node = this, *othrNode = othrHist; node != NULL;
       node = node->GetParent(), othrNode = othrNode->GetParent()) {
    InstCount thisInstNum =
        node->inst_ == NULL ? SCHD_STALL : node->inst_->GetNum();
    InstCount othrInstNum =
        othrNode->inst_ == NULL ? SCHD_STALL : othrNode->inst_->GetNum();

    if (thisInstNum != othrInstNum)
      return false;
  }

  return true;
}

void HistEnumTreeNode::SetCostInfo(EnumTreeNode *, bool, Enumerator *) {
  // Nothing.
}

const std::shared_ptr<std::vector<SchedInstruction *>> &
HistEnumTreeNode::GetSuffix() const {
  return suffix_;
}

void HistEnumTreeNode::SetSuffix(
    const std::shared_ptr<std::vector<SchedInstruction *>> &suffix) {
  suffix_ = suffix;
}

std::vector<InstCount> HistEnumTreeNode::GetPrefix() const {
  std::vector<InstCount> prefix;
  for (auto histNode = const_cast<HistEnumTreeNode *>(this);
       histNode != nullptr; histNode = histNode->GetParent()) {
    if (histNode->GetInstNum() != SCHD_STALL)
      prefix.push_back(histNode->GetInstNum());
  }
  std::reverse(prefix.begin(), prefix.end());
  return prefix;
}

CostHistEnumTreeNode::CostHistEnumTreeNode() {
  isLngthFsbl_ = true;
#ifdef IS_DEBUG
  costInfoSet_ = false;
#endif
}

CostHistEnumTreeNode::~CostHistEnumTreeNode() {}

void CostHistEnumTreeNode::Construct(EnumTreeNode *node, bool isTemp) {
#ifdef IS_DEBUG
  costInfoSet_ = false;
#endif
  HistEnumTreeNode::Construct(node, isTemp);
}

void CostHistEnumTreeNode::Init_() {
  HistEnumTreeNode::Init_();
#ifdef IS_DEBUG
  costInfoSet_ = false;
#endif
  cost_ = 0;
}

bool CostHistEnumTreeNode::DoesDominate(EnumTreeNode *node,
                                        Enumerator *enumrtr) {
  assert(isCnstrctd_);
  assert(enumrtr->IsCostEnum());

  InstCount shft = 0;

  // If the history node does not dominate the current node, we cannot
  // draw any conclusion and no pruning can be done.

  // (Chris): If scheduling for RP only, automatically assume all nodes are
  // feasible and just check for cost domination.
  if (!enumrtr->IsSchedForRPOnly()) {
    if (DoesDominate_(node, NULL, ETN_ACTIVE, enumrtr, shft) == false)
      return false;

    // if the history node dominates the current node, and there is
    // no feasible sched below the hist node, there cannot be a feasible
    // sched below the current node. So, prune the current node
    if (isLngthFsbl_ == false)
      return true;
  }

  // if the hist node dominates the current node, and the hist node
  // had at least one feasible sched below it, domination will be
  // determined by the cost domination condition
  return ChkCostDmntn_(node, enumrtr, shft);
}

bool CostHistEnumTreeNode::ChkCostDmntn_(EnumTreeNode *node,
                                         Enumerator *enumrtr,
                                         InstCount &maxShft) {
  return ChkCostDmntnForBBSpill_(node, enumrtr);
}

// For the SLIL cost function the improvement in cost when comparing the other
// prefix to the history prefix must be enough to improve upon the best cost
// found so far. The required improvement is:
// "History Total Cost" - "Best Total Cost"
static bool doesHistorySLILCostDominate(InstCount OtherPrefixCost,
                                        InstCount HistPrefixCost,
                                        InstCount HistTotalCost,
                                        LengthCostEnumerator *LCE) {
  auto RequiredImprovement = std::max(HistTotalCost - LCE->GetBestCost(), 0);
  auto ImprovementOnHistory = HistPrefixCost - OtherPrefixCost;
  return ImprovementOnHistory <= RequiredImprovement;
}

// For peak cost functions (PERP, PRP, Occupancy) the suffix cost does not
// depend on the prefix cost.
static bool doesHistoryPeakCostDominate(InstCount OtherPrefixCost,
                                        InstCount HistPrefixCost,
                                        InstCount HistTotalCost,
                                        LengthCostEnumerator *LCE) {
  // If we cannot improve the prefix, prune the candidate node. Likewise, if
  // the total cost is determined by the suffix schedule we cannot improve the
  // cost with a better prefix.
  if (OtherPrefixCost >= HistPrefixCost || HistTotalCost > HistPrefixCost)
    return true;

  // Prunes the candidate node if the improved prefix still has higher cost than
  // the best schedule found so far.
  return LCE->GetBestCost() <= OtherPrefixCost;
}

// Should we prune the other node based on RP cost.
bool CostHistEnumTreeNode::ChkCostDmntnForBBSpill_(EnumTreeNode *Node,
                                                   Enumerator *E) {
  if (time_ > Node->GetTime())
    return false;

  // If the other node's prefix cost is higher than or equal to the history
  // prefix cost the other node is pruned.
  assert(costInfoSet_);
  bool ShouldPrune;
  if (Node->GetCostLwrBound() >= partialCost_)
    ShouldPrune = true;
  else {
    ShouldPrune = false;
    LengthCostEnumerator *LCE = static_cast<LengthCostEnumerator *>(E);
    SPILL_COST_FUNCTION SpillCostFunc = LCE->GetSpillCostFunc();

    // We cannot prune based on prefix cost, but check for more aggressive
    // pruning conditions that are specific to the current cost function.
    if (SpillCostFunc == SCF_TARGET || SpillCostFunc == SCF_PRP ||
        SpillCostFunc == SCF_PERP)
      ShouldPrune = doesHistoryPeakCostDominate(Node->GetCostLwrBound(),
                                                partialCost_, totalCost_, LCE);

    else if (SpillCostFunc == SCF_SLIL)
      ShouldPrune = doesHistorySLILCostDominate(Node->GetCostLwrBound(),
                                                partialCost_, totalCost_, LCE);

    // If the cost function is peak plus avg, make sure that the fraction lost
    // by integer divsion does not lead to false domination.
    else if (SpillCostFunc == SCF_PEAK_PLUS_AVG && cost_ == Node->GetCost()) {
      InstCount instCnt = E->GetTotInstCnt();
      ShouldPrune =
          spillCostSum_ % instCnt >= Node->GetSpillCostSum() % instCnt;
    }
  }
  return ShouldPrune;
}

void CostHistEnumTreeNode::SetCostInfo(EnumTreeNode *node, bool, Enumerator *) {
  cost_ = node->GetCost();
  peakSpillCost_ = node->GetPeakSpillCost();
  spillCostSum_ = node->GetSpillCostSum();
  isLngthFsbl_ = node->IsLngthFsbl();

  // (Chris)
  partialCost_ = node->GetCostLwrBound();
  totalCost_ = node->GetTotalCost();
  totalCostIsActualCost_ = node->GetTotalCostIsActualCost();
  if (suffix_ == nullptr && node->GetSuffix().size() > 0)
    suffix_ =
        std::make_shared<std::vector<SchedInstruction *>>(node->GetSuffix());

#ifdef IS_DEBUG
  costInfoSet_ = true;
#endif

#if defined(IS_DEBUG_ARCHIVE)
  Logger::Info(
      "Setting cost info: cost=%d, peak=%d, partial=%d, total=%d, isreal=%d",
      cost_, peakSpillCost_, partialCost_, totalCost_,
      (totalCostIsActualCost_ ? 1 : 0));
#endif
}

InstCount HistEnumTreeNode::GetTime() { return time_; }

InstCount HistEnumTreeNode::GetInstNum() {
  return inst_ == NULL ? SCHD_STALL : inst_->GetNum();
}

bool HistEnumTreeNode::DoesMatch(EnumTreeNode *node, Enumerator *enumrtr) {
  BitVector *instsSchduld = enumrtr->bitVctr1_;
  BitVector *othrInstsSchduld = enumrtr->bitVctr2_;

  assert(instsSchduld != NULL && othrInstsSchduld != NULL);
  SetInstsSchduld_(instsSchduld);
  node->hstry_->SetInstsSchduld_(othrInstsSchduld);

  return *othrInstsSchduld == *instsSchduld;
}

bool HistEnumTreeNode::IsDominated(EnumTreeNode *node, Enumerator *enumrtr) {
  assert(node != NULL);
  InstCount shft = 0;
  return node->hstry_->DoesDominate_(NULL, this, ETN_HISTORY, enumrtr, shft);
}

HistEnumTreeNode *HistEnumTreeNode::GetParent() { return prevNode_; }

bool HistEnumTreeNode::IsPrdcsrViaStalls(HistEnumTreeNode *othrNode) {
  bool found = false;
  HistEnumTreeNode *node;
  assert(othrNode != this);

  for (node = this; node != NULL; node = node->GetParent()) {
    if (node->GetInstNum() != SCHD_STALL)
      break;
    if (node->GetParent() == othrNode) {
      found = true;
      break;
    }
  }

  return found;
}

void HistEnumTreeNode::ReplaceParent(HistEnumTreeNode *newParent) {
  assert(prevNode_ != NULL);
  assert(newParent->time_ <= prevNode_->time_);
  time_ = newParent->time_ + 1;
  prevNode_ = newParent;
}
