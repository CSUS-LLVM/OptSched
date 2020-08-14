/*******************************************************************************
Description:  Defines an enumerator class.
Author:       Ghassan Shobaki
Created:      Jun. 2002
Last Update:  Apr. 2020
*******************************************************************************/

#ifndef OPTSCHED_ENUM_ENUMERATOR_H
#define OPTSCHED_ENUM_ENUMERATOR_H

#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/gen_sched.h"
#include "opt-sched/Scheduler/mem_mngr.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/relaxed_sched.h"
#include <iostream>
#include <vector>

namespace llvm {
namespace opt_sched {

const int MAX_MEMBLOCK_SIZE = 10000;
const int TIMEOUT_TO_MEMBLOCK_RATIO = 10;

class SchedRegion;

// A pruning strategy.
struct Pruning {
  // Whether to apply relaxed pruning.
  bool rlxd;
  // Whether to apply node superiority.
  bool nodeSup;
  // Whether to apply history-based domination.
  bool histDom;
  // Whether to apply spill-cost pruning
  bool spillCost;
  // Whether to use suffix concatenation with history domination
  bool useSuffixConcatenation;
};

enum ENUMTREE_NODEMODE { ETN_PRELIM, ETN_ACTIVE, ETN_HISTORY };

struct TightndInst {
  SchedInstruction *inst;
  InstCount tightBound;
};

class Enumerator;
class HistEnumTreeNode;
class CostHistEnumTreeNode;

class EnumTreeNode {
private:
  friend class HistEnumTreeNode;
  friend class CostHistEnumTreeNode;

  class ExaminedInst {
  private:
    SchedInstruction *inst_;
    bool wasRlxInfsbl_; // was it infeasible because relaxed scheduling failed
    LinkedList<TightndInst> *tightndScsrs_;

  public:
    ExaminedInst(SchedInstruction *inst, bool wasRlxInfsbl,
                 LinkedList<SchedInstruction> *dirctTightndLst);
    ~ExaminedInst();

    inline SchedInstruction *GetInst();
    inline bool wasRlxInfsbl() { return wasRlxInfsbl_; }
    inline bool IsRsrcDmntd(SchedInstruction *cnddtInst);
  };

  // A pointer to the instruction whose scheduling has led from the previous
  // node to this node
  SchedInstruction *inst_;

  // Total number of branches at this node
  InstCount brnchCnt_;

  // The number of the current branch to explore next
  // All branches with smaller numbers have been explored already
  InstCount crntBrnchNum_;

  InstCount fsblBrnchCnt_;
  InstCount lngthFsblBrnchCnt_;

  // The current time or position (or step number) in the scheduling process
  // This is eqaul to the length of the path from the root node to this node
  InstCount time_;

  EnumTreeNode *prevNode_;

  Enumerator *enumrtr_;

  bool isEmpty_;

  bool isFsbl_;

  bool isLngthFsbl_;

  bool isLeaf_;

  ENUMTREE_NODEMODE mode_;

  // Array of instructions' forward lower bounds tightened up to this node
  InstCount *frwrdLwrBounds_;

  // Array hloding the number of issue slots available for each issue type
  // based on the target schedule length and the slots that have been taken
  InstCount avlblSlots_[MAX_ISSUTYPE_CNT];

  // Array hloding the number of issue slots available for each issue type
  // in the current machine cycle
  int16_t avlblSlotsInCrntCycle_[MAX_ISSUTYPE_CNT];

  bool crntCycleBlkd_;
  int realSlotNum_;

  // A list of "legal" instructions that have been examined at this node
  // along with a list of immediate successors that got tightened after
  // temporarily scheduling that instruction
  LinkedList<ExaminedInst> *exmndInsts_;

  InstCount legalInstCnt_;

  // A list of nodes that are dominated by this node
  HistEnumTreeNode *dmntdNode_;
  LinkedList<HistEnumTreeNode> *chldrn_;

  uint64_t num_;

  ReadyList *rdyLst_;

  HistEnumTreeNode *hstry_;

  bool isArchivd_;

  // The signature of the partial schedule up to this node
  InstSignature prtilSchedSig_;

  bool isCnstrctd_;
  bool isClean_;

  // Have we looked for an instruction in the ready list that uses a
  // register.
  // bool srchedInstWithUse_;
  // Did we find an instruction in the ready list that uses a register.
  bool foundInstWithUse_;

  InstCount cost_;
  InstCount costLwrBound_;
  InstCount peakSpillCost_;
  InstCount spillCostSum_;
  InstCount totalCost_ = -1;
  bool totalCostIsActualCost_ = false;
  ReserveSlot *rsrvSlots_;

  // (Chris)
  using SuffixType = std::vector<SchedInstruction *>;
  SuffixType suffix_;

  inline void CreateTmpHstry_();
  void FormPrtilSchedSig_();
  void Init_();
  inline bool IsNxtCycleNew_();

public:
  EnumTreeNode();
  ~EnumTreeNode();

  void Construct(EnumTreeNode *prevNode, SchedInstruction *inst,
                 Enumerator *enumrtr);
  void Clean();
  void Reset();

  void SetBranchCnt(InstCount rdyLstSize, bool isLeaf);

  // Notify this node that a new branch has been examined so that it advances
  // its branch pointer to the nex branch
  void NewBranchExmnd(SchedInstruction *inst, bool isLegal, bool isNodeDmntd,
                      bool isRlxInfsbl, bool isFsbl, DIRECTION dir,
                      bool isLngthFsbl);

  // Check the list of examined insts. to see if a superior inst. has been
  // examined already. If yes, the branch pointer will be advanced, since
  // it is assumed that the enumerator will skip this inst
  bool WasSprirNodeExmnd(SchedInstruction *cnddtInst);

  bool WasRsrcDmnntNodeExmnd(SchedInstruction *cnddtInst);

  inline void SetSlotAvlblty(InstCount avlblSlots[],
                             int16_t avlblSlotsInCrntCycle[]);

  inline InstCount GetBranchCnt(bool &isEmpty);
  inline InstCount GetBranchCnt();

  // Return a pointer to the array of lower bounds
  inline InstCount *GetLwrBounds(DIRECTION dir);

  inline void GetLwrBounds(DIRECTION dir, InstCount lwrBounds[]);
  inline void GetSlotAvlblty(InstCount avlblSlots[],
                             int16_t avlblSlotsInCrntCycle[]);
  inline InstCount GetCrntBranchNum();
  inline SchedInstruction *GetInst();
  inline InstCount GetInstNum();
  inline EnumTreeNode *GetParent();
  inline EnumTreeNode *GetGrandParent();
  void PrntPartialSched(std::ostream &out);
  inline bool IsRoot();

  inline uint64_t GetNum();
  inline void SetNum(uint64_t num);

  inline bool FoundInstWithUse();
  inline void SetFoundInstWithUse(bool foundInstWithUse);

  // Get the signature of the partial schedule up to this node
  inline InstSignature GetSig();

  // Get the time of this node in the schedule (total number of slots
  // scheduled) or, equivalently, the path from the root node to this node
  inline InstCount GetTime();

  bool DoesPartialSchedMatch(EnumTreeNode *othr);

  void SetLwrBounds(DIRECTION dir);
  void SetLwrBounds();

  void SetRsrvSlots(int16_t rsrvSlotCnt, ReserveSlot *rsrvSlots);

  // Add a node to the list of nodes dominated by this node
  inline void AddDmntdSubProb(HistEnumTreeNode *node);
  inline HistEnumTreeNode *GetDmntdSubProb();
  inline void AddChild(EnumTreeNode *node);

  inline void SetRdyLst(ReadyList *lst);

  inline ReadyList *GetRdyLst();

  inline ENUMTREE_NODEMODE GetMode();

  // Is the given branch dominated by a previously examined branch?
  inline bool IsBranchDominated(SchedInstruction *inst);

  inline InstCount GetLegalInstCnt();
  inline void LegalInstFound();
  inline InstCount GetChldrnCnt();

  inline void CreateHistory();
  inline void ReplaceHistory(HistEnumTreeNode *node);
  inline void SetHistory(HistEnumTreeNode *hstry);
  inline HistEnumTreeNode *GetHistory();

  inline bool IsArchived();
  void Archive();

  inline bool IsFeasible();
  inline bool IsLngthFsbl();
  inline void SetLngthFsblty(bool value);
  inline void SetFsblty(bool isFsbl);

  inline bool IsLeaf();
  inline void ChildInfsbl();
  inline void AddChild();

  inline void SetCost(InstCount cost);
  inline InstCount GetCost();

  inline void SetCostLwrBound(InstCount bound);
  inline InstCount GetCostLwrBound();

  inline void SetPeakSpillCost(InstCount cost);
  inline InstCount GetPeakSpillCost();

  inline void SetSpillCostSum(InstCount cost);
  inline InstCount GetSpillCostSum();

  bool ChkInstRdndncy(SchedInstruction *inst, int brnchNum);
  bool IsNxtSlotStall();

  bool GetCrntCycleBlkd() { return crntCycleBlkd_; }
  void SetCrntCycleBlkd(bool blkd) { crntCycleBlkd_ = blkd; }
  int GetRealSlotNum() { return realSlotNum_; }
  void SetRealSlotNum(int num) { realSlotNum_ = num; }

  inline InstCount GetTotalCost() const { return totalCost_; }
  inline void SetTotalCost(InstCount totalCost) { totalCost_ = totalCost; }

  inline InstCount GetTotalCostIsActualCost() const {
    return totalCostIsActualCost_;
  }
  inline void SetTotalCostIsActualCost(bool totalCostIsActualCost) {
    totalCostIsActualCost_ = totalCostIsActualCost;
  }

  inline const SuffixType &GetSuffix() const { return suffix_; }
  inline void SetSuffix(const SuffixType &suffix) { suffix_ = suffix; }
  inline void SetSuffix(SuffixType &&suffix) { suffix_ = std::move(suffix); }
};
/*****************************************************************************/

class EnumTreeNodeAlloc : public MemAlloc<EnumTreeNode> {
public:
  inline EnumTreeNodeAlloc(int maxSize);
  inline ~EnumTreeNodeAlloc();
  inline EnumTreeNode *Alloc(EnumTreeNode *prevNode, SchedInstruction *inst,
                             Enumerator *enumrtr);
  inline void Free(EnumTreeNode *node);
};
/*****************************************************************************/

class Enumerator : public ConstrainedScheduler {

protected:
  friend class EnumTreeNode;
  friend class HistEnumTreeNode;
  friend class CostHistEnumTreeNode;

  // TODO(max): Document.
  bool isCnstrctd_;

  Pruning prune_;
  bool enblStallEnum_;
  EnumTreeNode *rootNode_;
  EnumTreeNode *crntNode_;

  // The target length of which we are trying to find a feasible schedule
  InstCount trgtSchedLngth_;

  // A pointer to a relaxed scheduler
  RJ_RelaxedScheduler *rlxdSchdulr_;

  // Array holding the number of issue slots available for each issue type
  // based on the target schedule length and the slots that have been taken
  InstCount avlblSlots_[MAX_ISSUTYPE_CNT];

  // Array holding the number of insructions that have not been scheduled
  // for each issue type
  InstCount neededSlots_[MAX_ISSUTYPE_CNT];

  // To be used by the history node
  InstCount histNxtAvlblCycles_[MAX_ISSUTYPE_CNT];
  InstCount histInstsPerType_[MAX_ISSUTYPE_CNT];

  // Used for saving the original location when the scheduler advances
  // temporarily in order to probe the feasibility of a branch
  InstCount prevCycleNum_;
  InstCount prevSlotNum_;

  uint64_t maxNodeCnt_;
  uint64_t createdNodeCnt_;
  uint64_t exmndNodeCnt_;

  InstCount minUnschduldTplgclOrdr_;

  BinHashTable<HistEnumTreeNode> *exmndSubProbs_;

  // A list of insts whose lower bounds have been tightened to be used for
  // efficient untightening
  LinkedList<SchedInstruction> *tightndLst_;
  LinkedList<SchedInstruction> *bkwrdTightndLst_;
  LinkedList<SchedInstruction> *dirctTightndLst_;

  // A list of insts which got fixed in certain cycles to be used for
  // efficient unfixing
  LinkedList<SchedInstruction> *fxdLst_;

  InstCount fxdInstCnt_;

  // A structure for keeping track of any temporarily modified states so that
  // they can be restored
  typedef struct {
    bool instSchduld;
    bool issuSlotsProbed;
    bool lwrBoundsTightnd;
    bool instFxd;
    bool rlxSchduld;

  } State;
  State state_;

  // The number of prefixed instructions
  InstCount preFxdInstCnt_;
  // An array of prefixed instructions
  SchedInstruction **preFxdInsts_;

  int iterNum_;
  long backTrackCnt_;

  bool alctrsSetup_;
  MemAlloc<BinHashTblEntry<HistEnumTreeNode>> *hashTblEntryAlctr_;
  EnumTreeNodeAlloc *nodeAlctr_;

  InstCount *tmpLwrBounds_;

  int memAllocBlkSize_;

  HistEnumTreeNode *tmpHstryNode_;

  BitVector *bitVctr1_;
  BitVector *bitVctr2_;
  SchedInstruction **lastInsts_;
  SchedInstruction **othrLastInsts_;

  int fsblSchedCnt_;
  int imprvmntCnt_;
  InstCount prevTrgtLngth_;

  LISTSCHED_HEURISTIC enumHurstc_;

  bool isEarlySubProbDom_;

  // Should we ignore ilp and only schedule for register pressure.
  bool SchedForRPOnly_;

  // (Chris): Store the most recent matching hist node when checking for
  // history domination
  HistEnumTreeNode *mostRecentMatchingHistNode_ = nullptr;

  inline void ClearState_();
  inline bool IsStateClear_();

  // Can we find an instruction that uses a register in the ready list
  bool IsUseInRdyLst_();

  void StepFrwrd_(EnumTreeNode *&newNode);
  virtual bool BackTrack_();
  inline bool WasSolnFound_();

  void SetInstSigs_();
  bool InitPreFxdInsts_();

  // Try to find a new feasible branch from the current node.
  // If a feasible branch is found, a new node is created and
  // true is returned. Otherwise, false is returned
  bool FindNxtFsblBrnch_(EnumTreeNode *&newNode);
  inline bool ChkCrntNodeForFsblty_();

  void RestoreCrntState_(SchedInstruction *inst, EnumTreeNode *newNode);

  // Check if scheduling an instruction of a given type in the current
  // slot will break feasiblity from issue slot availbility point of view
  bool ProbeIssuSlotFsblty_(SchedInstruction *inst);

  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum);

  // Identify the current position in the schedule by linearizing the cycle
  // number and slot number into a single figure
  inline InstCount GetCrntTime_();

  inline InstCount GetCrntCycleNum_();

  inline int GetCrntSlotNum_();

  // Was a tree node (sub-problem) that dominates the candidate node examined?
  // The candidate node is the successor of the current node that is formed
  // by scheduling the given instruction next
  bool WasDmnntSubProbExmnd_(SchedInstruction *inst, EnumTreeNode *&newNode);

  bool TightnLwrBounds_(SchedInstruction *inst);
  void UnTightnLwrBounds_(SchedInstruction *newInst);
  void CmtLwrBoundTightnng_();

  bool FixInsts_(SchedInstruction *newInst);
  void UnFixInsts_(SchedInstruction *newInst);
  void CmtInstFxng_();

  void RestoreCrntLwrBounds_(SchedInstruction *unschduldInst);

  inline void CreateNewRdyLst_();
  bool RlxdSchdul_(EnumTreeNode *newNode);

  inline InstCount GetCycleNumFrmTime_(InstCount time);
  inline int GetSlotNumFrmTime_(InstCount time);

  virtual HistEnumTreeNode *AllocHistNode_(EnumTreeNode *node) = 0;
  virtual HistEnumTreeNode *AllocTempHistNode_(EnumTreeNode *node) = 0;
  virtual void FreeHistNode_(HistEnumTreeNode *histNode) = 0;

  inline EnumTreeNode *AllocNode_();
  inline void FreeNode_(EnumTreeNode *node);

  virtual void SetupAllocators_();
  virtual void FreeAllocators_();
  virtual void ResetAllocators_();

  void PrintLog_();

  FUNC_RESULT FindFeasibleSchedule_(InstSchedule *sched, InstCount trgtLngth,
                                    Milliseconds deadline);

  // Virtual Functions
  virtual bool WasObjctvMet_() = 0;

  // Check if branching from the current node by scheduling this instruction
  // in the current slot is feasible or not
  virtual bool ProbeBranch_(SchedInstruction *inst, EnumTreeNode *&newNode,
                            bool &isNodeDmntd, bool &isRlxInfsbl,
                            bool &isLngthFsbl);
  virtual bool Initialize_(InstSchedule *preSched, InstCount trgtLngth);
  virtual void CreateRootNode_();
  virtual bool EnumStall_();
  virtual void InitNewNode_(EnumTreeNode *newNode);

public:
  Enumerator(DataDepGraph *dataDepGraph, MachineModel *machMdl,
             InstCount schedUprBound, int16_t sigHashSize,
             SchedPriorities prirts, Pruning PruningStrategy,
             bool SchedForRPOnly, bool enblStallEnum, Milliseconds timeout,
             InstCount preFxdInstCnt = 0,
             SchedInstruction *preFxdInsts[] = NULL);
  virtual ~Enumerator();
  virtual void Reset();

  // Get the number of nodes that have been examined
  inline uint64_t GetNodeCnt();

  inline int GetSearchCnt();

  inline bool IsHistDom();
  inline bool IsRlxdPrnng();
  virtual bool IsCostEnum() = 0;

  // (Chris)
  inline bool IsSchedForRPOnly() const { return SchedForRPOnly_; }

  // Calculates the schedule and returns it in the passed argument.
  FUNC_RESULT FindSchedule(InstSchedule *sched, SchedRegion *rgn) {
    return RES_ERROR;
  }
};
/*****************************************************************************/

class LengthEnumerator : public Enumerator {

private:
  MemAlloc<HistEnumTreeNode> *histNodeAlctr_;

  // Virtual Functions
  virtual bool WasObjctvMet_();

  void SetupAllocators_();
  void FreeAllocators_();
  void ResetAllocators_();

  HistEnumTreeNode *AllocHistNode_(EnumTreeNode *node);
  HistEnumTreeNode *AllocTempHistNode_(EnumTreeNode *node);
  void FreeHistNode_(HistEnumTreeNode *histNode);

public:
  LengthEnumerator(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                   InstCount schedUprBound, int16_t sigHashSize,
                   SchedPriorities prirts, Pruning PruningStrategy,
                   bool SchedForRPOnly, bool enblStallEnum,
                   Milliseconds timeout, InstCount preFxdInstCnt = 0,
                   SchedInstruction *preFxdInsts[] = NULL);
  virtual ~LengthEnumerator();
  void Reset();

  // Given a schedule with some instructions possibly fixed, find a
  // feasible schedule of the given target length if possible
  FUNC_RESULT FindFeasibleSchedule(InstSchedule *sched, InstCount trgtLngth,
                                   Milliseconds deadline);
  bool IsCostEnum();
};
/*****************************************************************************/

class LengthCostEnumerator : public Enumerator {
private:
  int costChkCnt_;
  int costPruneCnt_;
  int costLwrBound_;
  MemAlloc<CostHistEnumTreeNode> *histNodeAlctr_;
  SPILL_COST_FUNCTION spillCostFunc_;

  // Virtual Functions
  void SetupAllocators_();
  void FreeAllocators_();
  void ResetAllocators_();

  HistEnumTreeNode *AllocHistNode_(EnumTreeNode *node);
  HistEnumTreeNode *AllocTempHistNode_(EnumTreeNode *node);
  void FreeHistNode_(HistEnumTreeNode *histNode);

  bool WasObjctvMet_();
  bool BackTrack_();
  InstCount GetBestCost_();
  void CreateRootNode_();

  // Check if branching from the current node by scheduling this instruction
  // in the current slot is feasible or not
  bool ProbeBranch_(SchedInstruction *inst, EnumTreeNode *&newNode,
                    bool &isNodeDmntd, bool &isRlxInfsbl, bool &isLngthFsbl);
  bool Initialize_(InstSchedule *preSched, InstCount trgtLngth);
  bool ChkCostFsblty_(SchedInstruction *inst, EnumTreeNode *&newNode);
  bool EnumStall_();
  void InitNewNode_(EnumTreeNode *newNode);

public:
  LengthCostEnumerator(DataDepGraph *dataDepGraph, MachineModel *machMdl,
                       InstCount schedUprBound, int16_t sigHashSize,
                       SchedPriorities prirts, Pruning PruningStrategy,
                       bool SchedForRPOnly, bool enblStallEnum,
                       Milliseconds timeout, SPILL_COST_FUNCTION spillCostFunc,
                       InstCount preFxdInstCnt = 0,
                       SchedInstruction *preFxdInsts[] = NULL);
  virtual ~LengthCostEnumerator();
  void Reset();

  // Given a schedule with some instructions possibly fixed, find a
  // feasible schedule of the given target length if possible
  FUNC_RESULT FindFeasibleSchedule(InstSchedule *sched, InstCount trgtLngth,
                                   SchedRegion *rgn, int costLwrBound,
                                   Milliseconds deadline);
  bool IsCostEnum();
  SPILL_COST_FUNCTION GetSpillCostFunc() { return spillCostFunc_; }
  inline InstCount GetBestCost() { return GetBestCost_(); }
};
/*****************************************************************************/

/******************************************************************************
Inline Functions
******************************************************************************/

void EnumTreeNode::ChildInfsbl() {
  assert(fsblBrnchCnt_ >= 1);
  fsblBrnchCnt_--;

  if (fsblBrnchCnt_ == 0) {
    isFsbl_ = false;
  }
}
/*****************************************************************************/

void EnumTreeNode::AddChild() {
  assert(fsblBrnchCnt_ == 0 && isFsbl_ == false);
  fsblBrnchCnt_++;
  isFsbl_ = true;
}
/*****************************************************************************/

void EnumTreeNode::LegalInstFound() {
  legalInstCnt_++;
  assert(legalInstCnt_ > 0);
}
/*****************************************************************************/

inline void EnumTreeNode::SetSlotAvlblty(InstCount avlblSlots[],
                                         int16_t avlblSlotsInCrntCycle[]) {
  int16_t issuTypeCnt = enumrtr_->issuTypeCnt_;

  for (int16_t i = 0; i < issuTypeCnt; i++) {
    avlblSlots_[i] = avlblSlots[i];
    avlblSlotsInCrntCycle_[i] = avlblSlotsInCrntCycle[i];
  }
}
/*****************************************************************************/

inline void EnumTreeNode::GetSlotAvlblty(InstCount avlblSlots[],
                                         int16_t avlblSlotsInCrntCycle[]) {
  int16_t issuTypeCnt = enumrtr_->issuTypeCnt_;

  for (int16_t i = 0; i < issuTypeCnt; i++) {
    avlblSlots[i] = avlblSlots_[i];
    avlblSlotsInCrntCycle[i] = avlblSlotsInCrntCycle_[i];
  }
}
/*****************************************************************************/

inline InstCount EnumTreeNode::GetBranchCnt(bool &isEmpty) {
  isEmpty = isEmpty_;
  return brnchCnt_;
}
/*****************************************************************************/

InstCount EnumTreeNode::GetBranchCnt() { return brnchCnt_; }
/**************************************************************************/

void EnumTreeNode::GetLwrBounds(DIRECTION dir, InstCount lwrBounds[]) {
  InstCount i;
  InstCount *nodeLwrBounds;
  InstCount instCnt = enumrtr_->totInstCnt_;

  assert(dir == DIR_FRWRD);
  nodeLwrBounds = frwrdLwrBounds_;

  for (i = 0; i < instCnt; i++) {
    lwrBounds[i] = nodeLwrBounds[i];
  }
}
/**************************************************************************/

InstCount *EnumTreeNode::GetLwrBounds(DIRECTION dir) {
  InstCount *nodeLwrBounds;
  assert(dir == DIR_FRWRD);
  nodeLwrBounds = frwrdLwrBounds_;
  return nodeLwrBounds;
}
/**************************************************************************/

InstCount EnumTreeNode::GetCrntBranchNum() { return crntBrnchNum_; }
/**************************************************************************/

SchedInstruction *EnumTreeNode::GetInst() { return inst_; }
/**************************************************************************/

inline InstCount EnumTreeNode::GetInstNum() {
  return inst_ == NULL ? SCHD_STALL : inst_->GetNum();
}
/**************************************************************************/

inline EnumTreeNode *EnumTreeNode::GetParent() { return prevNode_; }
/**************************************************************************/

inline EnumTreeNode *EnumTreeNode::GetGrandParent() {
  return prevNode_ == NULL ? NULL : prevNode_->GetParent();
}
/**************************************************************************/

inline bool EnumTreeNode::IsRoot() { return prevNode_ == NULL; }
/**************************************************************************/

inline uint64_t EnumTreeNode::GetNum() { return num_; }
/**************************************************************************/

inline void EnumTreeNode::SetNum(uint64_t num) { num_ = num; }
/**************************************************************************/

inline InstSignature EnumTreeNode::GetSig() { return prtilSchedSig_; }
/**************************************************************************/

inline InstCount EnumTreeNode::GetTime() { return time_; }
/**************************************************************************/

inline void EnumTreeNode::AddDmntdSubProb(HistEnumTreeNode *node) {
  dmntdNode_ = node;
}
/**************************************************************************/

inline HistEnumTreeNode *EnumTreeNode::GetDmntdSubProb() { return dmntdNode_; }
/**************************************************************************/

inline void EnumTreeNode::AddChild(EnumTreeNode *node) {
  assert(node->GetHistory() != NULL);
  chldrn_->InsrtElmnt(node->GetHistory());
}
/**************************************************************************/

inline void EnumTreeNode::SetRdyLst(ReadyList *lst) {
  rdyLst_ = lst;
  mode_ = ETN_ACTIVE;
}
/**************************************************************************/

inline bool EnumTreeNode::FoundInstWithUse() { return foundInstWithUse_; }
/**************************************************************************/

inline void EnumTreeNode::SetFoundInstWithUse(bool foundInstWithUse) {
  foundInstWithUse_ = foundInstWithUse;
}
/**************************************************************************/

inline ReadyList *EnumTreeNode::GetRdyLst() { return rdyLst_; }
/**************************************************************************/

inline ENUMTREE_NODEMODE EnumTreeNode::GetMode() { return mode_; }
/**************************************************************************/

inline InstCount EnumTreeNode::GetLegalInstCnt() { return legalInstCnt_; }
/**************************************************************************/

inline SchedInstruction *EnumTreeNode::ExaminedInst::GetInst() { return inst_; }
/**************************************************************************/

inline void EnumTreeNode::CreateHistory() {
  hstry_ = enumrtr_->AllocHistNode_(this);
}
/**************************************************************************/

inline void EnumTreeNode::CreateTmpHstry_() {
  hstry_ = enumrtr_->AllocTempHistNode_(this);
}
/**************************************************************************/

inline void EnumTreeNode::SetHistory(HistEnumTreeNode *hstry) {
  hstry_ = hstry;
}
/**************************************************************************/

inline HistEnumTreeNode *EnumTreeNode::GetHistory() { return hstry_; }
/**************************************************************************/

inline void EnumTreeNode::ReplaceHistory(HistEnumTreeNode *newHstry) {
  assert(hstry_ != NULL);

  hstry_ = newHstry;
}
/**************************************************************************/

inline bool EnumTreeNode::IsArchived() { return isArchivd_; }
/**************************************************************************/

inline bool EnumTreeNode::IsFeasible() {
  assert(isLeaf_ == false || isFsbl_);
  return isFsbl_;
}
/**************************************************************************/

inline void EnumTreeNode::SetFsblty(bool isFsbl) { isFsbl_ = isFsbl; }
/**************************************************************************/

inline void EnumTreeNode::SetLngthFsblty(bool value) { isLngthFsbl_ = value; }
/**************************************************************************/

inline bool EnumTreeNode::IsLeaf() { return isLeaf_; }

void EnumTreeNode::SetCost(InstCount cost) {
  assert(cost >= 0);
  cost_ = cost;
}
/*****************************************************************************/

InstCount EnumTreeNode::GetCost() { return cost_; }
/*****************************************************************************/

void EnumTreeNode::SetCostLwrBound(InstCount bound) {
  assert(bound >= 0);
  costLwrBound_ = bound;
}
/*****************************************************************************/

InstCount EnumTreeNode::GetCostLwrBound() { return costLwrBound_; }
/*****************************************************************************/

void EnumTreeNode::SetPeakSpillCost(InstCount cost) {
  assert(cost >= 0);
  peakSpillCost_ = cost;
}
/*****************************************************************************/

InstCount EnumTreeNode::GetPeakSpillCost() { return peakSpillCost_; }
/*****************************************************************************/

void EnumTreeNode::SetSpillCostSum(InstCount cost) {
  assert(cost >= 0);
  spillCostSum_ = cost;
}
/*****************************************************************************/

InstCount EnumTreeNode::GetSpillCostSum() { return spillCostSum_; }
/*****************************************************************************/

bool EnumTreeNode::IsNxtCycleNew_() {
  if (enumrtr_->issuRate_ == 1) {
    return true;
  }

  InstCount crntCycle = enumrtr_->GetCycleNumFrmTime_(time_);
  InstCount nxtCycle = enumrtr_->GetCycleNumFrmTime_(time_ + 1);
  assert(nxtCycle >= crntCycle);
  return (nxtCycle > crntCycle);
}
/*****************************************************************************/

bool EnumTreeNode::ExaminedInst::IsRsrcDmntd(SchedInstruction *) {
  if (!wasRlxInfsbl_)
    return false;

  for (TightndInst *tightndScsr = tightndScsrs_->GetFrstElmnt();
       tightndScsr != NULL; tightndScsr = tightndScsrs_->GetNxtElmnt()) {
    SchedInstruction *scsr = tightndScsr->inst;

    // If the release time of one successor of the examined instruction
    // was tighter than the current release time after the temp. scheduling
    // of the candidate instruction, then we cannot make a judgement whether
    // the examined dominates the candidate from rsource point of view
    if (tightndScsr->tightBound > scsr->GetCrntLwrBound(DIR_FRWRD)) {
      return false;
    }
  }

  return true;
}
/*****************************************************************************/

bool EnumTreeNode::IsLngthFsbl() { return isLngthFsbl_; }
/*****************************************************************************/

inline bool Enumerator::WasSolnFound_() {

  bool isCmplt = IsSchedComplete_();
  assert(crntSched_->GetCrntLngth() <= trgtSchedLngth_);
  bool isTrgt = crntSched_->GetCrntLngth() == trgtSchedLngth_;

  if (isCmplt && isTrgt) {
    fsblSchedCnt_++;
  }

  return isCmplt && isTrgt;
}
/*****************************************************************************/

inline InstCount Enumerator::GetCrntTime_() {
  return (crntCycleNum_ * issuRate_ + crntSlotNum_);
}
/*****************************************************************************/

inline InstCount Enumerator::GetCrntCycleNum_() { return crntCycleNum_; }
/*****************************************************************************/

inline int Enumerator::GetCrntSlotNum_() { return crntSlotNum_; }
/*****************************************************************************/

inline void Enumerator::UpdtRdyLst_(InstCount cycleNum, int slotNum) {
  InstCount prevCycleNum = cycleNum - 1;
  LinkedList<SchedInstruction> *lst1 = NULL;
  LinkedList<SchedInstruction> *lst2 = frstRdyLstPerCycle_[cycleNum];

  if (prirts_.isDynmc)
    rdyLst_->UpdatePriorities();

  if (slotNum == 0 && prevCycleNum >= 0) {
    // If at the begining of a new cycle other than the very first cycle, then
    // we also have to include the instructions that might have become ready in
    // the previous cycle due to a zero latency of the instruction scheduled in
    // the very last slot of that cycle [GOS 9.8.02].
    lst1 = frstRdyLstPerCycle_[prevCycleNum];
  }

  rdyLst_->AddLatestSubLists(lst1, lst2);
}
/*****************************************************************************/

inline void Enumerator::ClearState_() {
  state_.instSchduld = false;
  state_.issuSlotsProbed = false;
  state_.lwrBoundsTightnd = false;
  state_.instFxd = false;
  state_.rlxSchduld = false;
}
/****************************************************************************/

inline bool Enumerator::IsStateClear_() {
  if (state_.instSchduld) {
    return false;
  }

  if (state_.issuSlotsProbed) {
    return false;
  }

  if (state_.lwrBoundsTightnd) {
    return false;
  }

  return true;
}
/****************************************************************************/

inline uint64_t Enumerator::GetNodeCnt() { return exmndNodeCnt_; }
/****************************************************************************/

inline int Enumerator::GetSearchCnt() { return iterNum_; }
/****************************************************************************/

inline void Enumerator::CreateNewRdyLst_() {
  ReadyList *oldLst = rdyLst_;

  rdyLst_ = new ReadyList(dataDepGraph_, prirts_);

  if (oldLst != NULL) {
    rdyLst_->CopyList(oldLst);
  }
}
/****************************************************************************/

inline bool Enumerator::ChkCrntNodeForFsblty_() {
  SchedInstruction *inst;

  for (inst = bkwrdTightndLst_->GetFrstElmnt(); inst != NULL;
       inst = bkwrdTightndLst_->GetNxtElmnt()) {
    assert(inst->IsSchduld() == false);

    if (inst->GetCrntDeadline() < crntCycleNum_) {
      return false;
    }
  }

  return true;
}
/****************************************************************************/

inline InstCount Enumerator::GetCycleNumFrmTime_(InstCount time) {
  return (time - 1) / issuRate_;
}
/****************************************************************************/

inline int Enumerator::GetSlotNumFrmTime_(InstCount time) {
  assert(time >= 1);
  return (time - 1) % issuRate_;
}
/****************************************************************************/

bool Enumerator::IsHistDom() { return prune_.histDom; }
/******************************************************************************/

bool Enumerator::IsRlxdPrnng() { return prune_.rlxd; }
/******************************************************************************/

inline EnumTreeNodeAlloc::EnumTreeNodeAlloc(int maxSize)
    : MemAlloc<EnumTreeNode>(maxSize, maxSize) {}
/****************************************************************************/

inline EnumTreeNodeAlloc::~EnumTreeNodeAlloc() {}
/****************************************************************************/

inline EnumTreeNode *EnumTreeNodeAlloc::Alloc(EnumTreeNode *prevNode,
                                              SchedInstruction *inst,
                                              Enumerator *enumrtr) {
  EnumTreeNode *node = GetObject();
  node->Construct(prevNode, inst, enumrtr);
  return node;
}
/****************************************************************************/

inline void EnumTreeNodeAlloc::Free(EnumTreeNode *node) {
  node->Clean();
  FreeObject(node);
}
/****************************************************************************/

} // namespace opt_sched
} // namespace llvm

#endif
