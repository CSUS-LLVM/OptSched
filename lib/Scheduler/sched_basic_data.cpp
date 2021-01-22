#include "opt-sched/Scheduler/sched_basic_data.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/stats.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace llvm::opt_sched;

SPILL_COST_FUNCTION llvm::opt_sched::ParseSCFName(const std::string &name) {
  // PERP used to be called PEAK.
  if (name == "PERP" || name == "PEAK") {
    return SCF_PERP;
  } else if (name == "PRP") {
    return SCF_PRP;
  } else if (name == "PEAK_PER_TYPE") {
    return SCF_PEAK_PER_TYPE;
  } else if (name == "SUM") {
    return SCF_SUM;
  } else if (name == "PEAK_PLUS_AVG") {
    return SCF_PEAK_PLUS_AVG;
  } else if (name == "SLIL") {
    return SCF_SLIL;
  } else if (name == "OCC" || name == "TARGET") {
    return SCF_TARGET;
  }

  llvm::report_fatal_error(
      "Unrecognized option for SPILL_COST_FUNCTION setting: " + name, false);
}

SchedInstruction::SchedInstruction(InstCount num, const string &name,
                                   InstType instType, const string &opCode,
                                   InstCount maxInstCnt, int nodeID,
                                   InstCount fileSchedOrder,
                                   InstCount fileSchedCycle, InstCount fileLB,
                                   InstCount fileUB, MachineModel *model)
    : GraphNode(num, maxInstCnt) {
  // Static data that is computed only once.
  name_ = name;
  opCode_ = opCode;
  instType_ = instType;

  frwrdLwrBound_ = INVALID_VALUE;
  bkwrdLwrBound_ = INVALID_VALUE;
  abslutFrwrdLwrBound_ = INVALID_VALUE;
  abslutBkwrdLwrBound_ = INVALID_VALUE;
  crtclPathFrmRoot_ = INVALID_VALUE;
  crtclPathFrmLeaf_ = INVALID_VALUE;

  ltncyPerPrdcsr_ = NULL;
  memAllocd_ = false;
  sortedPrdcsrLst_ = NULL;
  sortedScsrLst_ = NULL;

  crtclPathFrmRcrsvScsr_ = NULL;
  crtclPathFrmRcrsvPrdcsr_ = NULL;

  // Dynamic data that changes during scheduling.
  ready_ = false;
  rdyCyclePerPrdcsr_ = NULL;
  minRdyCycle_ = INVALID_VALUE;
  prevMinRdyCyclePerPrdcsr_ = NULL;
  unschduldPrdcsrCnt_ = 0;
  unschduldScsrCnt_ = 0;

  crntRange_ = new SchedRange(this);

  crntSchedCycle_ = SCHD_UNSCHDULD;
  crntRlxdCycle_ = SCHD_UNSCHDULD;
  sig_ = 0;
  preFxdCycle_ = INVALID_VALUE;

  blksCycle_ = model->BlocksCycle(instType);
  pipelined_ = model->IsPipelined(instType);

  defCnt_ = 0;
  useCnt_ = 0;

  nodeID_ = nodeID;
  fileSchedOrder_ = fileSchedOrder;
  fileSchedCycle_ = fileSchedCycle;
  fileLwrBound_ = fileLB;
  fileUprBound_ = fileUB;

  mustBeInBBEntry_ = false;
  mustBeInBBExit_ = false;
}

SchedInstruction::~SchedInstruction() {
  if (memAllocd_)
    DeAllocMem_();
  delete crntRange_;
}

void SchedInstruction::SetupForSchdulng(InstCount instCnt, bool isCP_FromScsr,
                                        bool isCP_FromPrdcsr) {
  if (memAllocd_)
    DeAllocMem_();
  AllocMem_(instCnt, isCP_FromScsr, isCP_FromPrdcsr);

  SetPrdcsrNums_();
  SetScsrNums_();
  ComputeAdjustedUseCnt_();
}

bool SchedInstruction::UseFileBounds() {
  bool match = true;
#ifdef IS_DEBUG_BOUNDS
  stats::totalInstructions++;

  if (frwrdLwrBound_ == fileLwrBound_) {
    stats::instructionsWithEqualLB++;
  }

  if (fileLwrBound_ > frwrdLwrBound_) {
    stats::instructionsWithTighterFileLB++;
    stats::cyclesTightenedForTighterFileLB += fileLwrBound_ - frwrdLwrBound_;
  }

  if (frwrdLwrBound_ > fileLwrBound_) {
    stats::instructionsWithTighterRelaxedLB++;
    stats::cyclesTightenedForTighterRelaxedLB += frwrdLwrBound_ - fileLwrBound_;
  }

  if (frwrdLwrBound_ != fileLwrBound_) {
    match = false;
    Logger::Info("File LB =%d, Rec LB=%d, instNum=%d, pred Cnt=%d",
                 fileLwrBound_, frwrdLwrBound_, num_, prdcsrCnt_);
  }

  if (bkwrdLwrBound_ == fileUprBound_) {
    stats::instructionsWithEqualUB++;
  }

  if (fileUprBound_ > bkwrdLwrBound_) {
    stats::instructionsWithTighterFileUB++;
    stats::cyclesTightenedForTighterFileUB += fileUprBound_ - bkwrdLwrBound_;
  }

  if (bkwrdLwrBound_ > fileUprBound_) {
    stats::instructionsWithTighterRelaxedUB++;
    stats::cyclesTightenedForTighterRelaxedUB += bkwrdLwrBound_ - fileUprBound_;
  }

  if (bkwrdLwrBound_ != fileUprBound_) {
    match = false;
    Logger::Info("File UB =%d, Rec UB=%d, instNum=%d, pred Cnt=%d",
                 fileUprBound_, bkwrdLwrBound_, num_, prdcsrCnt_);
  }
#endif
  SetBounds(fileLwrBound_, fileUprBound_);
  return match;
}

bool SchedInstruction::InitForSchdulng(InstCount schedLngth,
                                       LinkedList<SchedInstruction> *fxdLst) {
  crntSchedCycle_ = SCHD_UNSCHDULD;
  crntRlxdCycle_ = SCHD_UNSCHDULD;

  for (InstCount i = 0; i < prdcsrCnt_; i++) {
    rdyCyclePerPrdcsr_[i] = INVALID_VALUE;
    prevMinRdyCyclePerPrdcsr_[i] = INVALID_VALUE;
  }

  ready_ = false;
  minRdyCycle_ = INVALID_VALUE;
  unschduldPrdcsrCnt_ = prdcsrCnt_;
  unschduldScsrCnt_ = scsrCnt_;
  lastUseCnt_ = 0;

  if (schedLngth != INVALID_VALUE) {
    bool fsbl = crntRange_->SetBounds(frwrdLwrBound_, bkwrdLwrBound_,
                                      schedLngth, fxdLst);
    if (!fsbl)
      return false;
  }

  return true;
}

void SchedInstruction::AllocMem_(InstCount instCnt, bool isCP_FromScsr,
                                 bool isCP_FromPrdcsr) {
  scsrCnt_ = GetScsrCnt();
  prdcsrCnt_ = GetPrdcsrCnt();
  rdyCyclePerPrdcsr_ = new InstCount[prdcsrCnt_];
  ltncyPerPrdcsr_ = new InstCount[prdcsrCnt_];
  prevMinRdyCyclePerPrdcsr_ = new InstCount[prdcsrCnt_];
  sortedPrdcsrLst_ = new PriorityList<SchedInstruction>;

  InstCount predecessorIndex = 0;
  for (GraphEdge *edge = GetFrstPrdcsrEdge(); edge != NULL;
       edge = GetNxtPrdcsrEdge()) {
    ltncyPerPrdcsr_[predecessorIndex++] = edge->label;
    sortedPrdcsrLst_->InsrtElmnt((SchedInstruction *)edge->GetOtherNode(this),
                                 edge->label, true);
  }

  if (isCP_FromScsr) {
    crtclPathFrmRcrsvScsr_ = new InstCount[instCnt];

    for (InstCount i = 0; i < instCnt; i++) {
      crtclPathFrmRcrsvScsr_[i] = INVALID_VALUE;
    }

    crtclPathFrmRcrsvScsr_[GetNum()] = 0;
  }

  if (isCP_FromPrdcsr) {
    crtclPathFrmRcrsvPrdcsr_ = new InstCount[instCnt];

    for (InstCount i = 0; i < instCnt; i++) {
      crtclPathFrmRcrsvPrdcsr_[i] = INVALID_VALUE;
    }

    crtclPathFrmRcrsvPrdcsr_[GetNum()] = 0;
  }

  memAllocd_ = true;
}

void SchedInstruction::DeAllocMem_() {
  assert(memAllocd_);

  if (rdyCyclePerPrdcsr_ != NULL)
    delete[] rdyCyclePerPrdcsr_;
  if (prevMinRdyCyclePerPrdcsr_ != NULL)
    delete[] prevMinRdyCyclePerPrdcsr_;
  if (ltncyPerPrdcsr_ != NULL)
    delete[] ltncyPerPrdcsr_;
  if (sortedPrdcsrLst_ != NULL)
    delete sortedPrdcsrLst_;
  if (sortedScsrLst_ != NULL)
    delete sortedScsrLst_;
  if (crtclPathFrmRcrsvScsr_ != NULL)
    delete[] crtclPathFrmRcrsvScsr_;
  if (crtclPathFrmRcrsvPrdcsr_ != NULL)
    delete[] crtclPathFrmRcrsvPrdcsr_;

  memAllocd_ = false;
}

InstCount SchedInstruction::CmputCrtclPath_(DIRECTION dir,
                                            SchedInstruction *ref) {
  // The idea of this function is considering each predecessor (successor) and
  // calculating the length of the path from the root (leaf) through that
  // predecessor (successor) and then taking the maximum value among all these
  // paths.
  InstCount crtclPath = 0;
  LinkedList<GraphEdge> *nghbrLst = GetNghbrLst(dir);

  for (GraphEdge *edg = nghbrLst->GetFrstElmnt(); edg != NULL;
       edg = nghbrLst->GetNxtElmnt()) {
    UDT_GLABEL edgLbl = edg->label;
    SchedInstruction *nghbr = (SchedInstruction *)(edg->GetOtherNode(this));

    InstCount nghbrCrtclPath;
    if (ref == NULL) {
      nghbrCrtclPath = nghbr->GetCrtclPath(dir);
    } else {
      // When computing relative critical paths, we only need to consider
      // neighbors that belong to the sub-tree rooted at the reference.
      if (!ref->IsRcrsvNghbr(dir, nghbr))
        continue;
      nghbrCrtclPath = nghbr->GetRltvCrtclPath(dir, ref);
    }
    assert(nghbrCrtclPath != INVALID_VALUE);

    if ((nghbrCrtclPath + edgLbl) > crtclPath) {
      crtclPath = nghbrCrtclPath + edgLbl;
    }
  }

  return crtclPath;
}

bool SchedInstruction::ApplyPreFxng(LinkedList<SchedInstruction> *tightndLst,
                                    LinkedList<SchedInstruction> *fxdLst) {
  return crntRange_->Fix(preFxdCycle_, tightndLst, fxdLst);
}

void SchedInstruction::AddDef(Register *reg) {
  if (defCnt_ >= MAX_DEFS_PER_INSTR) {
    llvm::report_fatal_error("An instruction can't have more than " +
                                 std::to_string(MAX_DEFS_PER_INSTR) + " defs",
                             false);
  }
  // Logger::Info("Inst %d defines reg %d of type %d and physNum %d and useCnt
  // %d",
  // num_, reg->GetNum(), reg->GetType(), reg->GetPhysicalNumber(),
  // reg->GetUseCnt());
  assert(reg != NULL);
  defs_[defCnt_++] = reg;
}

void SchedInstruction::AddUse(Register *reg) {
  if (useCnt_ >= MAX_USES_PER_INSTR) {
    llvm::report_fatal_error("An instruction can't have more than " +
                                 std::to_string(MAX_USES_PER_INSTR) + " uses",
                             false);
  }
  // Logger::Info("Inst %d uses reg %d of type %d and physNum %d and useCnt %d",
  // num_, reg->GetNum(), reg->GetType(), reg->GetPhysicalNumber(),
  // reg->GetUseCnt());
  assert(reg != NULL);
  uses_[useCnt_++] = reg;
}

bool SchedInstruction::FindDef(Register *reg) const {
  return llvm::any_of(GetDefs(), [reg](const Register *r) { return reg == r; });
}

bool SchedInstruction::FindUse(Register *reg) const {
  return llvm::any_of(GetUses(), [reg](const Register *r) { return reg == r; });
}

bool SchedInstruction::BlocksCycle() const { return blksCycle_; }

bool SchedInstruction::IsPipelined() const { return pipelined_; }

bool SchedInstruction::MustBeInBBEntry() const {
  return mustBeInBBEntry_;
  //  return opCode_=="CopyFromReg" || opCode_=="ADJCALLSTACKDOWN32";
}

bool SchedInstruction::MustBeInBBExit() const {
  return mustBeInBBExit_;
  //  return opCode_=="CopyToReg";
}

void SchedInstruction::SetMustBeInBBEntry(bool val) { mustBeInBBEntry_ = val; }

void SchedInstruction::SetMustBeInBBExit(bool val) { mustBeInBBExit_ = val; }

const char *SchedInstruction::GetName() const { return name_.c_str(); }

const char *SchedInstruction::GetOpCode() const { return opCode_.c_str(); }

int SchedInstruction::GetNodeID() const { return nodeID_; }

void SchedInstruction::SetNodeID(int nodeID) { nodeID_ = nodeID; }

int SchedInstruction::GetLtncySum() const { return GetScsrLblSum(); }

int SchedInstruction::GetMaxLtncy() const { return GetMaxEdgeLabel(); }

SchedInstruction *SchedInstruction::GetFrstPrdcsr(InstCount *scsrNum,
                                                  UDT_GLABEL *ltncy,
                                                  DependenceType *depType) {
  GraphEdge *edge = GetFrstPrdcsrEdge();
  if (!edge)
    return NULL;
  if (scsrNum)
    *scsrNum = edge->succOrder;
  if (ltncy)
    *ltncy = edge->label;
  if (depType)
    *depType = (DependenceType)edge->label2;
  return (SchedInstruction *)(edge->from);
}

SchedInstruction *SchedInstruction::GetNxtPrdcsr(InstCount *scsrNum,
                                                 UDT_GLABEL *ltncy,
                                                 DependenceType *depType) {
  GraphEdge *edge = GetNxtPrdcsrEdge();
  if (!edge)
    return NULL;
  if (scsrNum)
    *scsrNum = edge->succOrder;
  if (ltncy)
    *ltncy = edge->label;
  if (depType)
    *depType = (DependenceType)edge->label2;
  return (SchedInstruction *)(edge->from);
}

SchedInstruction *SchedInstruction::GetFrstScsr(InstCount *prdcsrNum,
                                                UDT_GLABEL *ltncy,
                                                DependenceType *depType,
                                                bool *IsArtificial) {
  GraphEdge *edge = GetFrstScsrEdge();
  if (!edge)
    return NULL;
  if (prdcsrNum)
    *prdcsrNum = edge->predOrder;
  if (ltncy)
    *ltncy = edge->label;
  if (depType)
    *depType = (DependenceType)edge->label2;
  if (IsArtificial)
    *IsArtificial = edge->IsArtificial;
  return (SchedInstruction *)(edge->to);
}

SchedInstruction *SchedInstruction::GetNxtScsr(InstCount *prdcsrNum,
                                               UDT_GLABEL *ltncy,
                                               DependenceType *depType,
                                               bool *IsArtificial) {
  GraphEdge *edge = GetNxtScsrEdge();
  if (!edge)
    return NULL;
  if (prdcsrNum)
    *prdcsrNum = edge->predOrder;
  if (ltncy)
    *ltncy = edge->label;
  if (depType)
    *depType = (DependenceType)edge->label2;
  if (IsArtificial)
    *IsArtificial = edge->IsArtificial;
  return (SchedInstruction *)(edge->to);
}

SchedInstruction *SchedInstruction::GetLastScsr(InstCount *prdcsrNum) {
  GraphEdge *edge = GetLastScsrEdge();
  if (!edge)
    return NULL;
  if (prdcsrNum)
    *prdcsrNum = edge->predOrder;
  return (SchedInstruction *)(edge->to);
}

SchedInstruction *SchedInstruction::GetPrevScsr(InstCount *prdcsrNum) {
  GraphEdge *edge = GetPrevScsrEdge();
  if (!edge)
    return NULL;
  if (prdcsrNum)
    *prdcsrNum = edge->predOrder;
  return (SchedInstruction *)(edge->to);
}

SchedInstruction *SchedInstruction::GetFrstNghbr(DIRECTION dir,
                                                 UDT_GLABEL *ltncy) {
  GraphEdge *edge = dir == DIR_FRWRD ? GetFrstScsrEdge() : GetFrstPrdcsrEdge();
  if (edge == NULL)
    return NULL;
  if (ltncy)
    *ltncy = edge->label;
  return (SchedInstruction *)((dir == DIR_FRWRD) ? edge->to : edge->from);
}

SchedInstruction *SchedInstruction::GetNxtNghbr(DIRECTION dir,
                                                UDT_GLABEL *ltncy) {
  GraphEdge *edge = dir == DIR_FRWRD ? GetNxtScsrEdge() : GetNxtPrdcsrEdge();
  if (edge == NULL)
    return NULL;
  if (ltncy)
    *ltncy = edge->label;
  return (SchedInstruction *)((dir == DIR_FRWRD) ? edge->to : edge->from);
}

InstCount SchedInstruction::CmputCrtclPathFrmRoot() {
  crtclPathFrmRoot_ = CmputCrtclPath_(DIR_FRWRD);
  return crtclPathFrmRoot_;
}

InstCount SchedInstruction::CmputCrtclPathFrmLeaf() {
  crtclPathFrmLeaf_ = CmputCrtclPath_(DIR_BKWRD);
  return crtclPathFrmLeaf_;
}

InstCount
SchedInstruction::CmputCrtclPathFrmRcrsvPrdcsr(SchedInstruction *ref) {
  InstCount refInstNum = ref->GetNum();
  crtclPathFrmRcrsvPrdcsr_[refInstNum] = CmputCrtclPath_(DIR_FRWRD, ref);
  return crtclPathFrmRcrsvPrdcsr_[refInstNum];
}

InstCount SchedInstruction::CmputCrtclPathFrmRcrsvScsr(SchedInstruction *ref) {
  InstCount refInstNum = ref->GetNum();
  crtclPathFrmRcrsvScsr_[refInstNum] = CmputCrtclPath_(DIR_BKWRD, ref);
  return crtclPathFrmRcrsvScsr_[refInstNum];
}

InstCount SchedInstruction::GetCrtclPath(DIRECTION dir) const {
  return dir == DIR_FRWRD ? crtclPathFrmRoot_ : crtclPathFrmLeaf_;
}

InstCount SchedInstruction::GetRltvCrtclPath(DIRECTION dir,
                                             SchedInstruction *ref) {
  InstCount refInstNum = ref->GetNum();

  if (dir == DIR_FRWRD) {
    assert(crtclPathFrmRcrsvPrdcsr_[refInstNum] != INVALID_VALUE);
    return crtclPathFrmRcrsvPrdcsr_[refInstNum];
  } else {
    assert(dir == DIR_BKWRD);
    assert(crtclPathFrmRcrsvScsr_[refInstNum] != INVALID_VALUE);
    return crtclPathFrmRcrsvScsr_[refInstNum];
  }
}

InstCount SchedInstruction::GetLwrBound(DIRECTION dir) const {
  return dir == DIR_FRWRD ? frwrdLwrBound_ : bkwrdLwrBound_;
}

void SchedInstruction::SetLwrBound(DIRECTION dir, InstCount bound,
                                   bool isAbslut) {
  if (dir == DIR_FRWRD) {
    assert(!isAbslut || bound >= frwrdLwrBound_);
    frwrdLwrBound_ = bound;

    if (isAbslut) {
      abslutFrwrdLwrBound_ = bound;
      crntRange_->SetFrwrdBound(frwrdLwrBound_);
    }
  } else {
    assert(!isAbslut || bound >= bkwrdLwrBound_);
    bkwrdLwrBound_ = bound;

    if (isAbslut) {
      abslutBkwrdLwrBound_ = bound;
      crntRange_->SetBkwrdBound(bkwrdLwrBound_);
    }
  }
}

void SchedInstruction::RestoreAbsoluteBounds() {
  frwrdLwrBound_ = abslutFrwrdLwrBound_;
  bkwrdLwrBound_ = abslutBkwrdLwrBound_;
  crntRange_->SetBounds(frwrdLwrBound_, bkwrdLwrBound_);
}

void SchedInstruction::SetBounds(InstCount flb, InstCount blb) {
  frwrdLwrBound_ = flb;
  bkwrdLwrBound_ = blb;
  abslutFrwrdLwrBound_ = flb;
  abslutBkwrdLwrBound_ = blb;
  crntRange_->SetBounds(frwrdLwrBound_, bkwrdLwrBound_);
}

bool SchedInstruction::PrdcsrSchduld(InstCount prdcsrNum, InstCount cycle,
                                     InstCount &rdyCycle) {
  assert(prdcsrNum < prdcsrCnt_);
  rdyCyclePerPrdcsr_[prdcsrNum] = cycle + ltncyPerPrdcsr_[prdcsrNum];
  prevMinRdyCyclePerPrdcsr_[prdcsrNum] = minRdyCycle_;

  if (rdyCyclePerPrdcsr_[prdcsrNum] > minRdyCycle_) {
    minRdyCycle_ = rdyCyclePerPrdcsr_[prdcsrNum];
  }

  rdyCycle = minRdyCycle_;
  unschduldPrdcsrCnt_--;
  return (unschduldPrdcsrCnt_ == 0);
}

bool SchedInstruction::PrdcsrUnSchduld(InstCount prdcsrNum,
                                       InstCount &rdyCycle) {
  assert(prdcsrNum < prdcsrCnt_);
  assert(rdyCyclePerPrdcsr_[prdcsrNum] != INVALID_VALUE);
  rdyCycle = minRdyCycle_;
  minRdyCycle_ = prevMinRdyCyclePerPrdcsr_[prdcsrNum];
  rdyCyclePerPrdcsr_[prdcsrNum] = INVALID_VALUE;
  unschduldPrdcsrCnt_++;
  assert(unschduldPrdcsrCnt_ != prdcsrCnt_ || minRdyCycle_ == INVALID_VALUE);
  return (unschduldPrdcsrCnt_ == 1);
}

bool SchedInstruction::ScsrSchduld() {
  unschduldScsrCnt_--;
  return unschduldScsrCnt_ == 0;
}

void SchedInstruction::SetInstType(InstType type) { instType_ = type; }

void SchedInstruction::SetIssueType(IssueType type) { issuType_ = type; }

InstType SchedInstruction::GetInstType() const { return instType_; }

IssueType SchedInstruction::GetIssueType() const { return issuType_; }

bool SchedInstruction::IsSchduld(InstCount *cycle) const {
  if (cycle)
    *cycle = crntSchedCycle_;
  return crntSchedCycle_ != SCHD_UNSCHDULD;
}

InstCount SchedInstruction::GetSchedCycle() const { return crntSchedCycle_; }

InstCount SchedInstruction::GetSchedSlot() const { return crntSchedSlot_; }

InstCount SchedInstruction::GetCrntDeadline() const {
  return IsSchduld() ? crntSchedCycle_ : crntRange_->GetDeadline();
}

InstCount SchedInstruction::GetCrntReleaseTime() const {
  return IsSchduld() ? crntSchedCycle_ : GetCrntLwrBound(DIR_FRWRD);
}

InstCount SchedInstruction::GetRlxdCycle() const {
  return IsSchduld() ? crntSchedCycle_ : crntRlxdCycle_;
}

void SchedInstruction::SetRlxdCycle(InstCount cycle) { crntRlxdCycle_ = cycle; }

void SchedInstruction::Schedule(InstCount cycleNum, InstCount slotNum) {
  assert(crntSchedCycle_ == SCHD_UNSCHDULD);
  crntSchedCycle_ = cycleNum;
  crntSchedSlot_ = slotNum;
}

bool SchedInstruction::IsInReadyList() const { return ready_; }

void SchedInstruction::PutInReadyList() { ready_ = true; }

void SchedInstruction::RemoveFromReadyList() { ready_ = false; }

InstCount SchedInstruction::GetCrntLwrBound(DIRECTION dir) const {
  return crntRange_->GetLwrBound(dir);
}

void SchedInstruction::SetCrntLwrBound(DIRECTION dir, InstCount bound) {
  crntRange_->SetLwrBound(dir, bound);
}

void SchedInstruction::UnSchedule() {
  assert(crntSchedCycle_ != SCHD_UNSCHDULD);
  crntSchedCycle_ = SCHD_UNSCHDULD;
  crntSchedSlot_ = SCHD_UNSCHDULD;
}

void SchedInstruction::UnTightnLwrBounds() { crntRange_->UnTightnLwrBounds(); }

void SchedInstruction::CmtLwrBoundTightnng() {
  crntRange_->CmtLwrBoundTightnng();
}

void SchedInstruction::SetSig(InstSignature sig) { sig_ = sig; }

InstSignature SchedInstruction::GetSig() const { return sig_; }

InstCount SchedInstruction::GetFxdCycle() const {
  assert(crntRange_->IsFxd());
  return crntRange_->GetLwrBound(DIR_FRWRD);
}

bool SchedInstruction::IsFxd() const { return crntRange_->IsFxd(); }

InstCount SchedInstruction::GetPreFxdCycle() const { return preFxdCycle_; }

bool SchedInstruction::TightnLwrBound(DIRECTION dir, InstCount newLwrBound,
                                      LinkedList<SchedInstruction> *tightndLst,
                                      LinkedList<SchedInstruction> *fxdLst,
                                      bool enforce) {
  return crntRange_->TightnLwrBound(dir, newLwrBound, tightndLst, fxdLst,
                                    enforce);
}

bool SchedInstruction::TightnLwrBoundRcrsvly(
    DIRECTION dir, InstCount newLwrBound,
    LinkedList<SchedInstruction> *tightndLst,
    LinkedList<SchedInstruction> *fxdLst, bool enforce) {
  return crntRange_->TightnLwrBoundRcrsvly(dir, newLwrBound, tightndLst, fxdLst,
                                           enforce);
}

bool SchedInstruction::ProbeScsrsCrntLwrBounds(InstCount cycle) {
  if (cycle <= crntRange_->GetLwrBound(DIR_FRWRD))
    return false;

  for (GraphEdge *edg = GetFrstScsrEdge(); edg != NULL;
       edg = GetNxtScsrEdge()) {
    UDT_GLABEL edgLbl = edg->label;
    SchedInstruction *nghbr = (SchedInstruction *)(edg->GetOtherNode(this));
    InstCount nghbrNewLwrBound = cycle + edgLbl;

    // If this neighbor will get delayed by scheduling this instruction in the
    // given cycle.
    if (nghbrNewLwrBound > nghbr->GetCrntLwrBound(DIR_FRWRD))
      return true;
  }

  return false;
}

void SchedInstruction::ComputeAdjustedUseCnt_() {
  adjustedUseCnt_ =
      NumUses() - llvm::count_if(GetUses(), [](const Register *use) {
        return use->IsLiveOut();
      });
}

InstCount SchedInstruction::GetFileSchedOrder() const {
  return fileSchedOrder_;
}

InstCount SchedInstruction::GetFileSchedCycle() const {
  return fileSchedCycle_;
}

void SchedInstruction::SetScsrNums_() {
  InstCount scsrNum = 0;

  for (GraphEdge *edge = GetFrstScsrEdge(); edge != NULL;
       edge = GetNxtScsrEdge()) {
    edge->succOrder = scsrNum++;
  }

  assert(scsrNum == GetScsrCnt());
}

void SchedInstruction::SetPrdcsrNums_() {
  InstCount prdcsrNum = 0;

  for (GraphEdge *edge = GetFrstPrdcsrEdge(); edge != NULL;
       edge = GetNxtPrdcsrEdge()) {
    edge->predOrder = prdcsrNum++;
  }

  assert(prdcsrNum == GetPrdcsrCnt());
}

int16_t SchedInstruction::CmputLastUseCnt() {
  lastUseCnt_ = 0;

  for (int i = 0; i < useCnt_; i++) {
    Register *reg = uses_[i];
    assert(reg->GetCrntUseCnt() < reg->GetUseCnt());
    if (reg->GetCrntUseCnt() + 1 == reg->GetUseCnt())
      lastUseCnt_++;
  }

  return lastUseCnt_;
}

/******************************************************************************
 * SchedRange                                                                 *
 ******************************************************************************/

SchedRange::SchedRange(SchedInstruction *inst) {
  InitVars_();
  inst_ = inst;
  frwrdLwrBound_ = INVALID_VALUE;
  bkwrdLwrBound_ = INVALID_VALUE;
  lastCycle_ = INVALID_VALUE;
}

bool SchedRange::TightnLwrBound(DIRECTION dir, InstCount newBound,
                                LinkedList<SchedInstruction> *tightndLst,
                                LinkedList<SchedInstruction> *fxdLst,
                                bool enforce) {
  InstCount *boundPtr = (dir == DIR_FRWRD) ? &frwrdLwrBound_ : &bkwrdLwrBound_;
  InstCount crntBound = *boundPtr;
  InstCount othrBound = (dir == DIR_FRWRD) ? bkwrdLwrBound_ : frwrdLwrBound_;

  assert(enforce || IsFsbl_());
  assert(newBound > crntBound);
  InstCount boundSum = newBound + othrBound;

  bool fsbl = true;
  if (boundSum > lastCycle_) {
    fsbl = false;
    if (!enforce)
      return false;
  }

  assert(enforce || !inst_->IsSchduld());
  assert(enforce || !isFxd_);

  // If the range equals exactly one cycle.
  if (boundSum == lastCycle_) {
    isFxd_ = true;
    fxdLst->InsrtElmnt(inst_);
  }

  bool *isTightndPtr = (dir == DIR_FRWRD) ? &isFrwrdTightnd_ : &isBkwrdTightnd_;
  bool isTightnd = isFrwrdTightnd_ || isBkwrdTightnd_;
  InstCount *prevBoundPtr =
      (dir == DIR_FRWRD) ? &prevFrwrdLwrBound_ : &prevBkwrdLwrBound_;

  // If this instruction is not already in the tightened instruction list.
  if (!isTightnd || enforce) {
    // Add it to the list.
    tightndLst->InsrtElmnt(inst_);
  }

  // If this particular LB has not been tightened.
  if (!*isTightndPtr && !enforce) {
    *prevBoundPtr = crntBound;
    *isTightndPtr = true;
  }

  // Now change the bound to the new bound.
  *boundPtr = newBound;

  return fsbl;
}

bool SchedRange::TightnLwrBoundRcrsvly(DIRECTION dir, InstCount newBound,
                                       LinkedList<SchedInstruction> *tightndLst,
                                       LinkedList<SchedInstruction> *fxdLst,
                                       bool enforce) {
  auto getNextNeighbor =
      dir == DIR_FRWRD
          ? +[](SchedRange &range) { return range.inst_->GetNxtScsrEdge(); }
          : +[](SchedRange &range) { return range.inst_->GetNxtPrdcsrEdge(); };

  InstCount crntBound = (dir == DIR_FRWRD) ? frwrdLwrBound_ : bkwrdLwrBound_;
  bool fsbl = IsFsbl_();

  assert(enforce || fsbl);
  assert(newBound >= crntBound);

  if (newBound > crntBound) {
    fsbl = TightnLwrBound(dir, newBound, tightndLst, fxdLst, enforce);

    if (!fsbl && !enforce)
      return false;

    for (GraphEdge *edg = dir == DIR_FRWRD ? inst_->GetFrstScsrEdge()
                                           : inst_->GetFrstPrdcsrEdge();
         edg != NULL; edg = getNextNeighbor(*this)) {
      UDT_GLABEL edgLbl = edg->label;
      SchedInstruction *nghbr = (SchedInstruction *)(edg->GetOtherNode(inst_));
      InstCount nghbrNewBound = newBound + edgLbl;

      if (nghbrNewBound > nghbr->GetCrntLwrBound(dir)) {
        bool nghbrFsblty = nghbr->TightnLwrBoundRcrsvly(
            dir, nghbrNewBound, tightndLst, fxdLst, enforce);
        if (!nghbrFsblty) {
          fsbl = false;
          if (!enforce)
            return false;
        }
      }
    }
  }

  assert(enforce || fsbl);
  return fsbl;
}

bool SchedRange::Fix(InstCount cycle, LinkedList<SchedInstruction> *tightndLst,
                     LinkedList<SchedInstruction> *fxdLst) {
  if (cycle < frwrdLwrBound_ || cycle > GetDeadline())
    return false;
  InstCount backBnd = lastCycle_ - cycle;
  return (TightnLwrBoundRcrsvly(DIR_FRWRD, cycle, tightndLst, fxdLst, false) &&
          TightnLwrBoundRcrsvly(DIR_BKWRD, backBnd, tightndLst, fxdLst, false));
}

void SchedRange::SetBounds(InstCount frwrdLwrBound, InstCount bkwrdLwrBound) {
  InitVars_();
  frwrdLwrBound_ = frwrdLwrBound;
  bkwrdLwrBound_ = bkwrdLwrBound;
}

bool SchedRange::SetBounds(InstCount frwrdLwrBound, InstCount bkwrdLwrBound,
                           InstCount schedLngth,
                           LinkedList<SchedInstruction> *fxdLst) {
  InitVars_();
  frwrdLwrBound_ = frwrdLwrBound;
  bkwrdLwrBound_ = bkwrdLwrBound;
  assert(schedLngth != INVALID_VALUE);
  lastCycle_ = schedLngth - 1;

  if (!IsFsbl_())
    return false;

  if (GetLwrBoundSum_() == lastCycle_) {
    isFxd_ = true;
    assert(fxdLst != NULL);
    fxdLst->InsrtElmnt(inst_);
  }

  return true;
}

void SchedRange::InitVars_() {
  prevFrwrdLwrBound_ = INVALID_VALUE;
  prevBkwrdLwrBound_ = INVALID_VALUE;
  isFrwrdTightnd_ = false;
  isBkwrdTightnd_ = false;
  isFxd_ = false;
}

void SchedRange::SetFrwrdBound(InstCount bound) {
  assert(bound >= frwrdLwrBound_);
  frwrdLwrBound_ = bound;
}

void SchedRange::SetBkwrdBound(InstCount bound) {
  assert(bound >= bkwrdLwrBound_);
  bkwrdLwrBound_ = bound;
}

InstCount SchedRange::GetLwrBoundSum_() const {
  return frwrdLwrBound_ + bkwrdLwrBound_;
}

InstCount SchedRange::GetDeadline() const {
  return lastCycle_ - bkwrdLwrBound_;
}

bool SchedRange::IsFsbl_() const { return GetLwrBoundSum_() <= lastCycle_; }

void SchedRange::UnTightnLwrBounds() {
  assert(IsFsbl_());
  assert(isFrwrdTightnd_ || isBkwrdTightnd_);

  if (isFrwrdTightnd_) {
    assert(frwrdLwrBound_ != prevFrwrdLwrBound_);
    frwrdLwrBound_ = prevFrwrdLwrBound_;
    isFrwrdTightnd_ = false;
  }

  if (isBkwrdTightnd_) {
    assert(bkwrdLwrBound_ != prevBkwrdLwrBound_);
    bkwrdLwrBound_ = prevBkwrdLwrBound_;
    isBkwrdTightnd_ = false;
  }

  if (isFxd_)
    isFxd_ = false;
}

void SchedRange::CmtLwrBoundTightnng() {
  assert(isFrwrdTightnd_ || isBkwrdTightnd_);
  isFrwrdTightnd_ = false;
  isBkwrdTightnd_ = false;
}

InstCount SchedRange::GetLwrBound(DIRECTION dir) const {
  return (dir == DIR_FRWRD) ? frwrdLwrBound_ : bkwrdLwrBound_;
}

bool SchedRange::IsFxd() const { return lastCycle_ == GetLwrBoundSum_(); }

void SchedRange::SetLwrBound(DIRECTION dir, InstCount bound) {
  InstCount &crntBound = (dir == DIR_FRWRD) ? frwrdLwrBound_ : bkwrdLwrBound_;
  bool &isTightnd = (dir == DIR_FRWRD) ? isFrwrdTightnd_ : isBkwrdTightnd_;

  if (isFxd_ && bound != crntBound) {
    assert(bound < crntBound);
    isFxd_ = false;
  }

  crntBound = bound;
#ifdef IS_DEBUG
  InstCount crntBoundPtr = (dir == DIR_FRWRD) ? frwrdLwrBound_ : bkwrdLwrBound_;
  assert(crntBoundPtr == bound);
#endif
  isTightnd = false;
}

bool SchedRange::IsTightnd(DIRECTION dir) const {
  return (dir == DIR_FRWRD) ? isFrwrdTightnd_ : isBkwrdTightnd_;
}
