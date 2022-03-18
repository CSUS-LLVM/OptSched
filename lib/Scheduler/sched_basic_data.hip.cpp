#include "opt-sched/Scheduler/sched_basic_data.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/dev_defines.h"

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

__host__
SchedInstruction::SchedInstruction(InstCount num, const char *name,
                                   InstType instType, const char *opCode,
                                   InstCount maxInstCnt, int nodeID,
                                   InstCount fileSchedOrder,
                                   InstCount fileSchedCycle, InstCount fileLB,
                                   InstCount fileUB, MachineModel *model)
    : GraphNode(num, maxInstCnt) {
  // Static data that is computed only once.
  int i = 0;
  do {
    name_[i] = name[i];}
  while (name[i++] != 0);

  i = 0;
  do {
    opCode_[i] = opCode[i];}
  while (opCode[i++] != 0);

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

__host__ __device__
SchedInstruction::SchedInstruction()
  :GraphNode() {}

__host__
SchedInstruction::~SchedInstruction() {
  if (memAllocd_)
    DeAllocMem_();

  delete crntRange_;
}

__host__
void SchedInstruction::SetupForSchdulng(InstCount instCnt, bool isCP_FromScsr,
                                        bool isCP_FromPrdcsr) {
  if (memAllocd_)
    DeAllocMem_();
  AllocMem_(instCnt, isCP_FromScsr, isCP_FromPrdcsr);

  SetPrdcsrNums_();
  SetScsrNums_();
  ComputeAdjustedUseCnt_();
}

__host__
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

__host__ __device__
bool SchedInstruction::InitForSchdulng(InstCount schedLngth,
                                       LinkedList<SchedInstruction> *fxdLst) {
#ifdef __HIP_DEVICE_COMPILE__
  dev_crntSchedCycle_[GLOBALTID] = SCHD_UNSCHDULD;
  dev_lastUseCnt_[GLOBALTID] = 0;
  dev_ready_[GLOBALTID] = false;
  dev_minRdyCycle_[GLOBALTID] = INVALID_VALUE;
  dev_unschduldPrdcsrCnt_[GLOBALTID] = prdcsrCnt_;
  dev_unschduldScsrCnt_[GLOBALTID] = scsrCnt_;
  
  for (InstCount i = 0; i < prdcsrCnt_; i++) {
    dev_rdyCyclePerPrdcsr_[GLOBALTID][i] = INVALID_VALUE;
    dev_prevMinRdyCyclePerPrdcsr_[GLOBALTID][i] = INVALID_VALUE;
  }
#else
  crntSchedCycle_ = SCHD_UNSCHDULD;
  lastUseCnt_ = 0;
  ready_ = false;
  minRdyCycle_ = INVALID_VALUE;
  unschduldPrdcsrCnt_ = prdcsrCnt_;
  unschduldScsrCnt_ = scsrCnt_;
  crntRlxdCycle_ = SCHD_UNSCHDULD;

  for (InstCount i = 0; i < prdcsrCnt_; i++) {
    rdyCyclePerPrdcsr_[i] = INVALID_VALUE;
    prevMinRdyCyclePerPrdcsr_[i] = INVALID_VALUE;
  }
#endif

  if (schedLngth != INVALID_VALUE) {
    bool fsbl = crntRange_->SetBounds(frwrdLwrBound_, bkwrdLwrBound_,
                                      schedLngth, fxdLst);
    if (!fsbl)
      return false;
  }

  return true;
}

__host__
void SchedInstruction::AllocMem_(InstCount instCnt, bool isCP_FromScsr,
                                 bool isCP_FromPrdcsr) {
  scsrCnt_ = GetScsrCnt();
  prdcsrCnt_ = GetPrdcsrCnt();
  rdyCyclePerPrdcsr_ = new InstCount[prdcsrCnt_];
  ltncyPerPrdcsr_ = new InstCount[prdcsrCnt_];
  prevMinRdyCyclePerPrdcsr_ = new InstCount[prdcsrCnt_];
  sortedPrdcsrLst_ = new PriorityArrayList<InstCount>(prdcsrCnt_);

  InstCount predecessorIndex = 0;
  for (GraphEdge *edge = GetFrstPrdcsrEdge(); edge != NULL;
       edge = GetNxtPrdcsrEdge()) {
    ltncyPerPrdcsr_[predecessorIndex++] = edge->label;
    sortedPrdcsrLst_->InsrtElmnt(edge->GetOtherNodeNum(this->GetNum()),
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

__host__
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

__host__ __device__
InstCount SchedInstruction::CmputCrtclPath_(DIRECTION dir,
                                            SchedInstruction *ref) {
  // The idea of this function is considering each predecessor (successor) and
  // calculating the length of the path from the root (leaf) through that
  // predecessor (successor) and then taking the maximum value among all these
  // paths.
  InstCount crtclPath = 0;
  ArrayList<GraphEdge *> *nghbrLst = GetNghbrLst(dir);

  for (GraphEdge *edg = nghbrLst->GetFrstElmnt(); edg != NULL;
       edg = nghbrLst->GetNxtElmnt()) {
    UDT_GLABEL edgLbl = edg->label;
    SchedInstruction *nghbr = (SchedInstruction *)
	    (nodes_[edg->GetOtherNodeNum(this->GetNum())]);

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

__host__ __device__
bool SchedInstruction::ApplyPreFxng(LinkedList<SchedInstruction> *tightndLst,
                                    LinkedList<SchedInstruction> *fxdLst) {
  return crntRange_->Fix(preFxdCycle_, tightndLst, fxdLst);
}

__host__ __device__
void SchedInstruction::AddDef(Register *reg) {
  if (defCnt_ >= MAX_DEFS_PER_INSTR) {
#ifndef __HIP_DEVICE_COMPILE__
    Logger::Fatal("An instruction can't have more than %d defs",
                  MAX_DEFS_PER_INSTR);
#endif
  }
#ifndef __HIP_DEVICE_COMPILE__
/*
  Logger::Info("Inst %d defines reg %d of type %d and physNum %d and useCnt %d",
  GetNum(), reg->GetNum(), reg->GetType(), reg->GetPhysicalNumber(),
  reg->GetUseCnt());
*/
#endif
  assert(reg != NULL);
  defs_[defCnt_].regType_ = (int)reg->GetType();
  defs_[defCnt_].regNum_ = (int)reg->GetNum();
  defCnt_++;
}

__host__ __device__
void SchedInstruction::AddUse(Register *reg) {
  if (useCnt_ >= MAX_USES_PER_INSTR) {
#ifndef __HIP_DEVICE_COMPILE__
    Logger::Fatal("An instruction can't have more than %d uses",
                  MAX_USES_PER_INSTR);
#endif
  }
#ifndef __HIP_DEVICE_COMPILE__
/*
  Logger::Info("Inst %d uses reg %d of type %d and physNum %d and useCnt %d",
  GetNum(), reg->GetNum(), reg->GetType(), reg->GetPhysicalNumber(),
  reg->GetUseCnt());
*/
#endif
  assert(reg != NULL);
  uses_[useCnt_].regType_ = (int)reg->GetType();
  uses_[useCnt_].regNum_ = (int)reg->GetNum();
  useCnt_++;
}

__host__ __device__
bool SchedInstruction::FindDef(Register *reg) const {
  assert(reg != NULL);

  for (int i = 0; i < defCnt_; i++) {
    if (defs_[i].regType_ == reg->GetType() && 
        defs_[i].regNum_ == reg->GetNum())
      return true;
  }

  return false;
}

__host__ __device__
bool SchedInstruction::FindUse(Register *reg) const {
  assert(reg != NULL);

  for (int i = 0; i < useCnt_; i++) {
    if (uses_[i].regType_ == reg->GetType() && 
	uses_[i].regNum_ == reg->GetNum())
      return true;
  }

  return false;
}

__host__ __device__
int16_t SchedInstruction::GetDefs(RegIndxTuple *&defs) {
  defs = (RegIndxTuple *)&defs_;
  return defCnt_;
}

__host__ __device__
int16_t SchedInstruction::GetUses(RegIndxTuple *&uses) {
  uses = (RegIndxTuple *)&uses_;
  return useCnt_;
}

__host__ __device__
bool SchedInstruction::BlocksCycle() const { return blksCycle_; }

__host__ __device__
bool SchedInstruction::IsPipelined() const { return pipelined_; }

__host__ __device__
bool SchedInstruction::MustBeInBBEntry() const {
  return mustBeInBBEntry_;
  //  return opCode_=="CopyFromReg" || opCode_=="ADJCALLSTACKDOWN32";
}

__host__ __device__
bool SchedInstruction::MustBeInBBExit() const {
  return mustBeInBBExit_;
  //  return opCode_=="CopyToReg";
}

__host__ __device__
void SchedInstruction::SetMustBeInBBEntry(bool val) { mustBeInBBEntry_ = val; }

__host__ __device__
void SchedInstruction::SetMustBeInBBExit(bool val) { mustBeInBBExit_ = val; }

__host__ __device__
const char *SchedInstruction::GetName() const { return name_; }

__host__ __device__
const char *SchedInstruction::GetOpCode() const { return opCode_; }

__host__ __device__
int SchedInstruction::GetNodeID() const { return nodeID_; }

__host__ __device__
void SchedInstruction::SetNodeID(int nodeID) { nodeID_ = nodeID; }

__host__ __device__
int SchedInstruction::GetLtncySum() const { return GetScsrLblSum(); }

__host__ __device__
int SchedInstruction::GetMaxLtncy() const { return GetMaxEdgeLabel(); }

__host__ __device__
int16_t SchedInstruction::GetLastUseCnt() { 
#ifdef __HIP_DEVICE_COMPILE__
  return dev_lastUseCnt_[GLOBALTID];
#else
  return lastUseCnt_;
#endif
}

__host__ __device__
SchedInstruction *SchedInstruction::GetFrstPrdcsr(InstCount *scsrNum,
                                                  UDT_GLABEL *ltncy,
                                                  DependenceType *depType,
						  InstCount *toNodeNum) {
  GraphEdge *edge = GetFrstPrdcsrEdge();
  if (!edge)
    return NULL;
  if (scsrNum)
    *scsrNum = edge->succOrder;
  if (ltncy)
    *ltncy = edge->label;
  if (depType)
    *depType = (DependenceType)edge->label2;
  if (toNodeNum)
    *toNodeNum = edge->from;
  return (SchedInstruction *)(nodes_[edge->from]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetNxtPrdcsr(InstCount *scsrNum,
                                                 UDT_GLABEL *ltncy,
                                                 DependenceType *depType,
						 InstCount *toNodeNum) {
  GraphEdge *edge = GetNxtPrdcsrEdge();
  if (!edge)
    return NULL;
  if (scsrNum)
    *scsrNum = edge->succOrder;
  if (ltncy)
    *ltncy = edge->label;
  if (depType)
    *depType = (DependenceType)edge->label2;
  if (toNodeNum)
    *toNodeNum = edge->from;
  return (SchedInstruction *)(nodes_[edge->from]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetFrstScsr(InstCount *prdcsrNum,
                                                UDT_GLABEL *ltncy,
                                                DependenceType *depType,
						InstCount *toNodeNum,
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
  if (toNodeNum)
    *toNodeNum = edge->to;
  if (IsArtificial)
    *IsArtificial = edge->IsArtificial;
  return (SchedInstruction *)(nodes_[edge->to]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetNxtScsr(InstCount *prdcsrNum,
                                               UDT_GLABEL *ltncy,
                                               DependenceType *depType,
					       InstCount *toNodeNum,
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
  if (toNodeNum)
    *toNodeNum = edge->to;
  if (IsArtificial)
    *IsArtificial = edge->IsArtificial;
  return (SchedInstruction *)(nodes_[edge->to]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetLastScsr(InstCount *prdcsrNum) {
  GraphEdge *edge = GetLastScsrEdge();
  if (!edge)
    return NULL;
  if (prdcsrNum)
    *prdcsrNum = edge->predOrder;
  return (SchedInstruction *)(nodes_[edge->to]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetPrevScsr(InstCount *prdcsrNum) {
  GraphEdge *edge = GetPrevScsrEdge();
  if (!edge)
    return NULL;
  if (prdcsrNum)
    *prdcsrNum = edge->predOrder;
  return (SchedInstruction *)(nodes_[edge->to]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetFrstNghbr(DIRECTION dir,
                                                 UDT_GLABEL *ltncy) {
  GraphEdge *edge = dir == DIR_FRWRD ? GetFrstScsrEdge() : GetFrstPrdcsrEdge();
  if (edge == NULL)
    return NULL;
  if (ltncy)
    *ltncy = edge->label;
  return (SchedInstruction *)
	  ((dir == DIR_FRWRD) ? nodes_[edge->to] : nodes_[edge->from]);
}

__host__ __device__
SchedInstruction *SchedInstruction::GetNxtNghbr(DIRECTION dir,
                                                UDT_GLABEL *ltncy) {
  GraphEdge *edge = dir == DIR_FRWRD ? GetNxtScsrEdge() : GetNxtPrdcsrEdge();
  if (edge == NULL)
    return NULL;
  if (ltncy)
    *ltncy = edge->label;
  return (SchedInstruction *)
	  ((dir == DIR_FRWRD) ? nodes_[edge->to] : nodes_[edge->from]);
}

__host__ __device__
InstCount SchedInstruction::CmputCrtclPathFrmRoot() {
  crtclPathFrmRoot_ = CmputCrtclPath_(DIR_FRWRD);
  return crtclPathFrmRoot_;
}

__host__ __device__
InstCount SchedInstruction::CmputCrtclPathFrmLeaf() {
  crtclPathFrmLeaf_ = CmputCrtclPath_(DIR_BKWRD);
  return crtclPathFrmLeaf_;
}

__host__ __device__
InstCount
SchedInstruction::CmputCrtclPathFrmRcrsvPrdcsr(SchedInstruction *ref) {
  InstCount refInstNum = ref->GetNum();
  crtclPathFrmRcrsvPrdcsr_[refInstNum] = CmputCrtclPath_(DIR_FRWRD, ref);
  return crtclPathFrmRcrsvPrdcsr_[refInstNum];
}

__host__ __device__
InstCount SchedInstruction::CmputCrtclPathFrmRcrsvScsr(SchedInstruction *ref) {
  InstCount refInstNum = ref->GetNum();
  crtclPathFrmRcrsvScsr_[refInstNum] = CmputCrtclPath_(DIR_BKWRD, ref);
  return crtclPathFrmRcrsvScsr_[refInstNum];
}

__host__ __device__
InstCount SchedInstruction::GetCrtclPath(DIRECTION dir) const {
  return dir == DIR_FRWRD ? crtclPathFrmRoot_ : crtclPathFrmLeaf_;
}

__host__ __device__
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

__host__ __device__
InstCount SchedInstruction::GetLwrBound(DIRECTION dir) const {
  return dir == DIR_FRWRD ? frwrdLwrBound_ : bkwrdLwrBound_;
}

__host__ __device__
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

__host__ __device__
void SchedInstruction::RestoreAbsoluteBounds() {
  frwrdLwrBound_ = abslutFrwrdLwrBound_;
  bkwrdLwrBound_ = abslutBkwrdLwrBound_;
  crntRange_->SetBounds(frwrdLwrBound_, bkwrdLwrBound_);
}

__host__ __device__
void SchedInstruction::SetBounds(InstCount flb, InstCount blb) {
  frwrdLwrBound_ = flb;
  bkwrdLwrBound_ = blb;
  abslutFrwrdLwrBound_ = flb;
  abslutBkwrdLwrBound_ = blb;
  crntRange_->SetBounds(frwrdLwrBound_, bkwrdLwrBound_);
}

__host__ __device__
bool SchedInstruction::PrdcsrSchduld(InstCount prdcsrNum, InstCount cycle,
                                     InstCount &rdyCycle) {
  assert(prdcsrNum < prdcsrCnt_);
#ifdef __HIP_DEVICE_COMPILE__
  dev_rdyCyclePerPrdcsr_[GLOBALTID][prdcsrNum] = cycle + ltncyPerPrdcsr_[prdcsrNum];
  dev_prevMinRdyCyclePerPrdcsr_[GLOBALTID][prdcsrNum] = dev_minRdyCycle_[GLOBALTID];

  if (dev_rdyCyclePerPrdcsr_[GLOBALTID][prdcsrNum] > dev_minRdyCycle_[GLOBALTID]) {
    dev_minRdyCycle_[GLOBALTID] = dev_rdyCyclePerPrdcsr_[GLOBALTID][prdcsrNum];
  }

  rdyCycle = dev_minRdyCycle_[GLOBALTID];
  dev_unschduldPrdcsrCnt_[GLOBALTID]--;
  return (dev_unschduldPrdcsrCnt_[GLOBALTID] == 0);
#else
  rdyCyclePerPrdcsr_[prdcsrNum] = cycle + ltncyPerPrdcsr_[prdcsrNum];
  prevMinRdyCyclePerPrdcsr_[prdcsrNum] = minRdyCycle_;

  if (rdyCyclePerPrdcsr_[prdcsrNum] > minRdyCycle_) {
    minRdyCycle_ = rdyCyclePerPrdcsr_[prdcsrNum];
  }

  rdyCycle = minRdyCycle_;
  unschduldPrdcsrCnt_--;
  return (unschduldPrdcsrCnt_ == 0);
#endif
}

__host__ __device__
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

__host__ __device__
bool SchedInstruction::ScsrSchduld() {
#ifdef __HIP_DEVICE_COMPILE__
  dev_unschduldScsrCnt_[GLOBALTID]--;
  return dev_unschduldScsrCnt_[GLOBALTID] == 0;
#else
  unschduldScsrCnt_--;
  return unschduldScsrCnt_ == 0;
#endif
}

__host__ __device__
void SchedInstruction::SetInstType(InstType type) { instType_ = type; }

__host__ __device__
void SchedInstruction::SetIssueType(IssueType type) { issuType_ = type; }

__host__ __device__
InstType SchedInstruction::GetInstType() const { return instType_; }

__host__ __device__
IssueType SchedInstruction::GetIssueType() const { return issuType_; }

__host__ __device__
bool SchedInstruction::IsSchduld(InstCount *cycle) const {
#ifdef __HIP_DEVICE_COMPILE__
  if (cycle)
    *cycle = dev_crntSchedCycle_[GLOBALTID];
  return dev_crntSchedCycle_[GLOBALTID] != SCHD_UNSCHDULD;
#else
  if (cycle)
    *cycle = crntSchedCycle_;
  return crntSchedCycle_ != SCHD_UNSCHDULD;
#endif
}

__host__ __device__
InstCount SchedInstruction::GetSchedCycle() const { 
#ifdef __HIP_DEVICE_COMPILE__
  return dev_crntSchedCycle_[GLOBALTID];
#else
  return crntSchedCycle_;
#endif
}

__host__ __device__
InstCount SchedInstruction::GetSchedSlot() const { 
#ifdef __HIP_DEVICE_COMPILE__
  return dev_crntSchedSlot_[GLOBALTID];
#else
  return crntSchedSlot_;
#endif
}

__host__ __device__
InstCount SchedInstruction::GetCrntDeadline() const {
#ifdef __HIP_DEVICE_COMPILE__
  return IsSchduld() ? dev_crntSchedCycle_[GLOBALTID] : crntRange_->GetDeadline();
#else
  return IsSchduld() ? crntSchedCycle_ : crntRange_->GetDeadline();
#endif
}

__host__ __device__
InstCount SchedInstruction::GetCrntReleaseTime() const {
#ifdef __HIP_DEVICE_COMPILE__
  return IsSchduld() ? dev_crntSchedCycle_[GLOBALTID] : GetCrntLwrBound(DIR_FRWRD);
#else
  return IsSchduld() ? crntSchedCycle_ : GetCrntLwrBound(DIR_FRWRD);
#endif
}

__host__ __device__
InstCount SchedInstruction::GetRlxdCycle() const {
#ifdef __HIP_DEVICE_COMPILE__
  return IsSchduld() ? dev_crntSchedCycle_[GLOBALTID] : crntRlxdCycle_;
#else
  return IsSchduld() ? crntSchedCycle_ : crntRlxdCycle_;
#endif
}

__host__ __device__
void SchedInstruction::SetRlxdCycle(InstCount cycle) { crntRlxdCycle_ = cycle; }

__host__ __device__
void SchedInstruction::Schedule(InstCount cycleNum, InstCount slotNum) {
#ifdef __HIP_DEVICE_COMPILE__
  assert(dev_crntSchedCycle_[GLOBALTID] == SCHD_UNSCHDULD);
  dev_crntSchedCycle_[GLOBALTID] = cycleNum;
  dev_crntSchedSlot_[GLOBALTID] = slotNum;
#else
  assert(crntSchedCycle_ == SCHD_UNSCHDULD);
  crntSchedCycle_ = cycleNum;
  crntSchedSlot_ = slotNum;
#endif
}

__host__ __device__
bool SchedInstruction::IsInReadyList() const { 
#ifdef __HIP_DEVICE_COMPILE__
  return dev_ready_[GLOBALTID];
#else
  return ready_;
#endif
}

__host__ __device__
void SchedInstruction::PutInReadyList() { 
#ifdef __HIP_DEVICE_COMPILE__
  dev_ready_[GLOBALTID] = true;
#else
  ready_ = true;
#endif
}

__host__ __device__
void SchedInstruction::RemoveFromReadyList() { 
#ifdef __HIP_DEVICE_COMPILE__
  dev_ready_[GLOBALTID] = false;
#else
  ready_ = false;
#endif
}

__host__ __device__
InstCount SchedInstruction::GetCrntLwrBound(DIRECTION dir) const {
  return crntRange_->GetLwrBound(dir);
}

__host__ __device__
void SchedInstruction::SetCrntLwrBound(DIRECTION dir, InstCount bound) {
  crntRange_->SetLwrBound(dir, bound);
}

__host__ __device__
void SchedInstruction::UnSchedule() {
  assert(crntSchedCycle_ != SCHD_UNSCHDULD);
  crntSchedCycle_ = SCHD_UNSCHDULD;
  crntSchedSlot_ = SCHD_UNSCHDULD;
}

__host__ __device__
void SchedInstruction::UnTightnLwrBounds() { crntRange_->UnTightnLwrBounds(); }

__host__ __device__
void SchedInstruction::CmtLwrBoundTightnng() {
  crntRange_->CmtLwrBoundTightnng();
}

__host__ __device__
void SchedInstruction::SetSig(InstSignature sig) { sig_ = sig; }

__host__ __device__
InstSignature SchedInstruction::GetSig() const { return sig_; }

__host__ __device__
InstCount SchedInstruction::GetFxdCycle() const {
  assert(crntRange_->IsFxd());
  return crntRange_->GetLwrBound(DIR_FRWRD);
}

__host__ __device__
bool SchedInstruction::IsFxd() const { return crntRange_->IsFxd(); }

__host__ __device__
InstCount SchedInstruction::GetPreFxdCycle() const { return preFxdCycle_; }

__host__ __device__
bool SchedInstruction::TightnLwrBound(DIRECTION dir, InstCount newLwrBound,
                                      LinkedList<SchedInstruction> *tightndLst,
                                      LinkedList<SchedInstruction> *fxdLst,
                                      bool enforce) {
  return crntRange_->TightnLwrBound(dir, newLwrBound, tightndLst, fxdLst,
                                    enforce);
}

__host__ __device__
bool SchedInstruction::TightnLwrBoundRcrsvly(
    DIRECTION dir, InstCount newLwrBound,
    LinkedList<SchedInstruction> *tightndLst,
    LinkedList<SchedInstruction> *fxdLst, bool enforce) {
  return crntRange_->TightnLwrBoundRcrsvly(dir, newLwrBound, tightndLst, fxdLst,
                                           enforce);
}

__host__ __device__
bool SchedInstruction::ProbeScsrsCrntLwrBounds(InstCount cycle) {
  if (cycle <= crntRange_->GetLwrBound(DIR_FRWRD))
    return false;

  for (GraphEdge *edg = GetFrstScsrEdge(); edg != NULL;
       edg = GetNxtScsrEdge()) {
    UDT_GLABEL edgLbl = edg->label;
    SchedInstruction *nghbr = (SchedInstruction *)
	    (nodes_[edg->GetOtherNodeNum(this->GetNum())]);
    InstCount nghbrNewLwrBound = cycle + edgLbl;

    // If this neighbor will get delayed by scheduling this instruction in the
    // given cycle.
    if (nghbrNewLwrBound > nghbr->GetCrntLwrBound(DIR_FRWRD))
      return true;
  }

  return false;
}

__host__ __device__
void SchedInstruction::ComputeAdjustedUseCnt_() {
  RegIndxTuple *uses;
  int useCnt = GetUses(uses);
  adjustedUseCnt_ = useCnt;

  for (int i = 0; i < useCnt; i++) {
    if (RegFiles_[uses[i].regType_].GetReg(uses[i].regNum_)->IsLiveOut())
      adjustedUseCnt_--;
  }
}

__host__ __device__
InstCount SchedInstruction::GetFileSchedOrder() const {
  return fileSchedOrder_;
}

__host__ __device__
InstCount SchedInstruction::GetFileSchedCycle() const {
  return fileSchedCycle_;
}

InstCount SchedInstruction::GetFileUB() const {
  return fileUprBound_;
}

InstCount SchedInstruction::GetFileLB() const {
  return fileLwrBound_;
}

__host__ __device__
void SchedInstruction::SetScsrNums_() {
  InstCount scsrNum = 0;

  for (GraphEdge *edge = GetFrstScsrEdge(); edge != NULL;
       edge = GetNxtScsrEdge()) {
    edge->succOrder = scsrNum++;
  }

  assert(scsrNum == GetScsrCnt());
}

__host__ __device__
void SchedInstruction::SetPrdcsrNums_() {
  InstCount prdcsrNum = 0;

  for (GraphEdge *edge = GetFrstPrdcsrEdge(); edge != NULL;
       edge = GetNxtPrdcsrEdge()) {
    edge->predOrder = prdcsrNum++;
  }

  assert(prdcsrNum == GetPrdcsrCnt());
}

__host__ __device__
int16_t SchedInstruction::CmputLastUseCnt() {
#ifdef __HIP_DEVICE_COMPILE__
  dev_lastUseCnt_[GLOBALTID] = 0;
#else
  lastUseCnt_ = 0;
#endif

  for (int i = 0; i < useCnt_; i++) {
    Register *reg = RegFiles_[uses_[i].regType_].GetReg(uses_[i].regNum_);
    assert(reg->GetCrntUseCnt() < reg->GetUseCnt());
    if (reg->GetCrntUseCnt() + 1 == reg->GetUseCnt())
#ifdef __HIP_DEVICE_COMPILE__
      dev_lastUseCnt_[GLOBALTID]++;
#else
      lastUseCnt_++;
#endif
  }
#ifdef __HIP_DEVICE_COMPILE__
  return dev_lastUseCnt_[GLOBALTID];
#else
  return lastUseCnt_;
#endif
}

__host__ __device__
void SchedInstruction::InitializeNode_(InstCount instNum, 
		         const char *const instName,
                         InstType instType, const char *const opCode,
                         InstCount maxNodeCnt, int nodeID, 
			 InstCount fileSchedOrder, InstCount fileSchedCycle, 
			 InstCount fileLB, InstCount fileUB, 
			 MachineModel *model, GraphNode **nodes,
			 RegisterFile *RegFiles) {
  RegFiles_ = RegFiles;
  nodes_ = nodes;
  int i = 0;
  do {
    name_[i] = instName[i];}
  while (instName[i++] != 0);

  i = 0;
  do {
    opCode_[i] = opCode[i];}
  while (opCode[i++] != 0);

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

  // if not on device create a new SchedRange/PrdcsrList/ScsrList
#ifndef __HIP_DEVICE_COMPILE__
  crntRange_ = new SchedRange(this);
  GraphNode::CreatePrdcsrScsrLists(maxNodeCnt);
#endif

  GraphNode::SetNum(instNum);
}

__device__
void SchedInstruction::CreateSchedRange() {
  crntRange_ = new SchedRange(this);
}

void SchedInstruction::CopyPointersToDevice(SchedInstruction *dev_inst,
                                            GraphNode **dev_nodes,
					                                  InstCount instCnt, 
					                                  RegisterFile *dev_regFiles,
                                            int numThreads, 
                                            std::vector<GraphEdge *> *edges,
                                            GraphEdge *dev_edges, 
                                            GraphEdge **dev_scsrElmnts, 
                                            unsigned long *dev_keys, 
                                            int &scsrIndex) {
  dev_inst->RegFiles_ = dev_regFiles;
  size_t memSize;
  memSize = sizeof(InstCount) * prdcsrCnt_;
  // Copy ltncyPerPrdcsr 
  InstCount *dev_ltncyPerPrdcsr;
  gpuErrchk(hipMalloc(&dev_ltncyPerPrdcsr, memSize));
  gpuErrchk(hipMemcpy(dev_ltncyPerPrdcsr, ltncyPerPrdcsr_, memSize, 
		       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_inst->ltncyPerPrdcsr_, &dev_ltncyPerPrdcsr,
		       sizeof(InstCount *), hipMemcpyHostToDevice));
  // Copy SchedRange
  SchedRange *dev_crntRange;
  memSize = sizeof(SchedRange);
  gpuErrchk(hipMalloc(&dev_crntRange, memSize));
  gpuErrchk(hipMemcpy(dev_crntRange, crntRange_, memSize,
		       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_inst->crntRange_, &dev_crntRange,
		       sizeof(SchedRange *), hipMemcpyHostToDevice));
  // Copy sortedScsrLst_
  GraphNode::CopyPointersToDevice((GraphNode *)dev_inst, dev_nodes, instCnt,
                                  edges, dev_edges, dev_scsrElmnts, 
                                  dev_keys, scsrIndex);
  // make sure managed mem is copied to device before kernel start
  memSize = sizeof(InstCount *) * numThreads;
  gpuErrchk(hipMemPrefetchAsync(dev_rdyCyclePerPrdcsr_, memSize, 0));
  memSize = sizeof(InstCount *) * numThreads;
  gpuErrchk(hipMemPrefetchAsync(dev_prevMinRdyCyclePerPrdcsr_, memSize, 0));
}

void SchedInstruction::FreeDevicePointers(int numThreads) {
  hipFree(ltncyPerPrdcsr_);
  hipFree(crntRange_);
  hipFree(dev_crntSchedCycle_);
  hipFree(dev_crntSchedSlot_);
  hipFree(dev_lastUseCnt_);
  hipFree(dev_ready_);
  hipFree(dev_minRdyCycle_);
  hipFree(dev_unschduldPrdcsrCnt_);
  hipFree(dev_unschduldScsrCnt_);
  hipFree(dev_rdyCyclePerPrdcsr_[0]);
  hipFree(dev_rdyCyclePerPrdcsr_);
  hipFree(dev_prevMinRdyCyclePerPrdcsr_[0]);
  hipFree(dev_prevMinRdyCyclePerPrdcsr_);
  if (rcrsvScsrLst_)
    hipFree(rcrsvScsrLst_->dev_crnt_);
  if (rcrsvPrdcsrLst_)
    hipFree(rcrsvPrdcsrLst_->dev_crnt_);
  hipFree(scsrLst_->dev_crnt_);
  GraphNode::FreeDevicePointers();
}

void SchedInstruction::AllocDevArraysForParallelACO(int numThreads) {
  // Create an array of size numThreads for each thread
  // to hold crntSchedCycle_ indexed by its GLOBALTID
  size_t memSize = sizeof(InstCount) * numThreads;
  gpuErrchk(hipMalloc(&dev_crntSchedCycle_, memSize));
  // allocate an array of lastUseCnt_ of size numThreads
  memSize = sizeof(int16_t) * numThreads;
  gpuErrchk(hipMalloc(&dev_lastUseCnt_, memSize));
  // Allocate an array of crntSchedSlot_ of size numThreads
  memSize = sizeof(InstCount) * numThreads;
  gpuErrchk(hipMalloc(&dev_crntSchedSlot_, memSize));
  // Allocate an array of ready_ of size numThreads
  memSize = sizeof(bool) * numThreads;
  gpuErrchk(hipMalloc(&dev_ready_, memSize));
  // Allocate an array of minRdyCycle_ of size numThreads
  memSize = sizeof(InstCount) * numThreads;
  gpuErrchk(hipMalloc(&dev_minRdyCycle_, memSize));
  // Allocate an array of unschduldPrdcsrCnt_ of size numThreads
  memSize = sizeof(InstCount) * numThreads;
  gpuErrchk(hipMalloc(&dev_unschduldPrdcsrCnt_, memSize));
  // Allocate an array of unschduldScsrCnt_ of size numThreads
  memSize = sizeof(InstCount) * numThreads;
  gpuErrchk(hipMalloc(&dev_unschduldScsrCnt_, memSize));
  // Allocate an array of rdyCyclePerPrdcsr_ of size numThreads
  memSize = sizeof(InstCount *) * numThreads;
  gpuErrchk(hipMallocManaged(&dev_rdyCyclePerPrdcsr_, memSize));
  // Trying new approach, malloc one huge array and set every multiple
  // of prdscrCnt_ as another index in the array of arrays
  memSize = sizeof(InstCount) * prdcsrCnt_ * numThreads;
  InstCount *temp_arr;
  gpuErrchk(hipMalloc(&temp_arr, memSize));
  for (int i = 0; i < numThreads; i++) {
    dev_rdyCyclePerPrdcsr_[i] = &temp_arr[i * prdcsrCnt_];
  }
  // Allocate an array of prevMinRdyCyclePerPrdcsr_ of size numThreads
  memSize = sizeof(InstCount *) * numThreads;
  gpuErrchk(hipMallocManaged(&dev_prevMinRdyCyclePerPrdcsr_, memSize));
  // use same one malloc approach as above
  memSize = sizeof(InstCount) * prdcsrCnt_ * numThreads;
  gpuErrchk(hipMalloc(&temp_arr, memSize));
  for (int i = 0; i < numThreads; i++) {
    dev_prevMinRdyCyclePerPrdcsr_[i] = &temp_arr[i * prdcsrCnt_];
  }
  // Alloc arrays of iterators for each array list
  // Alloc array of iterators for sortedScsrLst_
  memSize = sizeof(int) * numThreads;
  if (sortedScsrLst_)  
    gpuErrchk(hipMalloc(&sortedScsrLst_->dev_crnt_, memSize));
  if (sortedPrdcsrLst_)
    gpuErrchk(hipMalloc(&sortedPrdcsrLst_->dev_crnt_, memSize));
  if (rcrsvScsrLst_)
    gpuErrchk(hipMalloc(&rcrsvScsrLst_->dev_crnt_, memSize));
  if (rcrsvPrdcsrLst_)
    gpuErrchk(hipMalloc(&rcrsvPrdcsrLst_->dev_crnt_, memSize));
  gpuErrchk(hipMalloc(&scsrLst_->dev_crnt_, memSize));
}

/******************************************************************************
 * SchedRange                                                                 *
 ******************************************************************************/

__host__ __device__
SchedRange::SchedRange(SchedInstruction *inst) {
  InitVars_();
  inst_ = inst;
  frwrdLwrBound_ = INVALID_VALUE;
  bkwrdLwrBound_ = INVALID_VALUE;
  lastCycle_ = INVALID_VALUE;
}

__host__ __device__
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

__host__ __device__
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
      SchedInstruction *nghbr = (SchedInstruction *)
	      (inst_->nodes_[edg->GetOtherNodeNum(inst_->GetNum())]);
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

__host__ __device__
bool SchedRange::Fix(InstCount cycle, LinkedList<SchedInstruction> *tightndLst,
                     LinkedList<SchedInstruction> *fxdLst) {
  if (cycle < frwrdLwrBound_ || cycle > GetDeadline())
    return false;
  InstCount backBnd = lastCycle_ - cycle;
  return (TightnLwrBoundRcrsvly(DIR_FRWRD, cycle, tightndLst, fxdLst, false) &&
          TightnLwrBoundRcrsvly(DIR_BKWRD, backBnd, tightndLst, fxdLst, false));
}

__host__ __device__
void SchedRange::SetBounds(InstCount frwrdLwrBound, InstCount bkwrdLwrBound) {
  InitVars_();
  frwrdLwrBound_ = frwrdLwrBound;
  bkwrdLwrBound_ = bkwrdLwrBound;
}

__host__ __device__
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

__host__ __device__
void SchedRange::InitVars_() {
  prevFrwrdLwrBound_ = INVALID_VALUE;
  prevBkwrdLwrBound_ = INVALID_VALUE;
  isFrwrdTightnd_ = false;
  isBkwrdTightnd_ = false;
  isFxd_ = false;
}

__host__ __device__
void SchedRange::SetFrwrdBound(InstCount bound) {
  assert(bound >= frwrdLwrBound_);
  frwrdLwrBound_ = bound;
}

__host__ __device__
void SchedRange::SetBkwrdBound(InstCount bound) {
  assert(bound >= bkwrdLwrBound_);
  bkwrdLwrBound_ = bound;
}

__host__ __device__
InstCount SchedRange::GetLwrBoundSum_() const {
  return frwrdLwrBound_ + bkwrdLwrBound_;
}

__host__ __device__
InstCount SchedRange::GetDeadline() const {
  return lastCycle_ - bkwrdLwrBound_;
}

__host__ __device__
bool SchedRange::IsFsbl_() const { return GetLwrBoundSum_() <= lastCycle_; }

__host__ __device__
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

__host__ __device__
void SchedRange::CmtLwrBoundTightnng() {
  assert(isFrwrdTightnd_ || isBkwrdTightnd_);
  isFrwrdTightnd_ = false;
  isBkwrdTightnd_ = false;
}

__host__ __device__
InstCount SchedRange::GetLwrBound(DIRECTION dir) const {
  return (dir == DIR_FRWRD) ? frwrdLwrBound_ : bkwrdLwrBound_;
}

__host__ __device__
bool SchedRange::IsFxd() const { return lastCycle_ == GetLwrBoundSum_(); }

__host__ __device__
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

__host__ __device__
bool SchedRange::IsTightnd(DIRECTION dir) const {
  return (dir == DIR_FRWRD) ? isFrwrdTightnd_ : isBkwrdTightnd_;
}
