#include "opt-sched/Scheduler/relaxed_sched.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/utilities.h"
#include <algorithm>

using namespace llvm::opt_sched;

RelaxedScheduler::RelaxedScheduler(DataDepStruct *dataDepGraph,
                                   MachineModel *machMdl,
                                   InstCount schedUprBound, DIRECTION mainDir,
                                   RLXD_SCHED_TYPE schedType,
                                   InstCount maxInstCnt)
    : InstScheduler(dataDepGraph, machMdl, schedUprBound) {
  dataDepGraph_ = dataDepGraph;
  mainDir_ = mainDir;
  schedDir_ = mainDir_;
  schedType_ = schedType;
  useFxng_ = (schedType == RST_DYNMC);

  maxInstCnt_ = totInstCnt_;

  if (maxInstCnt != INVALID_VALUE) {
    maxInstCnt_ = maxInstCnt;
  }

  // Write the bounds back only if it is a full graph
  writeBack_ = dataDepGraph_->GetType() == DGT_FULL;

  instCntBits_ = Utilities::clcltBitsNeededToHoldNum(maxInstCnt_);

  // If scheduling in the reverse direction.
  if (mainDir_ == DIR_BKWRD) {
    // Swap the root and leaf insts.
    SchedInstruction *tmpInst;
    tmpInst = rootInst_;
    rootInst_ = leafInst_;
    leafInst_ = tmpInst;
  }

  instLst_ = new PriorityList<SchedInstruction>(maxInstCnt_);

  for (InstCount i = 0; i < issuTypeCnt_; i++) {
    avlblSlots_[i] = new int16_t[schedUprBound_];
    nxtAvlblCycles_[i] = new InstCount[schedUprBound_];
  }

  isFxd_ = NULL;

  if (useFxng_) {
    isFxd_ = new bool[maxInstCnt_];

    for (InstCount i = 0; i < maxInstCnt_; i++) {
      isFxd_[i] = false;
    }
  }

  if (schedType_ == RST_DYNMC) {
    for (InstCount i = 0; i < issuTypeCnt_; i++) {
      prevAvlblSlots_[i] = new int16_t[schedUprBound_];
    }
  }

  frwrdLwrBounds_ = NULL;
  bkwrdLwrBounds_ = NULL;
  fxdInstCnt_ = 0;
  schduldInstCnt_ = 0;
  dataDepGraph_->GetLwrBounds(frwrdLwrBounds_, bkwrdLwrBounds_);

#ifdef IS_DEBUG
  wasLwrBoundCmputd_ = new bool[maxInstCnt_];
#endif
}
/*****************************************************************************/

RelaxedScheduler::~RelaxedScheduler() {
  assert(instLst_ != NULL);
  delete instLst_;

  if (schedType_ == RST_DYNMC && useFxng_) {
    delete[] isFxd_;
  }

  for (int16_t i = 0; i < issuTypeCnt_; i++) {
    delete[] avlblSlots_[i];
    delete[] nxtAvlblCycles_[i];

    if (schedType_ == RST_DYNMC) {
      delete[] prevAvlblSlots_[i];
    }
  }

#ifdef IS_DEBUG
  delete[] wasLwrBoundCmputd_;
#endif
}
/*****************************************************************************/

void RelaxedScheduler::Reset_(InstCount startIndx) {
  fxdInstCnt_ = 0;
  schduldInstCnt_ = 0;

  for (int i = 0; i < issuTypeCnt_; i++) {
    for (InstCount j = startIndx; j < schedUprBound_; j++) {
      avlblSlots_[i][j] = (int16_t)slotsPerTypePerCycle_[i];
      nxtAvlblCycles_[i][j] = j;
    }
  }

  instLst_->ResetIterator();
}
/*****************************************************************************/

InstCount RelaxedScheduler::SchdulInst_(SchedInstruction *inst,
                                        InstCount minCycle, InstCount) {
  assert(inst != NULL);
  InstCount releaseTime = CmputReleaseTime_(inst);
  assert(releaseTime == GetCrntLwrBound_(inst, schedDir_));
  assert(GetCrntLwrBound_(rootInst_, mainDir_) == 0);
  assert(releaseTime < schedUprBound_);
  assert(releaseTime >= minCycle);
  IssueType issuType = inst->GetIssueType();
  assert(0 <= issuType && issuType < issuTypeCnt_);

  InstCount schedCycle = nxtAvlblCycles_[issuType][releaseTime];
  assert(minCycle <= schedCycle && schedCycle < schedUprBound_);

  if (avlblSlots_[issuType][schedCycle] == 0) {
    schedCycle = FindNxtAvlblCycle_(issuType, schedCycle + 1);
  }

  assert(minCycle <= schedCycle && schedCycle < schedUprBound_);
  nxtAvlblCycles_[issuType][releaseTime] = schedCycle;
  assert(avlblSlots_[issuType][schedCycle] > 0);
  avlblSlots_[issuType][schedCycle]--;
  assert(schedCycle >= releaseTime);
  return schedCycle;
}
/*****************************************************************************/

InstCount RelaxedScheduler::CmputReleaseTime_(SchedInstruction *inst) {
  return GetCrntLwrBound_(inst, schedDir_);
}
/*****************************************************************************/

InstCount RelaxedScheduler::GetCrntLwrBound_(SchedInstruction *inst,
                                             DIRECTION dir) {
  assert(inst != NULL);
  //  InstCount indx=inst->GetCrntIndx();
  InstCount indx = dataDepGraph_->GetInstIndx(inst);
  assert(indx < totInstCnt_);
  assert(dataDepGraph_->GetType() == DGT_SUB || indx == inst->GetNum());

  if (dir == DIR_FRWRD) {
    assert(frwrdLwrBounds_ != NULL);
    return frwrdLwrBounds_[indx];
  } else {
    assert(bkwrdLwrBounds_ != NULL);
    return bkwrdLwrBounds_[indx];
  }
}
/*****************************************************************************/

void RelaxedScheduler::SetCrntLwrBound_(SchedInstruction *inst, DIRECTION dir,
                                        InstCount newBound) {
  assert(inst != NULL);
  //  InstCount indx=inst->GetCrntIndx();
  InstCount indx = dataDepGraph_->GetInstIndx(inst);

  assert(indx < totInstCnt_);
  assert(dataDepGraph_->GetType() == DGT_SUB || indx == inst->GetNum());
#ifdef IS_DEBUG
  //  wasLwrBoundCmputd_[inst->GetCrntIndx()]=true;
  wasLwrBoundCmputd_[dataDepGraph_->GetInstIndx(inst)] = true;
#endif

  if (dir == DIR_FRWRD) {
    assert(frwrdLwrBounds_ != NULL);
    assert(newBound >= frwrdLwrBounds_[indx]);
    frwrdLwrBounds_[indx] = newBound;
  } else {
    assert(bkwrdLwrBounds_ != NULL);
    assert(newBound >= bkwrdLwrBounds_[indx]);
    bkwrdLwrBounds_[indx] = newBound;
  }
}
/*****************************************************************************/

InstCount RelaxedScheduler::PropagateLwrBound_(SchedInstruction *inst,
                                               DIRECTION dir) {
  InstCount crntBound = GetCrntLwrBound_(inst, dir);

  UDT_GLABEL ltncy;
  DependenceType depType;
  SchedInstruction *pred;
  SchedInstruction *scsr;

  if (dir == DIR_FRWRD) {
    for (pred = inst->GetFrstPrdcsr(NULL, &ltncy, &depType); pred != NULL;
         pred = inst->GetNxtPrdcsr(NULL, &ltncy, &depType)) {
      if (dataDepGraph_->IsInGraph(pred)) {
        InstCount predBound = GetCrntLwrBound_(pred, dir);

        if ((predBound + ltncy) > crntBound) {
          crntBound = predBound + ltncy;
        }
      }
    }
  } else {
    for (scsr = inst->GetFrstScsr(NULL, &ltncy, &depType); scsr != NULL;
         scsr = inst->GetNxtScsr(NULL, &ltncy, &depType)) {
      if (dataDepGraph_->IsInGraph(scsr)) {
        InstCount scsrBound = GetCrntLwrBound_(scsr, dir);

        if ((scsrBound + ltncy) > crntBound) {
          crntBound = scsrBound + ltncy;
        }
      }
    }
  }

  return crntBound;
}
/*****************************************************************************/

void RelaxedScheduler::WriteBoundsBack_() {
  InstCount i;
  SchedInstruction *inst;

  for (i = 0; i < totInstCnt_; i++) {
    inst = dataDepGraph_->GetInstByIndx(i);
    assert(inst != NULL);
#ifdef IS_DEBUG
    //    wasLwrBoundCmputd_[inst->GetCrntIndx()]=true;
    wasLwrBoundCmputd_[dataDepGraph_->GetInstIndx(inst)] = true;
#endif
    InstCount newBound = GetCrntLwrBound_(inst, mainDir_);
    assert(newBound >= inst->GetLwrBound(mainDir_));
    inst->SetLwrBound(mainDir_, newBound);
  }
}
/*****************************************************************************/

void RelaxedScheduler::ClearFxng() {
  for (InstCount i = 0; i < totInstCnt_; i++) {
    SetFix_(i, false);
  }
}
/*****************************************************************************/

RJ_RelaxedScheduler::RJ_RelaxedScheduler(
    DataDepStruct *dataDepGraph, MachineModel *machMdl, InstCount schedUprBound,
    DIRECTION mainDir, RLXD_SCHED_TYPE type, InstCount maxInstCnt)
    : RelaxedScheduler(dataDepGraph, machMdl, schedUprBound, mainDir, type,
                       maxInstCnt) {
  assert(instLst_->GetElmntCnt() == 0);
}
/*****************************************************************************/

RJ_RelaxedScheduler::~RJ_RelaxedScheduler() {}
/*****************************************************************************/

void RJ_RelaxedScheduler::Initialize_(bool setPrirtyLst) {
  totInstCnt_ = dataDepGraph_->GetInstCnt();
  rootInst_ = dataDepGraph_->GetRootInst();
  leafInst_ = dataDepGraph_->GetLeafInst();

  if (mainDir_ == DIR_BKWRD)
  // If scheduling in the reverse direction
  {
    // Swap the root and leaf insts.
    SchedInstruction *tmpInst;
    tmpInst = rootInst_;
    rootInst_ = leafInst_;
    leafInst_ = tmpInst;
  }

  dataDepGraph_->GetLwrBounds(frwrdLwrBounds_, bkwrdLwrBounds_);

  if (setPrirtyLst) {
    SetupPrirtyLst();
  }

  Reset_(0);
  chkdInstCnt_ = 0;

  if (useFxng_)
    for (InstCount i = 0; i < totInstCnt_; i++) {
      SetFix_(i, false);
    }

  instLst_->ResetIterator();
}
/*****************************************************************************/

void RJ_RelaxedScheduler::SetupPrirtyLst() {
  InstCount i;
  DIRECTION opstDir = DirAcycGraph::ReverseDirection(schedDir_);

  instLst_->Reset();

  for (i = 0; i < totInstCnt_; i++) {
    SchedInstruction *inst;

    if (schedDir_ == DIR_FRWRD) {
      inst = dataDepGraph_->GetInstByTplgclOrdr(i);
    } else {
      inst = dataDepGraph_->GetInstByRvrsTplgclOrdr(i);
    }

    // The right topological order will speed up the linear insertion
    // into the sorted list
    if (dataDepGraph_->IsInGraph(inst)) {
      instLst_->InsrtElmnt(inst, GetCrntLwrBound_(inst, opstDir), true);
    }
  }
}
/*****************************************************************************/

InstCount RJ_RelaxedScheduler::FindSchedule() {
  SchedInstruction *inst;
  InstCount delay;
  InstCount maxDelay = 0;
  InstCount schedCycle;
  InstCount trgtLastCycle;
  DIRECTION opstDir = DirAcycGraph::ReverseDirection(schedDir_);

  Initialize_(true);

  assert(GetCrntLwrBound_(rootInst_, opstDir) ==
         GetCrntLwrBound_(leafInst_, schedDir_));

  trgtLastCycle = GetCrntLwrBound_(leafInst_, schedDir_);

  assert(instLst_->GetElmntCnt() == totInstCnt_);

  while (!IsSchedComplete_()) {
    inst = instLst_->GetNxtPriorityElmnt();
    assert(inst != NULL);
    schedCycle = SchdulInst_(inst, 0, trgtLastCycle);
    schduldInstCnt_++;
    delay =
        CmputDelay_(schedCycle, trgtLastCycle, GetCrntLwrBound_(inst, opstDir));

    if (delay > maxDelay) {
      maxDelay = delay;
    }
  }

  return (trgtLastCycle + maxDelay + 1);
  // Return the relaxed schedule length, which is equal to
  // The target last cycle + the maximum delay + 1
}
/*****************************************************************************/

bool RJ_RelaxedScheduler::SchdulAndChkFsblty(InstCount crntCycle,
                                             InstCount lastCycle) {
  SchedInstruction *inst;
  InstCount delay;
  bool fsbl = true;
  InstCount schedCycle;
  DIRECTION opstDir = DirAcycGraph::ReverseDirection(schedDir_);

  assert(schedType_ == RST_DYNMC && useFxng_);
  InitChkng_(crntCycle);
#ifdef IS_DEBUG
  InstCount iterNum = 0;
#endif
  assert(instLst_->GetElmntCnt() == totInstCnt_);

  while (!IsSchedComplete_()) {
    inst = instLst_->GetNxtPriorityElmnt();
#ifdef IS_DEBUG
    assert(iterNum < totInstCnt_);
    iterNum++;
#endif
    assert(inst != NULL);

    if (GetFix_(inst)) {
      inst->SetRlxdCycle(inst->GetCrntReleaseTime());
      continue;
    }

    assert(inst->IsSchduld() == false);
    schedCycle = SchdulInst_(inst, crntCycle, lastCycle);
    inst->SetRlxdCycle(schedCycle);
    schduldInstCnt_++;
    chkdInstCnt_++;
    delay = CmputDelay_(schedCycle, lastCycle, GetCrntLwrBound_(inst, opstDir));

#ifdef IS_DEBUG_RJ
    Logger::Info("Relax scheduled inst. #%d with bounds [%d, %d] in cycle %d "
                 "and delay %d",
                 inst->GetNum(), inst->GetCrntReleaseTime(),
                 inst->GetCrntDeadline(), schedCycle, delay);
#endif

    if (delay > 0) {
      fsbl = false;
    } else {
      assert(GetFix_(inst) == false);
      fsbl = true;
    }

    if (fsbl == false) {
      break;
    }
  }

  EndChkng_(crntCycle);
  return fsbl;
}
/*****************************************************************************/

bool RJ_RelaxedScheduler::CmputDynmcLwrBound(InstCount trgtLastCycle,
                                             InstCount trgtLwrBound,
                                             InstCount &schedLwrBound) {
  DIRECTION opstDir = DirAcycGraph::ReverseDirection(schedDir_);

  Reset_(0);

  assert(GetCrntLwrBound_(rootInst_, schedDir_) == 0);
  // InstCount minLastCycle = GetCrntLwrBound_(leafInst_, schedDir_);
  assert(trgtLastCycle < schedUprBound_);
  assert(GetCrntLwrBound_(leafInst_, schedDir_) <= (trgtLwrBound - 1));
  //  InstCount trgtMaxDelay=trgtLwrBound-1-minLastCycle;
  //  InstCount maxDelay=0;
  InstCount lastCycle = 0;
  schedLwrBound = INVALID_VALUE;

  //  assert(instLst_->GetElmntCnt()==totInstCnt_);
  InstCount instCnt = instLst_->GetElmntCnt();
  schduldInstCnt_ = 0;

  //  while(!IsSchedComplete_())
  while (schduldInstCnt_ < instCnt) {
    SchedInstruction *inst = instLst_->GetNxtPriorityElmnt();
    assert(inst != NULL);

    assert(dataDepGraph_->IsInGraph(inst));
    /*    if(dataDepGraph_->IsInGraph(inst)==false)
        {
          schduldInstCnt_++;
          continue;
        }*/

    assert(GetCrntLwrBound_(inst, schedDir_) +
               GetCrntLwrBound_(inst, opstDir) <=
           trgtLastCycle);
    InstCount schedCycle = SchdulInst_(inst, 0, trgtLastCycle);

    schduldInstCnt_++;
    InstCount delay =
        CmputDelay_(schedCycle, trgtLastCycle, GetCrntLwrBound_(inst, opstDir));

    /*
    Logger::Info("Inst %d: FLB=%d, BLB=%d, DLF=%d, schedCycle=%d",
                 inst->GetNum(),
                 GetCrntLwrBound_(inst, schedDir_),
                 GetCrntLwrBound_(inst, opstDir),
                 dataDepGraph_->GetDistFrmLeaf(inst),
                 schedCycle);
    */

    assert(delay >= 0);

    if (delay > 0) {
      return false;
    }

    if (inst == leafInst_) {
      lastCycle = schedCycle;
    }

    /*    delay=CmputDelay_(schedCycle,GetCrntLwrBound_(leafInst_,schedDir_),
                        dataDepGraph_->GetDistFrmLeaf(inst));

        if(delay > trgtMaxDelay)
          return false;
        if(delay > maxDelay)
          maxDelay=delay;*/
  }

  assert(lastCycle <= trgtLastCycle);
  //  schedLwrBound=minLastCycle+maxDelay+1;
  schedLwrBound = lastCycle + 1;
  return true;
}
/*****************************************************************************/

bool RJ_RelaxedScheduler::FixInsts_(LinkedList<SchedInstruction> *fxdLst) {
  SchedInstruction *inst;
  assert(useFxng_);

  /*  for(InstCount i=0;i<totInstCnt_;i++)
      SetFix_(i,false);*/

  for (inst = fxdLst->GetFrstElmnt(); inst != NULL;
       inst = fxdLst->GetNxtElmnt()) {
    InstCount fxdCycle = GetCrntLwrBound_(inst, mainDir_);
    //    assert(inst->IsSchduld()==false || fxdCycle==inst->GetSchedCycle());
    assert(GetFix_(inst));
    bool isFsbl = FixInst(inst, fxdCycle);

    if (isFsbl == false) {
      return false;
    }
  }

  return true;
}
/*****************************************************************************/

void RJ_RelaxedScheduler::InitChkng_(InstCount crntCycle) {
  int16_t i;
  InstCount j;

  assert(schedType_ == RST_DYNMC);

  for (i = 0; i < issuTypeCnt_; i++) {
    for (j = crntCycle; j < schedUprBound_; j++) {
      assert(prevAvlblSlots_[i] != NULL);
      prevAvlblSlots_[i][j] = avlblSlots_[i][j];
      nxtAvlblCycles_[i][j] = j;
    }
  }

  chkdInstCnt_ = 0;
  instLst_->ResetIterator();
}
/*****************************************************************************/

void RJ_RelaxedScheduler::EndChkng_(InstCount crntCycle) {
  int16_t i;
  InstCount j;

  assert(schedType_ == RST_DYNMC);

  for (i = 0; i < issuTypeCnt_; i++) {
    for (j = crntCycle; j < schedUprBound_; j++) {
      assert(prevAvlblSlots_[i] != NULL);
      avlblSlots_[i][j] = prevAvlblSlots_[i][j];
    }
  }

  schduldInstCnt_ -= chkdInstCnt_;
}
/*****************************************************************************/

bool RJ_RelaxedScheduler::FixInst(SchedInstruction *inst, InstCount cycle) {
  assert(inst != NULL);
  assert(useFxng_);
  assert(GetCrntLwrBound_(inst, schedDir_) == cycle && cycle < schedUprBound_);
  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);

  if (avlblSlots_[issuType][cycle] == 0) {
    return false;
  }

  assert(avlblSlots_[issuType][cycle] > 0);
  avlblSlots_[issuType][cycle]--;
  fxdInstCnt_++;
  schduldInstCnt_++;
  SetFix_(inst, true);
  return true;
}
/*****************************************************************************/

void RJ_RelaxedScheduler::UnFixInst(SchedInstruction *inst, InstCount cycle) {
  assert(inst != NULL);
  assert(schedType_ == RST_DYNMC);

  if (GetFix_(inst) == false) {
    return;
  }

  IssueType issuType = inst->GetIssueType();
  assert(issuType < issuTypeCnt_);

  avlblSlots_[issuType][cycle]++;
  assert(avlblSlots_[issuType][cycle] <= slotsPerTypePerCycle_[issuType]);
  SetFix_(inst, false);
  fxdInstCnt_--;
  schduldInstCnt_--;
}
/*****************************************************************************/

LC_RelaxedScheduler::LC_RelaxedScheduler(DataDepStruct *dataDepGraph,
                                         MachineModel *machMdl,
                                         InstCount schedUprBound,
                                         DIRECTION mainDir)
    : RelaxedScheduler(dataDepGraph, machMdl, schedUprBound, mainDir, RST_STTC,
                       INVALID_VALUE) {
  // TEMP: Support for dynamic scheduling has not been implemented yet
  assert(schedType_ == RST_STTC);

  subGraphInstLst_ = new PriorityList<SchedInstruction>;

  schedDir_ = mainDir_;
}
/*****************************************************************************/

LC_RelaxedScheduler::~LC_RelaxedScheduler() {
  assert(subGraphInstLst_ != NULL);
  delete subGraphInstLst_;
}
/*****************************************************************************/

void LC_RelaxedScheduler::Initialize_() {
  //  dataDepGraph_->SetInstIndexes();
  dataDepGraph_->GetLwrBounds(frwrdLwrBounds_, bkwrdLwrBounds_);
#ifdef IS_DEBUG

  for (InstCount i = 0; i < totInstCnt_; i++) {
    wasLwrBoundCmputd_[i] = false;
  }

#endif
}
/*****************************************************************************/

void LC_RelaxedScheduler::InitSubGraph_() {
  Reset_(0);
  subGraphInstLst_->Reset();
}
/*****************************************************************************/

InstCount LC_RelaxedScheduler::FindSchedule() {
  SchedInstruction *inst = NULL;
  InstCount rcrsvLwrBound = 0;

  Initialize_();

  for (InstCount i = 0; i < totInstCnt_; i++) {
    if (mainDir_ == DIR_FRWRD) {
      inst = dataDepGraph_->GetInstByTplgclOrdr(i);
    } else {
      inst = dataDepGraph_->GetInstByRvrsTplgclOrdr(i);
    }

    assert(inst != NULL);
    assert(inst == rootInst_ || i > 0);
    rcrsvLwrBound = SchdulSubGraph_(inst, schedDir_);
    /*
        #ifdef IS_DEBUG
          if(schedDir_==DIR_FRWRD) {
            Logger::Info("LB of inst %d is %d", inst->GetNum(), rcrsvLwrBound);
          }
        #endif
    */
    SetCrntLwrBound_(inst, schedDir_, rcrsvLwrBound);
  }

  assert(inst == leafInst_);

  if (writeBack_) {
    WriteBoundsBack_();
  }

  return rcrsvLwrBound + 1;
  // The recursive lower bound on the leaf node's scheduling cycle plus
  // one is a lower bound on the total schedule length
}
/*****************************************************************************/

InstCount LC_RelaxedScheduler::SchdulSubGraph_(SchedInstruction *leaf,
                                               DIRECTION schedDir) {
  SchedInstruction *inst = NULL;
  InstCount delay;
  InstCount maxDelay = 0;
  InstCount schedCycle;
  InstCount trgtLastCycle;
  DIRECTION trvrslDir = DirAcycGraph::ReverseDirection(schedDir);
  LinkedList<GraphNode> *rcrsvPrdcsrLst = leaf->GetRcrsvNghbrLst(trvrslDir);
  GraphNode *node;
  InstCount rltvCP;
  DEP_GRAPH_TYPE graphType = dataDepGraph_->GetType();

  if (leaf == rootInst_) {
    return 0;
  }

  InitSubGraph_();
  //  trgtLastCycle=GetCrntLwrBound_(leaf,schedDir);
  trgtLastCycle = CmputReleaseTime_(leaf);
  /*
      #ifdef IS_DEBUG
        if(schedDir_==DIR_FRWRD && leaf->GetNum()==6) {
          Logger::Info("trgtLastCycle=%d", trgtLastCycle);
        }
      #endif
  */

  // Visit the nodes in topological order
  assert(graphType == DGT_SUB || rcrsvPrdcsrLst->GetFrstElmnt() == rootInst_);

  for (node = rcrsvPrdcsrLst->GetFrstElmnt(); node != NULL;
       node = rcrsvPrdcsrLst->GetNxtElmnt()) {
    inst = (SchedInstruction *)node;
    assert(graphType == DGT_FULL || inst != rootInst_);

    if (dataDepGraph_->IsInGraph(inst) == false) {
      continue;
    }

    //    assert(wasLwrBoundCmputd_[inst->GetCrntIndx()]==true);
    assert(wasLwrBoundCmputd_[dataDepGraph_->GetInstIndx(inst)]);
    rltvCP = dataDepGraph_->GetRltvCrtclPath(leaf, inst, trvrslDir);
    subGraphInstLst_->InsrtElmnt(inst, rltvCP, true);
  }

  if (graphType == DGT_SUB) {
    subGraphInstLst_->InsrtElmnt(rootInst_, trgtLastCycle, true);
  }

  subGraphInstLst_->InsrtElmnt(leaf, 0, true);
  assert(subGraphInstLst_->GetElmntCnt() <= totInstCnt_);
  assert(leaf != leafInst_ || subGraphInstLst_->GetElmntCnt() == totInstCnt_);

  for (inst = subGraphInstLst_->GetFrstElmnt(); inst != NULL;
       inst = subGraphInstLst_->GetNxtElmnt()) {
    //    assert(inst==leaf || wasLwrBoundCmputd_[inst->GetCrntIndx()]==true);
    assert(inst == leaf ||
           wasLwrBoundCmputd_[dataDepGraph_->GetInstIndx(inst)]);
    schedCycle = SchdulInst_(inst, 0, trgtLastCycle);
    rltvCP = dataDepGraph_->GetRltvCrtclPath(leaf, inst, trvrslDir);
    delay = CmputDelay_(schedCycle, trgtLastCycle, rltvCP);
    /*
        #ifdef IS_DEBUG
          if(schedDir_==DIR_FRWRD &&leaf->GetNum()==6) {
            Logger::Info("inst %d: schedCycle=%d, rltvCP=%d, delay=%d",
                         inst->GetNum(), schedCycle, rltvCP, delay);
          }
        #endif
    */
    if (delay > maxDelay) {
      maxDelay = delay;
    }
  }

  return (trgtLastCycle + maxDelay);
  // Return the tightened lower bound on the subgraph's leaf
}
/*****************************************************************************/

InstCount LC_RelaxedScheduler::CmputReleaseTime_(SchedInstruction *inst) {
  InstCount newBound = PropagateLwrBound_(inst, schedDir_);
  SetCrntLwrBound_(inst, schedDir_, newBound);
  return newBound;
}
/*****************************************************************************/

LPP_RelaxedScheduler::LPP_RelaxedScheduler(DataDepStruct *dataDepGraph,
                                           MachineModel *machMdl,
                                           InstCount schedUprBound,
                                           DIRECTION mainDir)
    : RelaxedScheduler(dataDepGraph, machMdl, schedUprBound, mainDir, RST_STTC,
                       INVALID_VALUE) {

  // TEMP: The new implementation for LPP has not been completed yet
  assert(false);

  // TEMP: Support for dynamic scheduling has not been implemented yet
  assert(schedType_ == RST_STTC);

  schedDir_ = DirAcycGraph::ReverseDirection(mainDir_);

  subGraphInstLst_ = new PriorityList<SchedInstruction>;

  crntFrwrdLwrBounds_ = new InstCount[totInstCnt_];
  crntBkwrdLwrBounds_ = new InstCount[totInstCnt_];

  tightndFrwrdLwrBounds_ = new LinkedList<SchedInstruction>(totInstCnt_);
  tightndBkwrdLwrBounds_ = new LinkedList<SchedInstruction>(totInstCnt_);

  lstEntries_ = new KeyedEntry<SchedInstruction> *[totInstCnt_];
}
/*****************************************************************************/

LPP_RelaxedScheduler::~LPP_RelaxedScheduler() {
  delete subGraphInstLst_;
  delete[] crntFrwrdLwrBounds_;
  delete[] crntBkwrdLwrBounds_;
  delete tightndFrwrdLwrBounds_;
  delete tightndBkwrdLwrBounds_;
  delete[] lstEntries_;
}
/*****************************************************************************/

void LPP_RelaxedScheduler::Initialize_(bool opstFsbl) {
  InstCount i;
  DIRECTION bkwrdDir = DirAcycGraph::ReverseDirection(schedDir_);

  //  dataDepGraph_->SetInstIndexes();
  instLst_->Reset();

  for (i = 0; i < totInstCnt_; i++) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(i);

    if (!opstFsbl) {
      inst->RestoreAbsoluteBounds();
    }

    crntFrwrdLwrBounds_[i] = inst->GetLwrBound(schedDir_);
    crntBkwrdLwrBounds_[i] = inst->GetLwrBound(bkwrdDir);
    tightndFrwrdLwrBounds_->Reset();
    tightndBkwrdLwrBounds_->Reset();

    unsigned long prirty = CmputTplgclPrirty_(inst, schedDir_);
    instLst_->InsrtElmnt(inst, prirty, true);

#ifdef IS_DEBUG
    wasLwrBoundCmputd_[i] = false;
#endif
  }
}
/*****************************************************************************/

bool LPP_RelaxedScheduler::InitLength_(InstCount lngth) {
  int16_t t;

  for (t = 0; t < issuTypeCnt_; t++) {
    neededSlots_[t] = instCntPerIssuType_[t];
  }

  if (CheckSlots_(0, lngth - 1) == false) {
    return false;
  }

  RestoreLwrBounds_(DirAcycGraph::ReverseDirection(schedDir_));
  instLst_->ResetIterator();
  subGraphInstLst_->Reset();

  for (t = 0; t < issuTypeCnt_; t++) {
    neededSlots_[t] = 0;
  }

#ifdef IS_DEBUG

  for (InstCount i = 0; i < totInstCnt_; i++) {
    wasLwrBoundCmputd_[i] = false;
  }

#endif
  return true;
}
/*****************************************************************************/

bool LPP_RelaxedScheduler::InitSubGraph_(SchedInstruction *newInst,
                                         InstCount lastCycle) {
  InstCount releaseTime = GetCrntLwrBound_(newInst, schedDir_);

  if (CheckSlots_(releaseTime, lastCycle) == false) {
    return false;
  }

  InstCount tightBound = PropagateLwrBound_(newInst, mainDir_);
  assert(tightBound >= newInst->GetLwrBound(mainDir_));
  SetCrntLwrBound_(newInst, mainDir_, tightBound);
  return true;
}
/*****************************************************************************/

void LPP_RelaxedScheduler::InitCycleProbe_(SchedInstruction *newInst) {
  InstCount releaseTime = newInst->GetLwrBound(schedDir_);
  Reset_(releaseTime);
  RestoreLwrBounds_(schedDir_);
}
/*****************************************************************************/

bool LPP_RelaxedScheduler::CheckSlots_(InstCount firstCycle,
                                       InstCount lastCycle) {
  int16_t i;
  InstCount avlblSlots[MAX_ISSUTYPE_CNT];
  InstCount cycleCnt = lastCycle - firstCycle + 1;

  for (i = 0; i < issuTypeCnt_; i++) {
    avlblSlots[i] = slotsPerTypePerCycle_[i] * cycleCnt;

    if (avlblSlots[i] < neededSlots_[i]) {
      return false;
    }
  }

  return true;
}
/*****************************************************************************/

InstCount LPP_RelaxedScheduler::FindSchedule() {
  DIRECTION opstDir = DirAcycGraph::ReverseDirection(mainDir_);
  InstCount schedLngth;
  InstCount schedUprBound = schedUprBound_;
  InstCount flb = leafInst_->GetLwrBound(mainDir_);
  InstCount blb = rootInst_->GetLwrBound(opstDir);
  InstCount initSchedLwrBound = std::max(flb, blb) + 1;
  InstCount schedLwrBound = initSchedLwrBound;

  Initialize_(false);

  for (schedLngth = initSchedLwrBound; schedLngth < schedUprBound;
       schedLngth++) {
    if (ProbeLength_(schedLngth)) {
      break;
    } else {
      schedLwrBound = schedLngth + 1;
    }
  }

  return schedLwrBound;
}
/*****************************************************************************/

bool LPP_RelaxedScheduler::FindSchedule(InstCount trgtLngth, bool opstFsbl) {
  Initialize_(opstFsbl);
  return ProbeLength_(trgtLngth);
}
/*****************************************************************************/

bool LPP_RelaxedScheduler::ProbeLength_(InstCount trgtLngth) {
  SchedInstruction *inst = 0;
  InstCount i;
  InstCount newLwrBound = 0;
  InstCount lastCycle = trgtLngth - 1;
  InstCount instNum;

  bool fsbl = InitLength_(trgtLngth);
  if (!fsbl)
    return false;

  for (i = 0; i < totInstCnt_; i++) {
    inst = instLst_->GetNxtPriorityElmnt();
    instNum = inst->GetNum();
    assert(inst != NULL);
    assert(inst == rootInst_ || i > 0);
    newLwrBound = SchdulSubGraph_(inst, lastCycle);

    if (newLwrBound == INVALID_VALUE)
      return false;

    assert(newLwrBound >= inst->GetLwrBound(mainDir_));
#ifdef IS_DEBUG
    wasLwrBoundCmputd_[instNum] = true;
#endif
    if (newLwrBound > GetCrntLwrBound_(inst, mainDir_)) {
      assert(lstEntries_[instNum] != NULL);
      subGraphInstLst_->BoostEntry(lstEntries_[instNum], newLwrBound);
    }

    SetCrntLwrBound_(inst, mainDir_, newLwrBound);
  }

  assert(inst == leafInst_);
  assert(newLwrBound + 1 <= trgtLngth);

  // If the length is feasible, set the permanent LBs to the current LBs
  for (i = 0; i < totInstCnt_; i++) {
    inst = dataDepGraph_->GetInstByIndx(i);
    inst->SetLwrBound(mainDir_, crntBkwrdLwrBounds_[i], false);
  }

  return true;
}
/*****************************************************************************/

InstCount LPP_RelaxedScheduler::SchdulSubGraph_(SchedInstruction *newInst,
                                                InstCount trgtLastCycle) {
  InstCount trgtCycle;

  if (InitSubGraph_(newInst, trgtLastCycle) == false) {
    return INVALID_VALUE;
  }

  InstCount FLB = GetCrntLwrBound_(newInst, schedDir_);
  InstCount BLB =
      GetCrntLwrBound_(newInst, DirAcycGraph::ReverseDirection(schedDir_));
  InstCount releaseTime = FLB;
  InstCount deadline = trgtLastCycle - BLB;

  KeyedEntry<SchedInstruction> *entry;
  entry = subGraphInstLst_->InsrtElmnt(newInst, BLB, true);
  neededSlots_[newInst->GetIssueType()]++;
  lstEntries_[newInst->GetNum()] = entry;

  if (newInst == rootInst_) {
    return 0;
  }

  for (trgtCycle = deadline; trgtCycle >= releaseTime; trgtCycle--) {
    bool fsbl = ProbeCycle_(newInst, trgtLastCycle, trgtCycle);

    if (fsbl) {
      InstCount newBLB = trgtLastCycle - trgtCycle;
      return newBLB;
    }
  }

  return INVALID_VALUE;
}
/*****************************************************************************/

bool LPP_RelaxedScheduler::ProbeCycle_(SchedInstruction *newInst,
                                       InstCount trgtLastCycle,
                                       InstCount trgtCycle) {
  SchedInstruction *inst;
  DIRECTION bkwrdDir = DirAcycGraph::ReverseDirection(schedDir_);

  assert(trgtCycle >= newInst->GetLwrBound(schedDir_));
  assert(trgtCycle <=
         trgtLastCycle -
             newInst->GetLwrBound(DirAcycGraph::ReverseDirection(schedDir_)));

  InitCycleProbe_(newInst);
  SetCrntLwrBound_(newInst, schedDir_, trgtCycle);

  for (inst = subGraphInstLst_->GetFrstElmnt(); inst != NULL;
       inst = subGraphInstLst_->GetNxtElmnt()) {
    assert(inst == newInst || wasLwrBoundCmputd_[inst->GetNum()]);
    InstCount schedCycle = SchdulInst_(inst, 0, trgtLastCycle);
    InstCount BLB = GetCrntLwrBound_(inst, bkwrdDir);
    InstCount delay = CmputDelay_(schedCycle, trgtLastCycle, BLB);

    if (delay > 0)
      return false;
  }

  return true;
}
/*****************************************************************************/

InstCount LPP_RelaxedScheduler::CmputReleaseTime_(SchedInstruction *inst) {
  InstCount tightLwrBound = PropagateLwrBound_(inst, schedDir_);
  InstCount instNum = inst->GetNum();

  if (tightLwrBound > crntFrwrdLwrBounds_[instNum]) {
    SetCrntLwrBound_(inst, schedDir_, tightLwrBound);
  }

  return tightLwrBound;
}
/*****************************************************************************/

InstCount LPP_RelaxedScheduler::GetCrntLwrBound_(SchedInstruction *inst,
                                                 DIRECTION dir) {
  InstCount instNum = inst->GetNum();

  assert(inst != NULL);

  if (dir == schedDir_) {
    return crntFrwrdLwrBounds_[instNum];
  } else {
    return crntBkwrdLwrBounds_[instNum];
  }
}
/*****************************************************************************/

void LPP_RelaxedScheduler::SetCrntLwrBound_(SchedInstruction *inst,
                                            DIRECTION dir, InstCount newBound) {
  assert(inst != NULL);
  InstCount instNum = inst->GetNum();

  if (dir == schedDir_) {
    if (newBound > GetCrntLwrBound_(inst, dir)) {
      if (crntFrwrdLwrBounds_[instNum] == inst->GetLwrBound(dir)) {
        tightndFrwrdLwrBounds_->InsrtElmnt(inst);
      }

      crntFrwrdLwrBounds_[instNum] = newBound;
    }
  } else {
    if (newBound > GetCrntLwrBound_(inst, dir)) {
      /*
      Logger::Info("Tightening BLB of inst. %d from %d to %d. "
                   "Tight. list size =%d",
                   inst->GetNum(),
                   GetCrntLwrBound_(inst, dir),
                   newBound,
                   tightndBkwrdLwrBounds_->GetElmntCnt());
      */
      if (crntBkwrdLwrBounds_[instNum] == inst->GetLwrBound(dir)) {
        tightndBkwrdLwrBounds_->InsrtElmnt(inst);
      }

      crntBkwrdLwrBounds_[instNum] = newBound;
    }
  }
}
/*****************************************************************************/

void LPP_RelaxedScheduler::RestoreLwrBounds_(DIRECTION dir) {
  InstCount instNum;
  SchedInstruction *inst;

  if (dir == schedDir_) {
    for (inst = tightndFrwrdLwrBounds_->GetFrstElmnt(); inst != NULL;
         inst = tightndFrwrdLwrBounds_->GetNxtElmnt()) {
      instNum = inst->GetNum();
      assert(crntFrwrdLwrBounds_[instNum] > inst->GetLwrBound(dir));
      crntFrwrdLwrBounds_[instNum] = inst->GetLwrBound(dir);
    }

    tightndFrwrdLwrBounds_->Reset();
  } else {
    for (inst = tightndBkwrdLwrBounds_->GetFrstElmnt(); inst != NULL;
         inst = tightndBkwrdLwrBounds_->GetNxtElmnt()) {
      instNum = inst->GetNum();
      assert(crntBkwrdLwrBounds_[instNum] > inst->GetLwrBound(dir));
      crntBkwrdLwrBounds_[instNum] = inst->GetLwrBound(dir);
    }

    tightndBkwrdLwrBounds_->Reset();
  }
}
/*****************************************************************************/
