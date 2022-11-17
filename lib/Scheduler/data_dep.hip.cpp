#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string.h>

#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/relaxed_sched.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/dev_defines.h"
#include "opt-sched/Scheduler/aco.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <hip/hip_runtime.h>

// only print pressure if enabled by sched.ini
extern bool OPTSCHED_gPrintSpills;

using namespace llvm::opt_sched;

// The maximum number of blocks allowed in a trace.
const int MAX_TRACE_BLOCKS = 100;

static const char *GetDependenceTypeName(DependenceType depType) {
  switch (depType) {
  case DEP_DATA:
    return "data";
  case DEP_ANTI:
    return "anti";
  case DEP_OUTPUT:
    return "output";
  case DEP_OTHER:
    return "other";
  }
  llvm_unreachable("Unknown dependence type!");
}

__host__ __device__
DataDepStruct::DataDepStruct(MachineModel *machMdl) {
  machMdl_ = machMdl;
  issuTypeCnt_ = (int16_t)machMdl->GetIssueTypeCnt();
  instCntPerIssuType_ = new InstCount[issuTypeCnt_];

  for (int16_t i = 0; i < issuTypeCnt_; i++) {
    instCntPerIssuType_[i] = 0;
  }

  schedLwrBound_ = INVALID_VALUE;
  schedUprBound_ = INVALID_VALUE;
  insts_ = NULL;
  instCnt_ = 0;
  frwrdLwrBounds_ = NULL;
  bkwrdLwrBounds_ = NULL;
  includesUnpipelined_ = false;
  edges_ = NULL;
}

__host__
DataDepStruct::~DataDepStruct() {
  delete[] instCntPerIssuType_;
  if (frwrdLwrBounds_ != NULL)
    delete[] frwrdLwrBounds_;
  if (bkwrdLwrBounds_ != NULL)
    delete[] bkwrdLwrBounds_;
  #ifndef __HIP_DEVICE_COMPILE__
    if (edges_)
      delete edges_;
  #endif
}

__host__ __device__
void DataDepStruct::GetInstCntPerIssuType(InstCount instCntPerIssuType[]) {
  for (int16_t i = 0; i < issuTypeCnt_; i++) {
    instCntPerIssuType[i] = instCntPerIssuType_[i];
  }
}

// Moved over from utilities.h.

// Removes the first and last characters of a string. In debug builds, makes
// sure these characters are double quotes. Assumes dest is at least srcLen-1
// in length.
static inline void rmvDblQuotes(const char *src, int srcLen, char *dest) {
  assert(src[0] == '"');
  assert(src[srcLen - 1] == '"');
  strncpy(dest, src + 1, srcLen - 2);
  dest[srcLen - 2] = '\0';
}

// Adds double quotes at the beginning and end of a string. Assumes dest is at
// least srcLen+3 in length.
static inline void addDblQuotes(const char *src, int srcLen, char *dest) {
  dest[0] = '"';
  strncpy(dest + 1, src, srcLen);
  dest[srcLen + 1] = '"';
  dest[srcLen + 2] = '\0';
}

__host__
InstCount DataDepStruct::CmputRsrcLwrBound_() {
  // Temp limitation
  assert(type_ == DGT_FULL);
  assert(type_ == DGT_SUB || GetInstCnt() > 0);

  if (GetInstCnt() == 0)
    return 0;

  int *slotsPerIssuType;
  slotsPerIssuType = new int[issuTypeCnt_];

  machMdl_->GetSlotsPerCycle(slotsPerIssuType);

  for (InstCount i = 0; i < issuTypeCnt_; i++) {
    instCntPerIssuType_[i] = 0;
  }

  for (InstCount i = 0; i < GetInstCnt(); i++) {
    SchedInstruction *inst = ((DataDepGraph*)(this))->GetInstByIndx(i);
    IssueType issuType = inst->GetIssueType();
    assert(issuType <= issuTypeCnt_);
    instCntPerIssuType_[issuType]++;
  }

  InstCount rsrcLwrBound = 0;

  for (InstCount i = 0; i < issuTypeCnt_; i++) {
    InstCount typeLwrBound =
        (instCntPerIssuType_[i] + slotsPerIssuType[i] - 1) /
        slotsPerIssuType[i];

    if (typeLwrBound > rsrcLwrBound)
      rsrcLwrBound = typeLwrBound;
  }

  assert(rsrcLwrBound != INVALID_VALUE);
  assert(rsrcLwrBound >= 1);
  delete[] slotsPerIssuType;
  return rsrcLwrBound;
}

__host__
InstCount DataDepStruct::CmputAbslutUprBound_() {
  InstCount i;
  InstCount ltncySum = 0;

  for (i = 0; i < instCnt_; i++) {
    UDT_GLABEL maxLtncy = insts_[i].GetMaxEdgeLabel();

    if (1 > maxLtncy)
      maxLtncy = 1;

    ltncySum += maxLtncy;
  }

  schedUprBound_ = ltncySum + 1;

  return schedUprBound_;
}

__host__
DataDepGraph::DataDepGraph(MachineModel *machMdl, LATENCY_PRECISION ltncyPrcsn)
    : DataDepStruct(machMdl) {
  int i;

  type_ = DGT_FULL;
  machMdl_ = machMdl;
  useFileLtncs_ = true;
  weight_ = 1.0;
  outptDags_ = ODG_ALL;
  maxOutptDagSize_ = 1000;
  ltncyPrcsn_ = ltncyPrcsn;
  includesCall_ = false;
  includesUnpipelined_ = false;

  bscBlkCnt_ = 0;
  maxLtncy_ = 0;
  maxLtncySum_ = 0;
  backTrackEnbl_ = false;
  realInstCnt_ = 0;
  isHard_ = false;

  fileSchedLwrBound_ = INVALID_VALUE;
  fileSchedUprBound_ = INVALID_VALUE;
  finalLwrBound_ = INVALID_VALUE;
  finalUprBound_ = INVALID_VALUE;
  fileSchedTrgtUprBound_ = INVALID_VALUE;
  fileCostUprBound_ = INVALID_VALUE;
  fileSchedLngth_ = INVALID_VALUE;
  minFileSchedCycle_ = 0;
  maxFileSchedOrder_ = 0;
  maxUseCnt_ = 0;

  dagFileFormat_ = DFF_BB;
  wasSetupForSchduling_ = false;

  char src[10] = "unknown";
  i = 0;
  do {
    dagID_[i] = src[i];}
  while (src[i++] != 0);

  instTypeCnt_ = (int16_t)machMdl->GetInstTypeCnt();
  instCntPerType_ = new InstCount[instTypeCnt_];

  for (i = 0; i < instTypeCnt_; i++) {
    instCntPerType_[i] = 0;
  }

  for (i = 0; i <= MAX_LATENCY_VALUE; i++) {
    edgeCntPerLtncy_[i] = 0;
  }

  lastBlkNum_ = 0;
  isPrblmtc_ = false;

  entryInstCnt_ = 0;
  exitInstCnt_ = 0;
  maxIndependentInstructions_ = 0;

#ifdef __HIP_DEVICE_COMPILE__
  graphTrans_ = NULL;
#else
  graphTrans_ = new SmallVector<std::unique_ptr<GraphTrans>, 0>;
#endif

  RegFiles = new RegisterFile[machMdl_->GetRegTypeCnt()];
}

__host__
DataDepGraph::~DataDepGraph() {
  if (insts_ != NULL) {
    delete[] insts_;
  }
  delete[] instCntPerType_;
}

__host__
FUNC_RESULT DataDepGraph::SetupForSchdulng(bool cmputTrnstvClsr) {
  assert(wasSetupForSchduling_ == false);

  InstCount i;

  maxUseCnt_ = 0;

  for (i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = &insts_[i];
    inst->SetupForSchdulng(instCnt_, cmputTrnstvClsr, cmputTrnstvClsr);
    InstType instType = inst->GetInstType();
    IssueType issuType = machMdl_->GetIssueType(instType);
    assert(issuType < issuTypeCnt_);
    inst->SetIssueType(issuType);
    instCntPerIssuType_[issuType]++;

    inst->SetMustBeInBBEntry(false);
    inst->SetMustBeInBBExit(false);

    if (inst->GetUseCnt() > maxUseCnt_)
      maxUseCnt_ = inst->GetUseCnt();
  }

  //  Logger::Info("Max use count = %d", maxUseCnt_);

  // Do a depth-first search leading to a topological sort
  if (!dpthFrstSrchDone_) {
    DepthFirstSearch();
  }

  frwrdLwrBounds_ = new InstCount[instCnt_];
  bkwrdLwrBounds_ = new InstCount[instCnt_];

  CmputCrtclPaths_();

  if (cmputTrnstvClsr) {
    if (FindRcrsvNghbrs(DIR_FRWRD) == RES_ERROR)
      return RES_ERROR;
    if (FindRcrsvNghbrs(DIR_BKWRD) == RES_ERROR)
      return RES_ERROR;
    CmputRltvCrtclPaths_(DIR_FRWRD);
    CmputRltvCrtclPaths_(DIR_BKWRD);
  }

  CmputAbslutUprBound_();
  CmputBasicLwrBounds_();
  wasSetupForSchduling_ = true;
  return RES_SUCCESS;
}

// __device__
// FUNC_RESULT DataDepGraph::Dev_SetupForSchdulng(bool cmputTrnstvClsr) {
//   assert(wasSetupForSchduling_ == false);

//   InstCount i = hipBlockIdx_x;

//   maxUseCnt_ = 0;
  
//   if (i < instCnt_) {
//     SchedInstruction *inst = &insts_[i];
//     inst->SetupForSchdulng(instCnt_, cmputTrnstvClsr, cmputTrnstvClsr);
//     InstType instType = inst->GetInstType();
//     IssueType issuType = machMdl_->GetIssueType(instType);
//     assert(issuType < issuTypeCnt_);
//     inst->SetIssueType(issuType);
//     instCntPerIssuType_[issuType]++;

//     inst->SetMustBeInBBEntry(false);
//     inst->SetMustBeInBBExit(false);
  
//     if (inst->GetUseCnt() > maxUseCnt_)
//       maxUseCnt_ = inst->GetUseCnt();
//   }

//   // Do a depth-first search leading to a topological sort
//   if (i == 0) {
//     if (!dpthFrstSrchDone_) {
//       DepthFirstSearch();
//     }

//     frwrdLwrBounds_ = new InstCount[instCnt_];
//     bkwrdLwrBounds_ = new InstCount[instCnt_];

//     CmputCrtclPaths_();

//     if (cmputTrnstvClsr) {
//       if (FindRcrsvNghbrs(DIR_FRWRD) == RES_ERROR)
//         return RES_ERROR;
//       if (FindRcrsvNghbrs(DIR_BKWRD) == RES_ERROR)
//         return RES_ERROR;
//       CmputRltvCrtclPaths_(DIR_FRWRD);
//       CmputRltvCrtclPaths_(DIR_BKWRD);
//     }

//     CmputAbslutUprBound_();
//     CmputBasicLwrBounds_();
//     wasSetupForSchduling_ = true;
//   }
//   return RES_SUCCESS;
// }

FUNC_RESULT DataDepGraph::UpdateSetupForSchdulng(bool cmputTrnstvClsr) {
  InstCount i;
  for (i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = &insts_[i];
    inst->SetupForSchdulng(instCnt_, cmputTrnstvClsr, cmputTrnstvClsr);
    InstType instType = inst->GetInstType();
    IssueType issuType = machMdl_->GetIssueType(instType);
    assert(issuType < issuTypeCnt_);
    inst->SetIssueType(issuType);

    inst->SetMustBeInBBEntry(false);
    inst->SetMustBeInBBExit(false);
  }

  // Do a depth-first search leading to a topological sort
  DepthFirstSearch();

  delete[] frwrdLwrBounds_;
  delete[] bkwrdLwrBounds_;

  frwrdLwrBounds_ = new InstCount[instCnt_];
  bkwrdLwrBounds_ = new InstCount[instCnt_];

  CmputCrtclPaths_();

  if (cmputTrnstvClsr) {
    if (FindRcrsvNghbrs(DIR_FRWRD) == RES_ERROR)
      return RES_ERROR;
    if (FindRcrsvNghbrs(DIR_BKWRD) == RES_ERROR)
      return RES_ERROR;
    CmputRltvCrtclPaths_(DIR_FRWRD);
    CmputRltvCrtclPaths_(DIR_BKWRD);
  }

  CmputAbslutUprBound_();
  CmputBasicLwrBounds_();

  return RES_SUCCESS;
}

__host__
void DataDepGraph::CmputBasicLwrBounds_() {
  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = GetInstByIndx(i);
    InstCount frwrdLwrBound = inst->GetCrtclPath(DIR_FRWRD);
    InstCount bkwrdLwrBound = inst->GetCrtclPath(DIR_BKWRD);
    inst->SetBounds(frwrdLwrBound, bkwrdLwrBound);
    frwrdLwrBounds_[i] = frwrdLwrBound;
    bkwrdLwrBounds_[i] = bkwrdLwrBound;
  }

  schedLwrBound_ = GetLeafInst()->GetLwrBound(DIR_FRWRD) + 1;
  InstCount rsrcLwrBound = CmputRsrcLwrBound_();
  //schedLwrBound_ = std::max(schedLwrBound_, rsrcLwrBound);
  if (schedLwrBound_ < rsrcLwrBound)
    schedLwrBound_ = rsrcLwrBound;
}

void DataDepGraph::SetSttcLwrBounds() {
  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = GetInstByIndx(i);
    InstCount frwrdLwrBound = inst->GetLwrBound(DIR_FRWRD);
    InstCount bkwrdLwrBound = inst->GetLwrBound(DIR_BKWRD);
    assert(inst->GetCrntLwrBound(DIR_FRWRD) >= frwrdLwrBound);
    assert(inst->GetCrntLwrBound(DIR_BKWRD) >= bkwrdLwrBound);
    frwrdLwrBounds_[i] = frwrdLwrBound;
    bkwrdLwrBounds_[i] = bkwrdLwrBound;
  }
}

void DataDepGraph::SetDynmcLwrBounds() {
  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = GetInstByIndx(i);
    InstCount frwrdLwrBound = inst->GetCrntLwrBound(DIR_FRWRD);
    InstCount bkwrdLwrBound = inst->GetCrntLwrBound(DIR_BKWRD);
    frwrdLwrBounds_[i] = frwrdLwrBound;
    bkwrdLwrBounds_[i] = bkwrdLwrBound;
  }
}

FUNC_RESULT DataDepGraph::ReadFrmFile(SpecsBuffer *buf,
                                      bool &endOfFileReached) {
  int pieceCnt;
  char *strngs[INBUF_MAX_PIECES_PERLINE];
  int lngths[INBUF_MAX_PIECES_PERLINE];
  InstCount fileTgtUprBound;

  if (endOfFileReached) {
    return RES_END;
  }

  NXTLINE_TYPE nxtLine = buf->GetNxtVldLine(pieceCnt, strngs, lngths);

  if (nxtLine == NXT_EOF || pieceCnt == 0) {
    endOfFileReached = true;
    return RES_END;
  }

  if (strncmp(strngs[0], "dag", 3) != 0) {
    Logger::Error("Invalid token %s in DAG file. Expected dag.", strngs[0]);
    return RES_ERROR;
  }

  dagFileFormat_ = DFF_BB;
  isTraceFormat_ = false;

  if (strncmp(strngs[0] + 4, "superblock", 10) == 0) {
    dagFileFormat_ = DFF_SB;
    isTraceFormat_ = true;
  }

  if (strncmp(strngs[0] + 4, "trace", 5) == 0) {
    dagFileFormat_ = DFF_TR;
    isTraceFormat_ = true;
  }

  instCnt_ = nodeCnt_ = atoi(strngs[1]);

  if (buf->checkTitle("{") == RES_ERROR) {
    return RES_ERROR;
  }

  buf->ReadSpec("dag_id", dagID_);

  weight_ = buf->ReadFloatSpec("dag_weight");

  buf->ReadSpec("compiler", compiler_);

  AllocArrays_(instCnt_);

  buf->GetNxtVldLine(pieceCnt, strngs, lngths);

  if (strcmp(strngs[0], "dag_lb") == 0) {
    fileSchedLwrBound_ = atoi(strngs[1]);

    buf->GetNxtVldLine(pieceCnt, strngs, lngths);
    fileSchedUprBound_ = atoi(strngs[1]);

    buf->GetNxtVldLine(pieceCnt, strngs, lngths);

    if (strcmp(strngs[0], "dag_tgt_ub") == 0) {
      fileTgtUprBound = atoi(strngs[1]);
      fileSchedTrgtUprBound_ = fileTgtUprBound;
      buf->GetNxtVldLine(pieceCnt, strngs, lngths); // skip the nodes line
    }

    if (strcmp(strngs[0], "dag_cost_ub") == 0) {
      fileCostUprBound_ = atoi(strngs[1]);
      fileCostUprBound_ /= 10; // convert denominator from 1000 to 100
      buf->GetNxtVldLine(pieceCnt, strngs, lngths); // skip the nodes line
    }
  } else {
    if (strcmp(strngs[0], "nodes") != 0 && strcmp(strngs[0], "blocks") != 0) {
      Logger::Error("Invalid token %s in DAG file. Expected nodes or blocks.",
                    strngs[0]);
      return RES_ERROR;
    }
  }

  FUNC_RESULT rslt;

  if (dagFileFormat_ == DFF_TR) {
    bscBlkCnt_ = atoi(strngs[1]);

    if (bscBlkCnt_ > MAX_TRACE_BLOCKS) {
      Logger::Error("Too many blocks in a trace. Limit is %d.",
                    MAX_TRACE_BLOCKS);
      return RES_ERROR;
    }

    rslt = ParseF2Blocks_(buf);

    if (rslt == RES_ERROR) {
      return rslt;
    }

    buf->GetNxtVldLine(pieceCnt, strngs, lngths); // skip the nodes line
  }

  rslt = ParseF2Nodes_(buf, machMdl_);

  if (rslt == RES_END) {
    endOfFileReached = true;
  }

  if (rslt == RES_ERROR) {
    return rslt;
  }

  rslt = ParseF2Edges_(buf, machMdl_);

  if (rslt == RES_END) {
    endOfFileReached = true;
  }

  if (rslt == RES_ERROR) {
    return rslt;
  }

  rslt = Finish_();
  return rslt;
}

FUNC_RESULT DataDepGraph::Finish_() {
  root_ = NULL;

  for (InstCount i = 0; i < instCnt_; i++) {
    if (FinishNode_(i) == RES_ERROR) {
      return RES_ERROR;
    }

    if (insts_[i].GetPrdcsrCnt() == 0) {
      if (root_ != NULL) {
        Logger::Error("Invalid format in DAG (%s): multiple root nodes.",
                      dagID_);
        return RES_ERROR;
      }

      root_ = &insts_[i];
    }
  }

  if (root_ == NULL) {
    Logger::Error("Invalid format in DAG (%s): missing root node", dagID_);
    return RES_ERROR;
  }

  return RES_SUCCESS;
}

void DataDepGraph::AllocArrays_(InstCount instCnt) { 
  instCnt_ = instCnt;
  nodeCnt_ = instCnt;
  insts_ = new SchedInstruction[instCnt_];
  // bit of a hack. Cannot simply do nodes_ = (GN*)insts_ anymore
  // since their sizes do not match so nodes indexes incorrectly
  // instead I am keeping nodes as pointers to insts for now
  // since this would be simple to recreate on device
  nodes_ = new GraphNode *[instCnt_];

  for (InstCount i = 0; i < instCnt_; i++) {
    nodes_[i] = (GraphNode *)&insts_[i];
  }
}

FUNC_RESULT DataDepGraph::ParseF2Blocks_(SpecsBuffer *buf) {
  // TODO(max): Get rid of this. It's reading and discarding irrelevant data.
  for (InstCount i = 0; i < bscBlkCnt_; i++) {
    if (buf->ReadIntSpec("block") != i) {
      Logger::Error("Invalid block number in DAG file. Expected %d.", i);
      return RES_ERROR;
    }
    buf->ReadIntSpec("index");
    buf->ReadIntSpec("frequency");
    buf->ReadIntSpec("off_trace_pred_freq");
  }

  return RES_SUCCESS;
}

FUNC_RESULT DataDepGraph::ParseF2Nodes_(SpecsBuffer *buf,
                                        MachineModel *machMdl) {
  NXTLINE_TYPE nxtLine;
  InstCount i;
  InstCount nodeNum;
  FUNC_RESULT rslt = RES_SUCCESS;
  InstCount maxFileSchedCycle = 0;
  char instName[MAX_INSTNAME_LNGTH] = " ";
  char prevInstName[MAX_INSTNAME_LNGTH] = " ";
  char opCode[MAX_INSTNAME_LNGTH] = " ";
  InstType instType;
  int blkNumForLastBranch = INVALID_VALUE;

  includesCall_ = false;
  includesUnpipelined_ = false;
  includesUnsupported_ = false;
  includesNonStandardBlock_ = false;

  for (i = 0; i < instCnt_; i++) {

    rslt = ReadInstName_(buf, i, instName, prevInstName, opCode, nodeNum,
                         instType, nxtLine);

    if (rslt != RES_SUCCESS)
      break;

    if (isTraceFormat_ && machMdl->IsBranch(instType)) {
      // TODO(max): Remove this. It's reading and discarding irrelevant data.
      buf->ReadIntSpec("On-trace_Prob");
    }

    InstCount fileInstLwrBound = 0;
    InstCount fileInstUprBound = 0;
    InstCount fileSchedOrder = 0;
    InstCount fileSchedCycle = 0;

    int nodeID = 0;

    if (machMdl_->IsArtificial(instType)) {
      if (i == 0) { // root
        fileSchedOrder = 0;
        fileSchedCycle = 0;
      }

      if (i == instCnt_ - 1) {
        fileSchedOrder = instCnt_ - 1;
        fileSchedCycle = maxFileSchedCycle + 1;
      }
    } else {
      fileSchedOrder = buf->ReadIntSpec("sched_order");
      fileSchedCycle = buf->ReadIntSpec("issue_cycle");
      maxFileSchedCycle = std::max(maxFileSchedCycle, fileSchedCycle);
    }

    if (machMdl_->IsRealInst(instType)) {
      realInstCnt_++;
    }

    int blkNum = lastBlkNum_;

    if (dagFileFormat_ == DFF_TR) {
      if (!machMdl_->IsArtificial(instType)) {
        blkNum = buf->ReadIntSpec("block_num");
      }

      if (blkNum >= bscBlkCnt_ || blkNum < lastBlkNum_) {
        Logger::Error("Invalid block number %d in DAG file. "
                      "Expected a value between %d and %d",
                      blkNum, lastBlkNum_, bscBlkCnt_ - 1);
        rslt = nxtLine == NXT_EOF ? RES_END : RES_ERROR;
        break;
      }

      lastBlkNum_ = blkNum;

      if (machMdl_->IsBranch(instType)) {
        if (blkNum == blkNumForLastBranch) {
          includesNonStandardBlock_ = true;
          Logger::Info("Non-standard basic block: "
                       "Multiple branches in block %d",
                       blkNum);
        }

        blkNumForLastBranch = blkNum;
      }
    }

    CreateNode_(nodeNum, instName, instType, opCode, nodeID, fileSchedOrder,
                fileSchedCycle, fileInstLwrBound, fileInstUprBound, blkNum);

    instCntPerType_[instType]++;
    stats::instructionTypeCounts.Increment(
        machMdl->GetInstTypeNameByCode(instType));
  }

  if (rslt == RES_SUCCESS) {
    AdjstFileSchedCycles_();
  }

  if (nxtLine == NXT_EOF) {
    rslt = RES_END;
  }

  return rslt;
}

FUNC_RESULT DataDepGraph::ReadInstName_(SpecsBuffer *buf, int i, char *instName,
                                        char *prevInstName, char *opCode,
                                        InstCount &nodeNum, InstType &instType,
                                        NXTLINE_TYPE &nxtLine) {
  int pieceCnt;
  char *strngs[INBUF_MAX_PIECES_PERLINE];
  int lngths[INBUF_MAX_PIECES_PERLINE];
  FUNC_RESULT rslt;

  nxtLine = buf->GetNxtVldLine(pieceCnt, strngs, lngths);
  int expctdPieceCnt = 3;

  if (pieceCnt > 3 && i > 0 && i < (instCnt_ - 1)) {
    expctdPieceCnt = 4;
  }

  if (pieceCnt != expctdPieceCnt || strcmp(strngs[0], "node") != 0) {
    Logger::Error("In defining inst %d: Invalid number of tockens near %s.", i,
                  strngs[0]);
    rslt = nxtLine == NXT_EOF ? RES_END : RES_ERROR;
    return rslt;
  }

  nodeNum = atoi(strngs[1]);
  /*
  if (nodeNum != i) {
    Logger::Error("Invalid node number %d for inst %d.", nodeNum, i);
    rslt = (nxtLine == NXT_EOF) ? RES_END : RES_ERROR;
    break;
  }
  */

  rmvDblQuotes(strngs[2], lngths[2], instName);
  instType = machMdl_->GetInstTypeByName(instName, prevInstName);

  if (instType == INVALID_INST_TYPE) {
    Logger::Error("Invalid inst type %s for node #%d", instName, nodeNum);
    rslt = nxtLine == NXT_EOF ? RES_END : RES_ERROR;
    return rslt;
  }

  strcpy(prevInstName, instName);

  if (machMdl_->IsPipelined(instType) == false) {
    includesUnpipelined_ = true;
  }

  if (machMdl_->IsSupported(instType) == false) {
    includesUnsupported_ = true;
  }

  if (machMdl_->IsCall(instType)) {
    includesCall_ = true;
  }

  if (pieceCnt == 4 && i > 0 && i < (instCnt_ - 1)) {
    rmvDblQuotes(strngs[3], lngths[3], opCode);
  } else {
    strcpy(opCode, " ");
  }

  return RES_SUCCESS;
}

void DataDepGraph::AdjstFileSchedCycles_() {
  InstCount fileSchedCycle = 0;
  InstCount minFileSchedCycle = 0;
  InstCount maxFileSchedCycle = 0;
  SchedInstruction *inst;
  InstCount i;

  for (i = 1; i < (instCnt_ - 1); i++) {
    inst = &insts_[i];
    fileSchedCycle = inst->GetFileSchedCycle();

    if (i == 1) {
      minFileSchedCycle = fileSchedCycle;
    } else {
      minFileSchedCycle = std::min(minFileSchedCycle, fileSchedCycle);
    }
  }

  for (i = 1; i < (instCnt_ - 1); i++) {
    inst = &insts_[i];
    fileSchedCycle = inst->GetFileSchedCycle();
    fileSchedCycle -= minFileSchedCycle;
    maxFileSchedCycle = std::max(maxFileSchedCycle, fileSchedCycle);
  }

  minFileSchedCycle_ = minFileSchedCycle;
  fileSchedLngth_ = maxFileSchedCycle + 3;
}

InstCount DataDepGraph::GetAdjustedFileSchedCycle(InstCount instNum) {
  InstCount adjstdSchedCycle;

  if (instNum == 0) {
    adjstdSchedCycle = 0;
  } else if (instNum == instCnt_ - 1) {
    adjstdSchedCycle = fileSchedLngth_ - 1;
  } else {
    adjstdSchedCycle =
        insts_[instNum].GetFileSchedCycle() - minFileSchedCycle_ + 1;
  }

  return adjstdSchedCycle;
}

FUNC_RESULT DataDepGraph::ParseF2Edges_(SpecsBuffer *buf,
                                        MachineModel *machMdl) {
  int pieceCnt;
  char *strngs[INBUF_MAX_PIECES_PERLINE];
  int lngths[INBUF_MAX_PIECES_PERLINE];
  int ltncy;
  NXTLINE_TYPE nxtLine;
  FUNC_RESULT rslt = RES_SUCCESS;

  if (buf->checkTitle("dependencies") == RES_ERROR)
    return RES_ERROR;

  do {
    nxtLine = buf->GetNxtVldLine(pieceCnt, strngs, lngths);

    if (pieceCnt >= 4) {
      if (strncmp(strngs[0], "dep", 3) != 0) {
        Logger::Error("Invalid edge definition. Expected dependence.");
        rslt = (nxtLine == NXT_EOF) ? RES_END : RES_ERROR;
        break;
        // return RES_ERROR;
      }

      InstCount frmNodeNum = atoi(strngs[1]);
      InstCount toNodeNum = atoi(strngs[2]);
      char depTypeName[MAX_NAMESIZE];

      rmvDblQuotes(strngs[3], lngths[3], depTypeName);

      DependenceType depType;
      if (strcmp(depTypeName, "data") == 0) {
        depType = DEP_DATA;
      } else if (strcmp(depTypeName, "anti") == 0) {
        depType = DEP_ANTI;
      } else if (strcmp(depTypeName, "output") == 0) {
        depType = DEP_OUTPUT;
      } else {
        depType = DEP_OTHER;
      }

      if (useFileLtncs_ && pieceCnt == 5) {
        ltncy = atoi(strngs[4]);
      } else {
        InstType frmInstType = insts_[frmNodeNum].GetInstType();
        ltncy = machMdl->GetLatency(frmInstType, depType);
      }

      CreateEdge_(frmNodeNum, toNodeNum, ltncy, depType);
    } else {
      if (strcmp(strngs[0], "schedule") == 0) {
        do {
          nxtLine = buf->GetNxtVldLine(pieceCnt, strngs, lngths);
        } while (pieceCnt != 1);
      }

      if (pieceCnt != 1 || strcmp(strngs[0], "}") != 0) {
        Logger::Error("Invalid DAG def near %s. Expected \"}\".", strngs[0]);
        rslt = nxtLine == NXT_EOF ? RES_END : RES_ERROR;
        break;
        // return RES_ERROR;
      }
    }
  } while (pieceCnt >= 4);

  if (nxtLine == NXT_EOF)
    rslt = RES_END;

  return rslt;
  //  return nxtLine==NXT_EOF? RES_END: RES_SUCCESS;
}

FUNC_RESULT DataDepGraph::SkipGraph(SpecsBuffer *buf, bool &endOfFileReached) {
  if (endOfFileReached)
    return RES_END;

  while (true) {
    int pieceCnt;
    char *strngs[INBUF_MAX_PIECES_PERLINE];
    int lngths[INBUF_MAX_PIECES_PERLINE];
    NXTLINE_TYPE nxtLine = buf->GetNxtVldLine(pieceCnt, strngs, lngths);

    if (nxtLine == NXT_EOF) {
      endOfFileReached = true;
      return RES_END;
    } else if (pieceCnt == 1 && strngs[0][0] == '}') {
      return RES_SUCCESS;
    }
  }

  return RES_SUCCESS;
}

__host__ __device__
SchedInstruction *DataDepGraph::CreateNode_(
    InstCount instNum, const char *const instName, InstType instType,
    const char *const opCode, int nodeID, InstCount fileSchedOrder,
    InstCount fileSchedCycle, InstCount fileLB, InstCount fileUB, int blkNum) {
  insts_[instNum].InitializeNode_(instNum, instName, instType, opCode,
                                  2 * instCnt_, nodeID, fileSchedOrder,
                                  fileSchedCycle, fileLB, fileUB, machMdl_,
				  nodes_, insts_, RegFiles);

  if ((instNum < 0 || instNum >= instCnt_) && instNum != UNINITIATED_NUM)
    printf("Invalid instruction number\n");
  if (fileSchedOrder > maxFileSchedOrder_)
    maxFileSchedOrder_ = fileSchedOrder;

  return &insts_[instNum];
}

void DataDepGraph::CreateEdge(SchedInstruction *frmNode,
                              SchedInstruction *toNode, int ltncy,
                              DependenceType depType) {
#if defined(IS_DEBUG) || defined(IS_DEBUG_DAG)
  InstCount frmNodeNum = frmNode->GetNum();
  InstCount toNodeNum = toNode->GetNum();
#endif

#ifdef IS_DEBUG_DAG
  Logger::Info("Creating extra edge from %d to %d of type %d and latency %d",
               frmNodeNum, toNodeNum, depType, ltncy);
#endif

  //assert(frmNodeNum < instCnt_);
  //assert(nodes_[frmNodeNum] != NULL);

  //assert(toNodeNum < instCnt_);
  //assert(nodes_[toNodeNum] != NULL);

#ifdef IS_DEBUG_LATENCIES
  stats::dependenceTypeLatencies.Add(GetDependenceTypeName(depType), ltncy);
  if (depType == DEP_DATA) {
    stats::instructionTypeLatencies.Add(
        machMdl_->GetInstTypeNameByCode(frmNode->GetInstType()), ltncy);
  }
#endif

  GraphEdge *edge = frmNode->FindScsr(toNode);
  int crntLtncy;

  if (edge != NULL) {
    assert(toNode->FindPrdcsr(frmNode) == edge);
    crntLtncy = edge->label;
#ifdef IS_DEBUG_DAG
    Logger::Info("Found existing edge of label %d", crntLtncy);
#endif

    if (crntLtncy < ltncy) {
      edge->label = ltncy;
      insts_[edge->from].UpdtMaxEdgLbl(ltncy);
    }

    return;
  }

  GraphEdge *newEdg = new GraphEdge(frmNode->GetNum(), toNode->GetNum(), 
		                    ltncy, depType);
  // If compiling on device, keep track of the pointers to all edges
  // Set instCnt_ comparison to REGION_MIN_SIZE
  if (DEV_ACO && instCnt_ >= REGION_MIN_SIZE) {
    // if the edges_ vector has not been created, create it
    if (!edges_)
      edges_ = new std::vector<GraphEdge *>();
    edges_->push_back(newEdg);
  }

  edgeCnt_++;
  frmNode->AddScsr(newEdg);
  toNode->AddPrdcsr(newEdg);

  if (ltncy > maxLtncy_) {
    maxLtncy_ = ltncy;
  }
}

__host__
void DataDepGraph::CreateEdge_(InstCount frmNodeNum, InstCount toNodeNum,
                               int ltncy, DependenceType depType,
                               bool IsArtificial) {
  GraphEdge *edge;

  assert(frmNodeNum < instCnt_);
  assert(insts_ + frmNodeNum != NULL);

  assert(toNodeNum < instCnt_);

  assert(insts_ + toNodeNum != NULL);

  GraphNode *frmNode = insts_ + frmNodeNum;
  GraphNode *toNode = insts_ +toNodeNum;
  
#ifdef IS_DEBUG_LATENCIES
  stats::dependenceTypeLatencies.Add(GetDependenceTypeName(depType), ltncy);
  if (depType == DEP_DATA) {
    InstType inst = ((SchedInstruction *)frmNode)->GetInstType();
    stats::instructionTypeLatencies.Add(machMdl_->GetInstTypeNameByCode(inst),
                                        ltncy);
  }
#endif

  edge = frmNode->FindScsr(toNode);

  if (edge == NULL) {
#ifdef IS_DEBUG_DAG
    Logger::Info("Creating edge from %d to %d of type %d and latency %d",
                 frmNodeNum, toNodeNum, depType, ltncy);
#endif
    edge = new GraphEdge(frmNode->GetNum(), toNode->GetNum(), ltncy, depType,
                         IsArtificial);
    // If compiling on device, keep track of the pointers to all edges
    // Set instCnt_ comparison to REGION_MIN_SIZE
    if (DEV_ACO && instCnt_ >= REGION_MIN_SIZE) {
      // if the edges_ vector has not been created, create it
      if (!edges_)
        edges_ = new std::vector<GraphEdge *>();
      edges_->push_back(edge);
    }
    edgeCnt_++;
    frmNode->AddScsr(edge);
    toNode->AddPrdcsr(edge);
  } else {
    if (ltncy > edge->label) {
#ifdef IS_DEBUG_DAG
      Logger::Info("Setting latency of the edge from %d to %d to %d",
                   frmNodeNum, toNodeNum, ltncy);
#endif
      edge->label = ltncy;
      insts_[edge->from].UpdtMaxEdgLbl(ltncy);
    }
  }

  if (ltncy > maxLtncy_) {
    maxLtncy_ = ltncy;
  }

  if (ltncy <= MAX_LATENCY_VALUE) {
    edgeCntPerLtncy_[ltncy]++;
  }
}

FUNC_RESULT DataDepGraph::FinishNode_(InstCount nodeNum, InstCount edgeCnt) {
  if (edgeCnt != -1) {
    // assert(edgeCnt == insts_[nodeNum]->GetScsrCnt());
  }

  edgeCnt = insts_[nodeNum].GetScsrCnt();

  if (edgeCnt > maxScsrCnt_) {
    maxScsrCnt_ = edgeCnt;
  }

  if (edgeCnt == 0) {
    if (leaf_ != NULL) {
      Logger::Error("Multiple leaves are not allowed in the dependence "
                    "graph file.");
      return RES_ERROR;
    }

    leaf_ = &insts_[nodeNum];
  }

  UDT_GLABEL ltncySum = insts_[nodeNum].GetLtncySum();

  if (ltncySum > maxLtncySum_) {
    maxLtncySum_ = ltncySum;
  }

  return RES_SUCCESS;
}

void DataDepGraph::EnableBackTracking() { backTrackEnbl_ = true; }

void DataDepGraph::EnforceProgOrder() {
  for (InstCount i = 0; i < (instCnt_ - 1); i++) {
    CreateEdge_(i, i + 1, 0, DEP_DATA);
  }
}

void DataDepGraph::RestoreAbsoluteBounds() {
  for (InstCount i = 0; i < instCnt_; i++) {
    insts_[i].RestoreAbsoluteBounds();
  }
}

FUNC_RESULT DataDepGraph::WriteToFile(FILE *file, FUNC_RESULT rslt,
                                      InstCount imprvmnt, long number) {
  char titleStrng[MAX_NAMESIZE];
  bool prnt = false;

  switch (outptDags_) {
  case ODG_NONE:
    prnt = false;
    break;
  case ODG_IMP:

    if (rslt == RES_SUCCESS && imprvmnt > 0) {
      prnt = true;
    }

    break;
  case ODG_OPT:

    if (rslt == RES_SUCCESS) {
      prnt = true;
    }

    break;
  case ODG_HARD:

    if (isHard_) {
      prnt = true;
    }

    break;
  case ODG_ALL:
    prnt = true;
    break;
  }

  if (instCnt_ > maxOutptDagSize_) {
    prnt = false;
  }

  if (prnt == false) {
    return RES_FAIL;
  }

  strcpy(titleStrng, "dag");

  fprintf(file, "%s %d \"%s\"\n", titleStrng, instCnt_,
          machMdl_->GetModelName().c_str());

  fprintf(file, "{\n");

  if (strcmp(dagID_, "unknown") == 0) {
    sprintf(dagID_, "%ld", number);
  }

  fprintf(file, "dag_id %s\n", dagID_);

  fprintf(file, "dag_weight %f\n", weight_);

  fprintf(file, "compiler %s\n", compiler_);

  fprintf(file, "dag_lb %d \n", finalLwrBound_);
  fprintf(file, "dag_ub %d \n", finalUprBound_);

  WriteNodeInfoToF2File_(file);
  WriteDepInfoToF2File_(file);

  fprintf(file, "}\n");
  return RES_SUCCESS;
}

void DataDepGraph::WriteNodeInfoToF2File_(FILE *file) {
  InstCount i;

  fprintf(file, "nodes\n");

  for (i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = &insts_[i];
    fprintf(file, "  node %d ", inst->GetNum());

    if (i > 0 && i < (instCnt_ - 1)) {
      fprintf(file, "\"%s\"  \"%s\"\n", inst->GetName(), inst->GetOpCode());
    } else {
      fprintf(file, "\"%s\"\n", inst->GetName());
    }

    if (inst->GetInstType() != machMdl_->GetInstTypeByName("artificial")) {
      fprintf(file, "    sched_order %d\n", inst->GetFileSchedOrder());
      fprintf(file, "    issue_cycle %d\n", inst->GetFileSchedCycle());
    }
  }
}

void DataDepGraph::WriteDepInfoToF2File_(FILE *file) {
  fprintf(file, "dependencies\n");

  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = &insts_[i];

    int ltncy;
    DependenceType depType;
    for (SchedInstruction *scsr = inst->GetFrstScsr(NULL, &ltncy, &depType);
         scsr != NULL; scsr = inst->GetNxtScsr(NULL, &ltncy, &depType)) {
      const char *bareDepTypeName = GetDependenceTypeName(depType);
      int bareDepTypeLngth = strlen(bareDepTypeName);
      char depTypeName[MAX_NAMESIZE];
      addDblQuotes(bareDepTypeName, bareDepTypeLngth, depTypeName);
      fprintf(file, "  %s %d %d %s %d\n", "dep", inst->GetNum(), scsr->GetNum(),
              depTypeName, ltncy);
    }
  }
}

bool DataDepGraph::UseFileBounds() {
  bool match = true;

  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = GetInstByIndx(i);

    if (inst->UseFileBounds() == false) {
      match = false;
    }
  }

  return match;
}

__host__
void DataDepGraph::CmputRltvCrtclPaths_(DIRECTION dir) {
  InstCount i;

  if (dir == DIR_FRWRD) {
    for (i = 0; i < instCnt_; i++) {
      CmputCrtclPathsFrmRcrsvPrdcsr_(&insts_[i]);
    }
  } else {
    assert(dir == DIR_BKWRD);

    for (i = 0; i < instCnt_; i++) {
      CmputCrtclPathsFrmRcrsvScsr_(&insts_[i]);
    }
  }
}

__host__
void DataDepGraph::CmputCrtclPathsFrmRcrsvPrdcsr_(SchedInstruction *ref) {
  ArrayList<InstCount> *rcrsvScsrLst = ref->GetRcrsvNghbrLst(DIR_FRWRD);
  SchedInstruction *inst = GetLeafInst();
  InstCount nodeNum;

  assert(rcrsvScsrLst != NULL);

  // Visit the nodes in reverse topological order
  for (nodeNum = rcrsvScsrLst->GetLastElmnt(); nodeNum != END;
       nodeNum = rcrsvScsrLst->GetPrevElmnt()) {
    inst = &insts_[nodeNum];
    inst->CmputCrtclPathFrmRcrsvPrdcsr(ref);
  }

  assert(inst == GetLeafInst()); // the last instruction must be the leaf

  // The forward CP of the root relative to this entry must be
  // equal to the backward CP of the entry relative to the leaf
  assert(inst->CmputCrtclPathFrmRcrsvPrdcsr(ref) ==
         ref->GetCrtclPath(DIR_BKWRD));
}

__host__
void DataDepGraph::CmputCrtclPathsFrmRcrsvScsr_(SchedInstruction *ref) {
  ArrayList<InstCount> *rcrsvPrdcsrLst = ref->GetRcrsvNghbrLst(DIR_BKWRD);
  SchedInstruction *inst = GetRootInst();
  InstCount nodeNum;

  assert(rcrsvPrdcsrLst != NULL);

  // Visit the nodes in reverse topological order
  for (nodeNum = rcrsvPrdcsrLst->GetLastElmnt(); nodeNum != END;
       nodeNum = rcrsvPrdcsrLst->GetPrevElmnt()) {
    inst = &insts_[nodeNum];
    inst->CmputCrtclPathFrmRcrsvScsr(ref);
  }

  assert(inst == GetRootInst()); // the last instruction must be the root

  // The backward CP of the root relative to this exit must be
  // equal to the forward CP of th exit relative to the root
  assert(inst->CmputCrtclPathFrmRcrsvScsr(ref) == ref->GetCrtclPath(DIR_FRWRD));
}

void DataDepGraph::PrintLwrBounds(DIRECTION dir, std::ostream &out,
                                  const char *const title) {
  out << '\n' << title;
  for (InstCount i = 0; i < instCnt_; i++) {
    out << "\nLB(" << i << ")= " << insts_[i].GetLwrBound(dir);
  }
}

void DataDepGraph::PrintInstTypeInfo(FILE *file) {
  int16_t i;
  fprintf(file, "\n\nInst. Types:");

  for (i = 0; i < instTypeCnt_; i++) {
    fprintf(file, "\n%s: %d", machMdl_->GetInstTypeNameByCode(i),
            instCntPerType_[i]);
  }
}

void DataDepGraph::CountDeps(InstCount &totDepCnt, InstCount &crossDepCnt) {
  totDepCnt = 0;
  crossDepCnt = 0;

  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction *inst = &insts_[i];
    int ltncy;

    for (SchedInstruction *child = inst->GetFrstScsr(&ltncy); child != NULL;
         child = inst->GetNxtScsr(&ltncy)) {
      if (inst->GetIssueType() != child->GetIssueType()) {
        crossDepCnt++;
      }
      totDepCnt++;
    }
  }
}

/*void DataDepGraph::CountDefs(RegisterFile regFiles[]) {
  int intDefCnt = 0, fpDefCnt = 0;

  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction* inst = insts_[i];
    int ltncy;
    DependenceType depType;
    for (SchedInstruction* child = inst->GetFrstScsr(NULL, &ltncy, &depType);
         child != NULL;
         child = inst->GetNxtScsr(NULL, &ltncy, &depType)) {
      if (depType == DEP_DATA) {
        if (machMdl_->IsFloat(inst->GetInstType())) {
          fpDefCnt++;
        } else {
          intDefCnt++;
        }
        break;
      }
    }
  }

  regFiles[0].SetRegCnt(intDefCnt);
  regFiles[1].SetRegCnt(fpDefCnt);
}*/

/*
void DataDepGraph::AddDefsAndUses(RegisterFile regFiles[]) {
  int intRegCnt = 0, fpRegCnt = 0;

  for (InstCount i = 0; i < instCnt_; i++) {
    SchedInstruction* inst = insts_[i];
    Register* reg = NULL;

    DependenceType depType;
    int ltncy;
    for (SchedInstruction* child = inst->GetFrstScsr(NULL, &ltncy, &depType);
         child != NULL;
         child = inst->GetNxtScsr(NULL, &ltncy, &depType)) {
      if (depType == DEP_DATA) {
        if (reg == NULL) {
          int16_t regType;
          int* regCntPtr;
          if (machMdl_->IsFloat(inst->GetInstType())) {
            regType = 1;
            regCntPtr = &fpRegCnt;
          } else {
            regType = 0;
            regCntPtr = &intRegCnt;
          }

          reg = regFiles[regType].GetReg(*regCntPtr);
          (*regCntPtr)++;
          inst->AddDef(reg);
          reg->AddDef();
        }

        child->AddUse(reg);
        reg->AddUse();
      }
    }
  }
}*/

/*
void DataDepGraph::AddOutputEdges() {
  // Nothing to do. By default output edges are predefined.
}
*/

void DataDepGraph::PrintEdgeCntPerLtncyInfo() {
  int totEdgeCnt = 0;
  Logger::Info("Latency Distribution:");
  for (int i = 0; i <= MAX_LATENCY_VALUE; i++) {
    if (edgeCntPerLtncy_[i] > 0)
      Logger::Info("Latency %d: %d edges", i, edgeCntPerLtncy_[i]);
    totEdgeCnt += edgeCntPerLtncy_[i];
  }
  Logger::Info("Total edge count: %d", totEdgeCnt);
}

InstCount DataDepGraph::GetRltvCrtclPath(SchedInstruction *ref,
                                         SchedInstruction *inst,
                                         DIRECTION dir) {
  return inst->GetRltvCrtclPath(dir, ref);
}

/*
DataDepSubGraph::DataDepSubGraph(DataDepGraph *fullGraph, InstCount maxInstCnt,
                                 MachineModel *machMdl)
    : DataDepStruct(machMdl) {
  InstCount i;

  fullGraph_ = fullGraph;
  type_ = DGT_SUB;

  maxInstCnt += 2;
  maxInstCnt_ = maxInstCnt;
  subType_ = SGT_DISC;

  insts_ = new SchedInstruction *[maxInstCnt_];

  for (i = 0; i < maxInstCnt; i++) {
    insts_[i] = NULL;
  }

  frwrdCrtclPaths_ = NULL;
  bkwrdCrtclPaths_ = NULL;
  dynmcFrwrdLwrBounds_ = NULL;
  dynmcBkwrdLwrBounds_ = NULL;
  rootInst_ = NULL;
  leafInst_ = NULL;
  instCnt_ = 2;
  extrnlInstCnt_ = 0;
  cmpnstdInstCnt_ = 0;
  instsChngd_ = false;
  insts_[0] = rootInst_;
  insts_[instCnt_ - 1] = leafInst_;

  dynmcLwrBoundsSet_ = false;
  schedLwrBound_ = 2;
  totLwrBound_ = 0;
  unstsfidLtncy_ = 0;
  rejoinCycle_ = INVALID_VALUE;
  instAdded_ = false;

  rootVctr_ = new BitVector(fullGraph_->GetInstCnt());
  leafVctr_ = new BitVector(fullGraph_->GetInstCnt());

  numToIndx_ = new InstCount[fullGraph_->GetInstCnt()];

  fxdLst_ = NULL;

  lostInsts_ = new Stack<LostInst>;

  for (i = 0; i < fullGraph_->GetInstCnt(); i++) {
    numToIndx_[i] = INVALID_VALUE;
  }

  dynmcRlxdSchdulr_ = NULL;
  RJRlxdSchdulr_ = NULL;
  RJRvrsRlxdSchdulr_ = NULL;
  LCRlxdSchdulr_ = NULL;
  LCRvrsRlxdSchdulr_ = NULL;
  includesUnpipelined_ = false;

#ifdef IS_DEBUG
  errorCnt_ = 0;
#endif
#ifdef IS_DEBUG_TRACE_ENUM
  smplDynmcLB_ = false;
#endif
}

__host__ __device__
DataDepSubGraph::~DataDepSubGraph() {
#ifdef __HIP_DEVICE_COMPILE__
#else
  DelRootAndLeafInsts_(true);
#endif

  delete[] insts_;
  delete[] numToIndx_;

  if (frwrdCrtclPaths_ != NULL)
    delete[] frwrdCrtclPaths_;
  if (bkwrdCrtclPaths_ != NULL)
    delete[] bkwrdCrtclPaths_;
  if (dynmcFrwrdLwrBounds_ != NULL)
    delete[] dynmcFrwrdLwrBounds_;
  if (dynmcBkwrdLwrBounds_ != NULL)
    delete[] dynmcBkwrdLwrBounds_;
  if (dynmcRlxdSchdulr_ != NULL)
    delete dynmcRlxdSchdulr_;
  if (RJRlxdSchdulr_ != NULL)
    delete RJRlxdSchdulr_;
  if (RJRvrsRlxdSchdulr_ != NULL)
    delete RJRvrsRlxdSchdulr_;
  if (LCRlxdSchdulr_ != NULL)
    delete LCRlxdSchdulr_;
  if (LCRvrsRlxdSchdulr_ != NULL)
    delete LCRvrsRlxdSchdulr_;

  assert(lostInsts_ != NULL);
  for (LostInst *inst = lostInsts_->GetFrstElmnt(); inst != NULL;
       inst = lostInsts_->GetNxtElmnt()) {
    delete inst;
  }
  lostInsts_->Reset();
  delete lostInsts_;
}

void DataDepSubGraph::SetupForDynmcLwrBounds(InstCount schedUprBound) {
  // To account for the root and leaf insts
  InstCount subGraphUprBound = schedUprBound + 2;
  subGraphUprBound += SCHED_UB_EXTRA;

  dynmcRlxdSchdulr_ = new RJ_RelaxedScheduler(
      this, machMdl_, subGraphUprBound, DIR_FRWRD, RST_SUBDYNMC, maxInstCnt_);

  AllocDynmcData_();
}

void DataDepSubGraph::AllocSttcData_() {
  frwrdCrtclPaths_ = new InstCount[maxInstCnt_];
  bkwrdCrtclPaths_ = new InstCount[maxInstCnt_];

  frwrdLwrBounds_ = new InstCount[maxInstCnt_];
  bkwrdLwrBounds_ = new InstCount[maxInstCnt_];
}

void DataDepSubGraph::AllocDynmcData_() {
  dynmcFrwrdLwrBounds_ = new InstCount[maxInstCnt_];
  dynmcBkwrdLwrBounds_ = new InstCount[maxInstCnt_];
}

// Called before lower bound computation after all instructions have been added
void DataDepSubGraph::InitForLwrBounds_() {
  assert(instCnt_ >= 3);
  DelRootAndLeafInsts_(false);
  CreateRootAndLeafInsts_();
  SetRootsAndLeaves_();
  CmputAbslutUprBound_();
}

// Called before dynamic lower bound computation after all insts. have been
// added
void DataDepSubGraph::InitForDynmcLwrBounds_() {
  assert(instCnt_ >= 3);

  if (instsChngd_) {
    DelRootAndLeafInsts_(false);
    CreateRootAndLeafInsts_();
    SetRootsAndLeaves_();

    UpdtSttcLwrBounds_();
    instsChngd_ = false;
  }
}

void DataDepSubGraph::UpdtSttcLwrBounds_() {
  // TEMP CHANGE
  //  InstCount extrnlIndx=instCnt_-1-extrnlInstCnt_;
  //  InstCount extrnlIndx = 0;

  frwrdCrtclPaths_[0] = 0; // Set the leaf's BCP to 0
  frwrdLwrBounds_[0] = 0;  // Set the leaf's BLB to 0

  // Recompute the forward CPs and LBs of all external insts and the leaf inst
  PropagateFrwrdLwrBounds_(1, instCnt_ - 1, frwrdCrtclPaths_, true);
  PropagateFrwrdLwrBounds_(1, instCnt_ - 1, frwrdLwrBounds_, true);

  bkwrdCrtclPaths_[instCnt_ - 1] = 0; // Set the leaf's BCP to 0
  bkwrdLwrBounds_[instCnt_ - 1] = 0;  // Set the leaf's BLB to 0

  // Recompute all backward critical paths
  PropagateBkwrdLwrBounds_(instCnt_ - 2, 0, bkwrdCrtclPaths_, true);

  // Recompute backward lower bounds of external insts only and keep the
  // static backward lower bounds since they are potentially tighter
  if (extrnlInstCnt_ > 0) {
    PropagateBkwrdLwrBounds_(instCnt_ - 2, 0, bkwrdLwrBounds_, true);
  }
}

void DataDepSubGraph::InitForSchdulng(bool clearAll) {
  assert(clearAll == false || extrnlInstCnt_ == 0);

  //  SchedInstruction* inst;
  LostInst *lostInst;

  while ((lostInst = lostInsts_->ExtractElmnt()) != NULL) {
    InstCount indx = lostInst->indx;
    InstCount num = lostInst->inst->GetNum();
    assert(numToIndx_[num] == INVALID_VALUE);
    numToIndx_[num] = indx;
    delete lostInst;
  }

  if (clearAll) {
    Clear_();
  } else {
    RmvExtrnlInsts_();
  }

  if (rootInst_ != NULL) {
    assert(leafInst_ != NULL);
    DelRootAndLeafInsts_(false);
    CreateRootAndLeafInsts_();

    if (instCnt_ > 2) {
      SetRootsAndLeaves_();
    }

    UpdtSttcLwrBounds_();
  }

  instsChngd_ = false;
  cmpnstdInstCnt_ = 0;
}

void DataDepSubGraph::CreateRootAndLeafInsts_() {
  InstType instType = machMdl_->GetInstTypeByName("artificial");
  IssueType issuType = machMdl_->GetIssueType(instType);
  assert(0 <= issuType && issuType < issuTypeCnt_);

  assert(rootInst_ == NULL && leafInst_ == NULL);

  rootInst_ =
      new SchedInstruction(INVALID_VALUE, "root", instType, " ", maxInstCnt_, 0,
                           INVALID_VALUE, INVALID_VALUE, 0, 0, machMdl_);

  rootInst_->SetIssueType(issuType);

  leafInst_ =
      new SchedInstruction(INVALID_VALUE, "leaf", instType, " ", maxInstCnt_, 0,
                           INVALID_VALUE, INVALID_VALUE, 0, 0, machMdl_);

  leafInst_->SetIssueType(issuType);

  InstCount rootIndx = 0;
  InstCount leafIndx = instCnt_ - 1;

  insts_[rootIndx] = rootInst_;
  insts_[leafIndx] = leafInst_;

  rootVctr_->Reset();
  leafVctr_->Reset();
}

void DataDepSubGraph::DelRootAndLeafInsts_(bool isFinal) {
  if (rootInst_ != NULL) {
    delete rootInst_;
    rootInst_ = NULL;
  }

  if (leafInst_ != NULL) {
    leafInst_->DelPrdcsrLst();
    delete leafInst_;
    leafInst_ = NULL;
  }

  if (isFinal) {
    delete rootVctr_;
    delete leafVctr_;
  } else {
    rootVctr_->Reset();
    leafVctr_->Reset();
  }

  insts_[0] = rootInst_;
  insts_[instCnt_ - 1] = leafInst_;
}

void DataDepSubGraph::RmvExtrnlInsts_() {
  InstCount i;
  InstCount indx = instCnt_ - 2;

  for (i = 0; i < extrnlInstCnt_; i++, indx--) {
    RmvInst(insts_[indx]);
  }

  extrnlInstCnt_ = 0;
  instsChngd_ = true;
}

// Called to clear all current instructions and get ready for a new set of
// instructions
void DataDepSubGraph::Clear_() {
  assert(subType_ == SGT_DISC);
  instCnt_ = 2;
  RmvExtrnlInsts_();
  schedLwrBound_ = 2;
  totLwrBound_ = 0;
  unstsfidLtncy_ = 0;
  rejoinCycle_ = INVALID_VALUE;
  instAdded_ = false;

  InstCount i;

  for (i = 0; i < fullGraph_->GetInstCnt(); i++) {
    numToIndx_[i] = INVALID_VALUE;
  }

  for (i = 0; i < issuTypeCnt_; i++) {
    instCntPerIssuType_[i] = 0;
  }
}

void DataDepSubGraph::AddInst(SchedInstruction *inst) {
  // Logger::Info("Adding inst %d to subdag %d of size %d",
  //             inst->GetNum(), this, instCnt_);
  assert(instCnt_ >= 2);
  assert(instCnt_ < maxInstCnt_);
  assert(inst != NULL);
  assert(insts_[instCnt_ - 1] == leafInst_);
  instCnt_++;
  insts_[instCnt_ - 2] = inst;
  insts_[instCnt_ - 1] = leafInst_;
  numToIndx_[inst->GetNum()] = instCnt_ - 2;

  if (subType_ == SGT_DISC && inst != rootInst_ && inst != leafInst_) {
    IssueType issuType = inst->GetIssueType();
    assert(issuType < issuTypeCnt_);
    instCntPerIssuType_[issuType]++;
  }

  instAdded_ = true;
}

void DataDepSubGraph::RmvInst(SchedInstruction *inst) {
  // Logger::Info("Removing inst %d from subdag %d of size %d.",
  //             inst->GetNum(), this, instCnt_);

  assert(instCnt_ >= 3);
  assert(inst != NULL);
  assert(insts_[instCnt_ - 2] == inst);
  numToIndx_[inst->GetNum()] = INVALID_VALUE;
  instCnt_--;

  if (subType_ == SGT_DISC && inst != rootInst_ && inst != leafInst_) {
    IssueType issuType = inst->GetIssueType();
    assert(issuType < issuTypeCnt_);
    instCntPerIssuType_[issuType]--;
  }

  insts_[instCnt_ - 1] = leafInst_;
}

void DataDepSubGraph::RmvExtrnlInst(SchedInstruction *inst) {
  // Logger::Info("Removing external inst %d from subdag %d",
  //             inst->GetNum(), this);
  RmvInst(inst);

  extrnlInstCnt_--;
  instsChngd_ = true;
}

void DataDepSubGraph::AddExtrnlInst(SchedInstruction *inst) {
  // Logger::Info("Adding external inst %d to subdag %d of size %d",
  //             inst->GetNum(), this, instCnt_);
  AddInst(inst);

  extrnlInstCnt_++;
  instsChngd_ = true;

  assert(IsLeaf_(inst));
  assert(numToIndx_[inst->GetNum()] == instCnt_ - 2);
}

InstCount DataDepSubGraph::GetAvlblSlots(IssueType issuType) {
  int slotsPerCycle = machMdl_->GetSlotsPerCycle(issuType);
  assert(schedLwrBound_ >= 2);
  int totSlots = (schedLwrBound_ - 2) * slotsPerCycle;
  int avlblSlots = totSlots - instCntPerIssuType_[issuType];
  assert(avlblSlots >= 0);
  return avlblSlots;
}

void DataDepSubGraph::InstLost(SchedInstruction *inst) {
  assert(inst != NULL);

  assert(inst != rootInst_ && inst != leafInst_);
  InstCount instNum = inst->GetNum();
  assert(0 <= instNum && instNum < fullGraph_->GetInstCnt());
  InstCount instIndx = numToIndx_[instNum];
  assert(instIndx != INVALID_VALUE);
  LostInst *lostInst = new LostInst;

  lostInst->inst = inst;
  lostInst->indx = instIndx;
  lostInsts_->InsrtElmnt(lostInst);
  numToIndx_[instNum] = INVALID_VALUE;

  instsChngd_ = true;
}

void DataDepSubGraph::UndoInstLost(SchedInstruction *inst) {
  assert(inst != NULL);
  assert(inst != rootInst_ && inst != leafInst_);

  InstCount instNum = inst->GetNum();
  assert(0 <= instNum && instNum < fullGraph_->GetInstCnt());

  assert(numToIndx_[instNum] == INVALID_VALUE);
  LostInst *lostInst = lostInsts_->ExtractElmnt();

  assert(lostInst->inst == inst);
  InstCount instIndx = lostInst->indx;
  numToIndx_[instNum] = instIndx;
  delete lostInst;

  instsChngd_ = true;
}

void DataDepSubGraph::PropagateFrwrdLwrBounds_(InstCount frmIndx,
                                               InstCount toIndx,
                                               InstCount frwrdLwrBounds[],
                                               bool reset) {
  InstCount indx;

  assert(frmIndx <= toIndx);

  for (indx = frmIndx; indx <= toIndx; indx++) {
    assert(0 < indx && indx < instCnt_);
    if (!IsInGraph(insts_[indx]))
      continue;
    if (reset)
      frwrdLwrBounds[indx] = 1;
    TightnLwrBound_(DIR_FRWRD, indx, frwrdLwrBounds);
  }
}

void DataDepSubGraph::PropagateBkwrdLwrBounds_(InstCount frmIndx,
                                               InstCount toIndx,
                                               InstCount bkwrdLwrBounds[],
                                               bool reset) {
  InstCount indx;

  assert(frmIndx >= toIndx);

  for (indx = frmIndx; indx >= toIndx; indx--) {
    assert(0 <= indx && indx < instCnt_ - 1);
    if (!IsInGraph(insts_[indx]))
      return;
    if (reset)
      bkwrdLwrBounds[indx] = 1;
    TightnLwrBound_(DIR_BKWRD, indx, bkwrdLwrBounds);
  }
}

void DataDepSubGraph::TightnLwrBound_(DIRECTION dir, InstCount indx,
                                      InstCount lwrBounds[]) {
  assert((dir == DIR_FRWRD && indx > 0 && indx < instCnt_) ||
         (dir == DIR_BKWRD && indx >= 0 && indx < instCnt_ - 1));

  SchedInstruction *inst = insts_[indx];
  InstCount lwrBound = lwrBounds[indx];
  DIRECTION opstDir = DirAcycGraph::ReverseDirection(dir);
  UDT_GLABEL ltncy;

  for (SchedInstruction *nghbr = inst->GetFrstNghbr(opstDir, &ltncy);
       nghbr != NULL; nghbr = inst->GetNxtNghbr(opstDir, &ltncy)) {
    if (IsInGraph(nghbr)) {
      InstCount nghbrIndx = numToIndx_[nghbr->GetNum()];
      assert((dir == DIR_FRWRD && nghbrIndx < indx) ||
             (dir == DIR_BKWRD && nghbrIndx > indx));
      InstCount rltvLwrBound = lwrBounds[nghbrIndx] + ltncy;

      if (rltvLwrBound > lwrBound) {
        lwrBound = rltvLwrBound;
      }
    }
  }

  lwrBounds[indx] = lwrBound;
}

void DataDepSubGraph::SetRootsAndLeaves_() {
  SchedInstruction *inst;
  InstCount i;

  assert(instCnt_ >= 3);
  rootInst_->AllocRcrsvInfo(DIR_FRWRD, fullGraph_->GetInstCnt());
  leafInst_->AllocRcrsvInfo(DIR_BKWRD, fullGraph_->GetInstCnt());

  for (i = 1; i < instCnt_ - 1; i++) {
    inst = insts_[i];

    if (!IsInGraph(inst))
      continue;
    if (IsRoot_(inst))
      AddRoot_(inst);
    if (IsLeaf_(inst))
      AddLeaf_(inst);

    rootInst_->AddRcrsvNghbr(inst, DIR_FRWRD);
    leafInst_->AddRcrsvNghbr(inst, DIR_BKWRD);
#ifdef IS_DEBUG
    if (i > 1) {
      assert(!insts_[i]->IsRcrsvScsr(insts_[i - 1]));
      assert(!insts_[i - 1]->IsRcrsvPrdcsr(insts_[i]));
    }
#endif
  }

  assert(rootInst_->GetScsrCnt() >= 1);
  assert(leafInst_->GetPrdcsrCnt() >= 1);
}

void DataDepSubGraph::AddRoot_(SchedInstruction *inst) {
  CreateEdge_(rootInst_, inst);
  rootVctr_->SetBit(inst->GetNum());
}

void DataDepSubGraph::AddLeaf_(SchedInstruction *inst) {
  CreateEdge_(inst, leafInst_);
  leafVctr_->SetBit(inst->GetNum());
}

void DataDepSubGraph::RmvLastRoot_(SchedInstruction *inst) {
  rootInst_->RmvLastScsr(inst, true);
  rootVctr_->SetBit(inst->GetNum(), false);
}

void DataDepSubGraph::RmvLastLeaf_(SchedInstruction *inst) {
  leafInst_->RmvLastPrdcsr(inst, true);
  leafVctr_->SetBit(inst->GetNum(), false);
}

void DataDepSubGraph::CreateEdge_(SchedInstruction *frmInst,
                                  SchedInstruction *toInst) {
  GraphNode *frmNode = frmInst;
  GraphNode *toNode = toInst;

  //  assert(frmInst==rootInst_ || toInst==leafInst_);
  GraphEdge *newEdg = new GraphEdge(frmNode, toNode, 1);

  if (toInst != leafInst_) {
    frmNode->ApndScsr(newEdg);
  }

  if (frmInst != rootInst_) {
    toNode->ApndPrdcsr(newEdg);
  }
}

void DataDepSubGraph::RmvEdge_(SchedInstruction *frmInst,
                               SchedInstruction *toInst) {
  GraphNode *frmNode = frmInst;
  GraphNode *toNode = toInst;

  frmNode->RmvLastScsr(toInst, false);
  toNode->RmvLastPrdcsr(frmInst, true);
}

bool DataDepSubGraph::IsRoot_(SchedInstruction *inst) {
  bool isRoot = true;
  SchedInstruction *pred;
  InstCount num;
  assert(inst != NULL);
  assert(IsInGraph(inst));

  for (pred = inst->GetFrstPrdcsr(&num); pred != NULL;
       pred = inst->GetNxtPrdcsr(&num)) {
    assert(pred != inst);
    // If the instruction has a predecessor that belongs to this subDAG, then
    // it is not a root instruction of the subDAG.
    if (IsInGraph(pred)) {
      isRoot = false;
      break;
    }
  }

  return isRoot;
}

bool DataDepSubGraph::IsLeaf_(SchedInstruction *inst) {
  bool isLeaf = true;
  InstCount num;
  assert(inst != NULL);
  assert(IsInGraph(inst));

  for (SchedInstruction *scsr = inst->GetFrstScsr(&num); scsr != NULL;
       scsr = inst->GetNxtScsr(&num)) {
    assert(scsr != inst);

    // If the instruction has a successor that belongs to this subDAG, then
    // it is not a leaf instruction of the subDAG.
    if (IsInGraph(scsr)) {
      isLeaf = false;
      break;
    }
  }

  return isLeaf;
}

void DataDepSubGraph::CmputTotLwrBound(LB_ALG lbAlg, InstCount rejoinCycle,
                                       SchedInstruction *inst,
                                       InstCount &lwrBound,
                                       InstCount &unstsfidLtncy,
                                       bool &crtnRejoin,
                                       InstCount &instGapSize) {

  totLwrBound_ = CmputLwrBound(lbAlg, true, rejoinCycle, inst, instGapSize);
  lwrBound = totLwrBound_;
  unstsfidLtncy = unstsfidLtncy_;
  rejoinCycle_ = crtnRejoin ? rejoinCycle : INVALID_VALUE;
}

InstCount DataDepSubGraph::CmputLwrBound(LB_ALG lbAlg, bool addExtrnlLtncs,
                                         InstCount rejoinCycle,
                                         SchedInstruction *inst,
                                         InstCount &instGapSize) {
  assert(instCnt_ >= 2);

  if (instCnt_ == 2) {
    assert(instAdded_ == false);
    schedLwrBound_ = 2;
    instGapSize = 0;
    unstsfidLtncy_ = 0;
    return 0;
  }

  InitForLwrBounds_();
  CmputCrtclPaths_();

  InstCount i;

  for (i = 0; i < instCnt_; i++) {
    frwrdLwrBounds_[i] = frwrdCrtclPaths_[i];
    bkwrdLwrBounds_[i] = bkwrdCrtclPaths_[i];
  }

  if (addExtrnlLtncs) {
    unstsfidLtncy_ = CmputUnstsfidLtncy_();

    if (unstsfidLtncy_ > 0) {
      instGapSize = CmputExtrnlLtncs_(rejoinCycle, inst);
    }
  }

  RelaxedScheduler *rlxdSchdulr = NULL;
  RelaxedScheduler *rvrsRlxdSchdulr = NULL;

  AllocRlxdSchdulr_(lbAlg, rlxdSchdulr, rvrsRlxdSchdulr);

  InstCount frwrdLwrBound = rlxdSchdulr->FindSchedule();
  InstCount bkwrdLwrBound = rvrsRlxdSchdulr->FindSchedule();
  schedLwrBound_ = std::max(frwrdLwrBound, bkwrdLwrBound);

  assert(schedLwrBound_ <= schedUprBound_);

  FreeRlxdSchdulr_(lbAlg);
  instAdded_ = false;
  return schedLwrBound_ - 2;
}

__host__ __device__
InstCount DataDepSubGraph::CmputAbslutUprBound_() {
  InstCount maxLtncy;
  //std::max(1, fullGraph_->GetMaxLtncy());
  if (1 > fullGraph_->GetMaxLtncy())
    maxLtncy = 1;
  else
    maxLtncy = fullGraph_->GetMaxLtncy();

  assert(maxInstCnt_ >= 3);
  InstCount ltncySum = maxLtncy * maxInstCnt_;
  schedUprBound_ = ltncySum + 1;
  return schedUprBound_;
}

void DataDepSubGraph::AllocRlxdSchdulr_(LB_ALG lbAlg,
                                        RelaxedScheduler *&rlxdSchdulr,
                                        RelaxedScheduler *&rvrsRlxdSchdulr) {
  switch (lbAlg) {
  case LBA_RJ:
    if (RJRlxdSchdulr_ == NULL) {
      RJRlxdSchdulr_ = new RJ_RelaxedScheduler(
          this, machMdl_, schedUprBound_, DIR_FRWRD, RST_STTC, maxInstCnt_);
    }

    if (RJRvrsRlxdSchdulr_ == NULL) {
      RJRvrsRlxdSchdulr_ = new RJ_RelaxedScheduler(
          this, machMdl_, schedUprBound_, DIR_BKWRD, RST_STTC, maxInstCnt_);
    }

    rlxdSchdulr = RJRlxdSchdulr_;
    rvrsRlxdSchdulr = RJRvrsRlxdSchdulr_;
    break;
  case LBA_LC:
    // if(LCRlxdSchdulr_==NULL)
    LCRlxdSchdulr_ =
        new LC_RelaxedScheduler(this, machMdl_, schedUprBound_, DIR_FRWRD);
    // if(LCRvrsRlxdSchdulr_==NULL)
    LCRvrsRlxdSchdulr_ =
        new LC_RelaxedScheduler(this, machMdl_, schedUprBound_, DIR_BKWRD);
    rlxdSchdulr = LCRlxdSchdulr_;
    rvrsRlxdSchdulr = LCRvrsRlxdSchdulr_;
  }
}

void DataDepSubGraph::FreeRlxdSchdulr_(LB_ALG lbAlg) {
  if (lbAlg == LBA_LC) {
    delete LCRlxdSchdulr_;
    delete LCRvrsRlxdSchdulr_;
    LCRlxdSchdulr_ = NULL;
    LCRvrsRlxdSchdulr_ = NULL;
  } else {
    assert(lbAlg == LBA_RJ);
  }
}

InstCount DataDepSubGraph::CmputUnstsfidLtncy_() {
  InstCount unstsfidLtncy = 0;
  InstCount maxUnstsfidLtncy = fullGraph_->GetMaxLtncy() - 1;
  InstCount i;

  for (i = instCnt_ - 2; i >= 1; i--) {
    SchedInstruction *inst = insts_[i];
    InstCount instLtncy = inst->GetMaxLtncy() - 1;
    instLtncy -= (bkwrdLwrBounds_[i] - 1);

    if (instLtncy > unstsfidLtncy) {
      unstsfidLtncy = instLtncy;
    }

    if (unstsfidLtncy == maxUnstsfidLtncy) {
      break;
    }
  }

  return unstsfidLtncy;
}

bool DataDepSubGraph::TightnDynmcLwrBound_(InstCount frstCycle,
                                           InstCount minLastCycle,
                                           InstCount maxLastCycle,
                                           InstCount trgtLwrBound,
                                           InstCount &dynmcLwrBound) {
  InstCount lastCycle;
  InstCount pushDwn = frstCycle - 1;
  bool trgtFsbl = false;
  bool lastCycleFsbl;
  InstCount lwrBound;
  int iterCnt = 0;
  InstCount initDynmcLwrBound = dynmcLwrBound;

  trgtLwrBound += pushDwn;

  assert(frstCycle != INVALID_VALUE);

  for (lastCycle = minLastCycle; lastCycle <= maxLastCycle; lastCycle++) {
    bool fsbl = SetDynmcLwrBounds_(frstCycle, lastCycle, 0, trgtLwrBound, true,
                                   lastCycleFsbl);
    iterCnt++;

    if (!fsbl || !lastCycleFsbl)
      continue;

    fsbl = ChkInstRanges_(lastCycle);

    if (!fsbl)
      continue;

    dynmcRlxdSchdulr_->SetupPrirtyLst();

    fsbl = dynmcRlxdSchdulr_->CmputDynmcLwrBound(lastCycle, trgtLwrBound,
                                                 lwrBound);

    if (fsbl) {
      assert((lwrBound - pushDwn) >= dynmcLwrBound);
      assert(dynmcLwrBound >= schedLwrBound_);
      trgtFsbl = (dynmcLwrBound <= trgtLwrBound);
      break;
    } else {
      dynmcLwrBound++;
    }
  }

  assert(trgtFsbl);
  if (!trgtFsbl) {
    trgtFsbl = true;
    dynmcLwrBound = initDynmcLwrBound;
  }

  return trgtFsbl;
}

bool DataDepSubGraph::CmputEntTrmnlDynmcLwrBound_(InstCount &dynmcLwrBound,
                                                  InstCount trgtLwrBound) {
  assert(instCnt_ >= 3);
  assert(schedLwrBound_ >= 3);

  trgtLwrBound += 2;
  InstCount frstCycle = 1;
  InstCount trgtLastCycle = CmputMaxDeadline_() + 2;
  InstCount shft = 1;

  assert(trgtLwrBound >= schedLwrBound_);
  dynmcLwrBound = INVALID_VALUE;

  bool fsbl = true;
  bool trgtFsbl = true;

  fsbl = SetDynmcLwrBounds_(frstCycle, trgtLastCycle, shft, trgtLwrBound, false,
                            trgtFsbl);

  if (!fsbl || !trgtFsbl) {
    return false;
  }

  dynmcRlxdSchdulr_->Initialize(true);
  fsbl = ChkInstRanges_(trgtLastCycle);
  assert(fsbl);
  if (!fsbl)
    return false;

  fsbl = dynmcRlxdSchdulr_->CmputDynmcLwrBound(trgtLastCycle, trgtLwrBound,
                                               dynmcLwrBound);

  // assert(dynmcLwrBound == INVALID_VALUE || dynmcLwrBound >= schedLwrBound_);
  if (dynmcLwrBound != INVALID_VALUE && dynmcLwrBound < schedLwrBound_) {
    dynmcLwrBound = schedLwrBound_;
  }

#ifdef IS_DEBUG
  if (dynmcLwrBound != INVALID_VALUE && dynmcLwrBound < schedLwrBound_) {
    Logger::Info("Problematic graph encountered.");
  }
#endif

  dynmcLwrBound -= 2;
  return fsbl;
}

void DataDepSubGraph::FindFrstCycleRange_(InstCount &minFrstCycle,
                                          InstCount &maxFrstCycle) {
  minFrstCycle = INVALID_VALUE;
  maxFrstCycle = INVALID_VALUE;

  for (SchedInstruction *inst = rootInst_->GetFrstScsr(); inst != NULL;
       inst = rootInst_->GetNxtScsr()) {
    InstCount releaseTime = inst->GetCrntReleaseTime();
    InstCount deadline = inst->GetCrntDeadline();
    assert(inst->IsSchduld() == false || releaseTime == deadline);

    if (releaseTime == 0) {
      assert(inst == fullGraph_->GetRootInst());
      minFrstCycle = 0;
      maxFrstCycle = 0;
      break;
    }

    if (minFrstCycle == INVALID_VALUE || releaseTime < minFrstCycle) {
      minFrstCycle = releaseTime;
    }

    if (maxFrstCycle == INVALID_VALUE || deadline < maxFrstCycle) {
      maxFrstCycle = deadline;
    }
  }
}

// static int gTightGlblDLs=0;

bool DataDepSubGraph::SetDynmcLwrBounds_(InstCount frstCycle,
                                         InstCount lastCycle, InstCount shft,
                                         InstCount trgtLwrBound,
                                         bool useDistFrmLeaf, bool &trgtFsbl) {
  bool fsbl = true;
  trgtFsbl = true;

  fsbl = SetDynmcFrwrdLwrBounds_(frstCycle, lastCycle, shft);

  if (fsbl == false) {
    return fsbl;
  }

  InstCount minLastCycle = dynmcFrwrdLwrBounds_[instCnt_ - 1];
  trgtFsbl = minLastCycle <= (trgtLwrBound - 1);

  fsbl = SetDynmcBkwrdLwrBounds_(lastCycle, shft, useDistFrmLeaf);

  dynmcLwrBoundsSet_ = true;
  return fsbl;
}

bool DataDepSubGraph::SetDynmcFrwrdLwrBounds_(InstCount frstCycle,
                                              InstCount lastCycle,
                                              InstCount shft) {
  InstCount i;
  bool tightnFrwrd = false;
  bool fsbl = true;

  assert(frwrdLwrBounds_[0] == 0); // FLB of the root
  dynmcFrwrdLwrBounds_[0] = 0;     // FLB of the root

  for (i = 1; i < (instCnt_ - 1); i++) {
    SchedInstruction *inst = insts_[i];

    if (IsInGraph(inst) == false) {
      continue;
    }

    dynmcFrwrdLwrBounds_[i] = inst->GetCrntReleaseTime() + shft;
    assert(frwrdLwrBounds_[i] >= frwrdCrtclPaths_[i]);
    assert(GetLostInstCnt_() > 0 ||
           dynmcFrwrdLwrBounds_[i] >= frwrdLwrBounds_[i]);
    assert(shft == 0 || dynmcFrwrdLwrBounds_[i] >= frstCycle);

    if (dynmcFrwrdLwrBounds_[i] < frstCycle) {
      dynmcFrwrdLwrBounds_[i] = frstCycle;
      tightnFrwrd = true;
    }

    if (tightnFrwrd) {
      TightnLwrBound_(DIR_FRWRD, i, dynmcFrwrdLwrBounds_);
    }
  }

  // FLB of the leaf
  InstCount adjstdSttcBound = frwrdLwrBounds_[instCnt_ - 1] + frstCycle - 1;
  dynmcFrwrdLwrBounds_[instCnt_ - 1] = adjstdSttcBound;
  TightnLwrBound_(DIR_FRWRD, instCnt_ - 1, dynmcFrwrdLwrBounds_);

  if (dynmcFrwrdLwrBounds_[instCnt_ - 1] > lastCycle) {
    fsbl = false;
  }

  return fsbl;
}

bool DataDepSubGraph::SetDynmcBkwrdLwrBounds_(InstCount lastCycle,
                                              InstCount shft,
                                              bool useDistFrmLeaf) {
  InstCount i;
  bool fsbl = true;

  assert(bkwrdLwrBounds_[instCnt_ - 1] == 0);
  dynmcBkwrdLwrBounds_[instCnt_ - 1] = 0; // BLB of the leaf
  dynmcBkwrdLwrBounds_[0] = lastCycle;    // BLB of the root

  for (i = instCnt_ - 2; i > 0; i--) {
    SchedInstruction *inst = insts_[i];

    if (IsInGraph(inst) == false) {
      continue;
    }

    InstCount glblDeadline = inst->GetCrntDeadline() + shft;
    InstCount glblDistFrmLeaf = lastCycle - glblDeadline;
    assert(useDistFrmLeaf || glblDistFrmLeaf > 0);

    dynmcBkwrdLwrBounds_[i] = glblDistFrmLeaf;

    if (useDistFrmLeaf) {
      dynmcBkwrdLwrBounds_[i] = bkwrdLwrBounds_[i];

      if (bkwrdCrtclPaths_[i] > dynmcBkwrdLwrBounds_[i]) {
        dynmcBkwrdLwrBounds_[i] = bkwrdCrtclPaths_[i];
      }

      TightnLwrBound_(DIR_BKWRD, i, dynmcBkwrdLwrBounds_);

      if (glblDistFrmLeaf > dynmcBkwrdLwrBounds_[i]) {
        dynmcBkwrdLwrBounds_[i] = glblDistFrmLeaf;
      }

      assert(dynmcBkwrdLwrBounds_[i] >= bkwrdLwrBounds_[i]);
    }

    InstCount instDstnc = dynmcBkwrdLwrBounds_[i] + dynmcFrwrdLwrBounds_[i];

    if (instDstnc > lastCycle) {
      fsbl = false;
    }
  }

  return fsbl;
}

bool DataDepSubGraph::ChkInstRanges_(InstCount lastCycle) {
  InstCount i;

  for (i = 1; i < (instCnt_ - 1); i++) {
    SchedInstruction *inst = insts_[i];
    if (IsInGraph(inst) && !ChkInstRange_(inst, i, lastCycle))
      return false;
  }

  return true;
}

bool DataDepSubGraph::ChkInstRange_(SchedInstruction *inst, InstCount indx,
                                    InstCount lastCycle) {
  assert(inst != rootInst_ && inst != leafInst_);
  // InstCount glblDeadline = inst->GetCrntDeadline() + shft;
  // assert(glblDeadline == lastCycle-dynmcBkwrdLwrBounds_[indx]);
  InstCount deadline = lastCycle - dynmcBkwrdLwrBounds_[indx];

  if (dynmcFrwrdLwrBounds_[indx] > deadline) {
    return false;
  }

  assert(IsInGraph(inst) == false || inst->IsSchduld() == false ||
         dynmcFrwrdLwrBounds_[indx] == deadline);
  assert(IsInGraph(inst) == false || inst->IsFxd() == false ||
         dynmcFrwrdLwrBounds_[indx] == deadline);
  /*
  if (dynmcFrwrdLwrBounds_[indx] == deadline) {
    if (dynmcRlxdSchdulr_->IsInstFxd(indx) == false) {
      fxdLst_->InsrtElmnt(inst);
      dynmcRlxdSchdulr_->SetInstFxng(indx);
    }
  }
  */
  //return true;
//}
/*
InstCount DataDepSubGraph::CmputMaxReleaseTime_() {
  SchedInstruction *leaf;
  InstCount maxReleaseTime = 0;

  for (leaf = leafInst_->GetFrstPrdcsr(); leaf != NULL;
       leaf = leafInst_->GetNxtPrdcsr()) {
    if (leaf->GetCrntReleaseTime() > maxReleaseTime) {
      maxReleaseTime = leaf->GetCrntReleaseTime();
    }
  }

  return maxReleaseTime;
}

InstCount DataDepSubGraph::CmputMaxDeadline_() {
  SchedInstruction *leaf;
  InstCount maxDeadline = 0;

  for (leaf = leafInst_->GetFrstPrdcsr(); leaf != NULL;
       leaf = leafInst_->GetNxtPrdcsr()) {
    if (leaf->GetCrntDeadline() > maxDeadline) {
      maxDeadline = leaf->GetCrntDeadline();
    }
  }

  return maxDeadline;
}

bool DataDepSubGraph::CmputSmplDynmcLwrBound_(InstCount &dynmcLwrBound,
                                              InstCount trgtLwrBound,
                                              bool &trgtFsbl) {
  InstCount realInstCnt = GetRealInstCnt_();
  dynmcLwrBound = INVALID_VALUE;

  if (realInstCnt <= 2) {
    if (realInstCnt == 1) {
      assert(!insts_[1]->IsSchduld());
      dynmcLwrBound = 1;
    } else {
      assert(realInstCnt == 2);
      dynmcLwrBound = CmputTwoInstDynmcLwrBound_();
    }
  } else if (rootVctr_->GetOneCnt() == realInstCnt) {
    dynmcLwrBound = CmputIndpndntInstDynmcLwrBound_();
  }

  if (dynmcLwrBound != INVALID_VALUE) {
    assert(dynmcLwrBound + 2 >= schedLwrBound_);
    trgtFsbl = dynmcLwrBound <= trgtLwrBound;
    return true;
  }

  return false;
}

InstCount DataDepSubGraph::CmputTwoInstDynmcLwrBound_() {
  SchedInstruction *inst1;
  SchedInstruction *inst2;

  if (insts_[1]->GetCrntDeadline() <= insts_[2]->GetCrntDeadline()) {
    inst1 = insts_[1];
    inst2 = insts_[2];
  } else {
    inst1 = insts_[2];
    inst2 = insts_[1];
  }

  assert(inst1->IsSchduld() == false || inst2->IsSchduld() == false);

  InstCount ltncyLB = 1;
  assert(inst2->IsRcrsvScsr(inst1) == false &&
         inst1->IsRcrsvPrdcsr(inst2) == false);

  if (inst1->IsRcrsvScsr(inst2)) {
    ltncyLB = inst2->GetRltvCrtclPath(DIR_FRWRD, inst1);
    assert(ltncyLB != INVALID_VALUE);
    ltncyLB += 1;
  }

  InstCount rangeLB;
  inst1->GetCrntReleaseTime(); // Meaningless after assignment removed?
  InstCount inst1DL = inst1->GetCrntDeadline();
  InstCount inst2RT = inst2->GetCrntReleaseTime();
#ifdef IS_DEBUG
  InstCount inst2DL = inst1->GetCrntDeadline();
  assert(inst1DL <= inst2DL);
#endif

  // If the sched ranges overlap.
  if (inst1DL >= inst2RT) {
    rangeLB = 1;
  } else {
    assert(inst1DL < inst2RT);
    rangeLB = inst2RT - inst1DL + 1;
  }

  IssueType inst1IssuType = inst1->GetIssueType();
  IssueType inst2IssuType = inst2->GetIssueType();
  int16_t issuRate = (int16_t)machMdl_->GetIssueRate();
  InstCount rsrcLB;

  if (issuRate == 1) {
    rsrcLB = 2;
  } else {
    if (inst1IssuType != inst2IssuType) {
      rsrcLB = 1;
    } else {
      int16_t slotsPerCycle =
          (int16_t)machMdl_->GetSlotsPerCycle(inst1IssuType);
      rsrcLB = slotsPerCycle < 2 ? 2 : 1;
    }
  }

  ltncyLB = std::max(ltncyLB, rangeLB);
  return std::max(ltncyLB, rsrcLB);
}

InstCount DataDepSubGraph::CmputIndpndntInstDynmcLwrBound_() {
  return INVALID_VALUE;
}

void DataDepSubGraph::CmputCrtclPaths_() {
  assert(instCnt_ >= 3);

  if (frwrdCrtclPaths_ == NULL) {
    AllocSttcData_();
  }

  CmputCrtclPaths_(DIR_FRWRD);
  CmputCrtclPaths_(DIR_BKWRD);
  assert(frwrdCrtclPaths_[instCnt_ - 1] == bkwrdCrtclPaths_[0]);
}

InstCount DataDepSubGraph::GetRltvCrtclPath(SchedInstruction *ref,
                                            SchedInstruction *inst,
                                            DIRECTION dir) {
  InstCount rltvCP;
  InstCount indx;
  SchedInstruction *othrInst;

  assert(frwrdCrtclPaths_[instCnt_ - 1] == bkwrdCrtclPaths_[0]);

  if (ref == rootInst_ || inst == rootInst_) {
    othrInst = ref == rootInst_ ? inst : ref;
    assert((ref == rootInst_ && dir == DIR_FRWRD) ||
           (inst == rootInst_ && dir == DIR_BKWRD));
    indx = GetInstIndx(othrInst);
    assert(0 <= indx && indx < instCnt_);
    assert(othrInst == rootInst_ || othrInst == leafInst_ ||
           IsInGraph(othrInst));
    rltvCP = frwrdCrtclPaths_[indx];
    assert(rltvCP != INVALID_VALUE);
  } else if (ref == leafInst_ || inst == leafInst_) {
    othrInst = ref == leafInst_ ? inst : ref;
    assert((ref == leafInst_ && dir == DIR_BKWRD) ||
           (inst == leafInst_ && dir == DIR_FRWRD));
    indx = GetInstIndx(othrInst);
    assert(0 <= indx && indx < instCnt_);
    assert(othrInst == rootInst_ || othrInst == leafInst_ ||
           IsInGraph(othrInst));
    rltvCP = bkwrdCrtclPaths_[indx];
    assert(rltvCP != INVALID_VALUE);
  } else {
    assert(ref != rootInst_ && ref != leafInst_);
    assert(inst != rootInst_ && inst != leafInst_);
    assert(IsInGraph(ref));
    assert(IsInGraph(inst));
    rltvCP = inst->GetRltvCrtclPath(dir, ref);
  }

  return rltvCP;
}

void DataDepSubGraph::CmputCrtclPaths_(DIRECTION dir) {
  SchedInstruction *mainRef = dir == DIR_FRWRD ? rootInst_ : leafInst_;
  SchedInstruction *lastInst = dir == DIR_FRWRD ? leafInst_ : rootInst_;
  BitVector *refVctr_ = dir == DIR_FRWRD ? rootVctr_ : leafVctr_;
  InstCount *crtclPaths =
      dir == DIR_FRWRD ? frwrdCrtclPaths_ : bkwrdCrtclPaths_;
  SchedInstruction *inst;
  InstCount indx;

  crtclPaths[GetInstIndx(mainRef)] = 0;
  assert((mainRef == rootInst_ && GetInstIndx(mainRef) == 0) ||
         (mainRef == leafInst_ && GetInstIndx(mainRef) == instCnt_ - 1));
  InstCount i;

  for (i = 1; i < instCnt_; i++) {
    indx = dir == DIR_FRWRD ? i : instCnt_ - 1 - i;
    inst = GetInstByTplgclOrdr(indx);
    assert(IsInGraph(inst));
    crtclPaths[indx] = 1;

    assert(inst == lastInst || inst->GetNum() != INVALID_VALUE);

    if (inst != lastInst && refVctr_->GetBit(inst->GetNum()))
    // if the instruction is one of the real root/leaf instructions
    // then its lower bound is 1 and we are done
    {
      continue;
    }

    TightnLwrBound_(dir, indx, crtclPaths);
  }
}

InstCount DataDepSubGraph::CmputExtrnlLtncs_(InstCount rejoinCycle,
                                             SchedInstruction *inst) {
  InstCount gapSize = 0;
  InstCount instGapSize = 0;

  // Find the successors of the instructions in this subDAG
  for (InstCount i = 1; i < instCnt_ - 1; i++) {
    SchedInstruction *pred = insts_[i];
    assert(pred->IsSchduld());

    UDT_GLABEL ltncy;
    DependenceType depType;
    for (SchedInstruction *scsr = pred->GetFrstScsr(NULL, &ltncy, &depType);
         scsr != NULL; scsr = pred->GetNxtScsr(NULL, &ltncy, &depType)) {
      assert(scsr != leafInst_);

      if (IsInGraph(scsr) == false && ltncy > 1) {
        gapSize = CmputExtrnlLtncy_(pred, scsr, rejoinCycle,
                                    scsr->GetCrntDeadline(), false, true);

        if (scsr == inst) {
          instGapSize = gapSize;
        }
      }
    }
  }

  PropagateBkwrdLwrBounds_(instCnt_ - 2, 0, bkwrdLwrBounds_, false);
  frwrdLwrBounds_[instCnt_ - 1] = bkwrdLwrBounds_[0];
  return instGapSize;
}

InstCount DataDepSubGraph::CmputExtrnlLtncy_(
    SchedInstruction *pred, SchedInstruction *scsr, InstCount rejoinCycle,
    InstCount scsrCycle, bool isSchduld, bool tightnLwrBound) {
  InstCount gapSize = 0;
  InstCount instGapSize = 0;
  assert(scsr != leafInst_);

  // We need to consider only successors outside the subDAG
  assert(IsInGraph(pred) && IsInGraph(scsr) == false);
  assert(pred->IsRcrsvScsr(scsr) && scsr->IsRcrsvPrdcsr(pred));

  int ltncy = scsr->GetRltvCrtclPath(DIR_FRWRD, pred);
  assert(ltncy >= 0);

  if (ltncy > 1) {
    assert(isSchduld == false || scsr->IsSchduld());
    assert(scsrCycle >= rejoinCycle);
    InstCount scsrDstnc = scsrCycle - rejoinCycle;
    InstCount predDstnc = ltncy - 1 - scsrDstnc;
    InstCount predIndx = GetInstIndx(pred);
    instGapSize = predDstnc + 1 - bkwrdCrtclPaths_[predIndx];
    gapSize = predDstnc + 1 - bkwrdLwrBounds_[predIndx];

    if (gapSize > 0 && tightnLwrBound) {
      bkwrdLwrBounds_[predIndx] = predDstnc + 1;
    }

    // Logger::Info("Dep %d->%d: "
    //             "latency =%d, rejCycle=%d, predDstnc=%d, scsrDstnc=%d",
    //             pred->GetNum(),
    //             scsr->GetNum(),
    //             ltncy,
    //             rejoinCycle,
    //             predDstnc,
    //             scsrDstnc);
  }

  return instGapSize;
}

InstCount DataDepSubGraph::GetDistFrmLeaf(SchedInstruction *inst) {
  InstCount indx = GetInstIndx(inst);
  assert(0 <= indx && indx <= instCnt_);
  InstCount distFrmLeaf =
      std::max(bkwrdLwrBounds_[indx], bkwrdCrtclPaths_[indx]);
  return distFrmLeaf;
}
*/

InstSchedule::InstSchedule(MachineModel *machMdl, DataDepGraph *dataDepGraph,
                           bool vrfy) {
  machMdl_ = machMdl;
  issuRate_ = machMdl->GetIssueRate();
  totInstCnt_ = dataDepGraph->GetInstCnt();
  schedUprBound_ = dataDepGraph->GetAbslutSchedUprBound();
  totSlotCnt_ = schedUprBound_ * issuRate_;
  vrfy_ = vrfy;

  instInSlot_ = new InstCount[totSlotCnt_];
  slotForInst_ = new InstCount[totInstCnt_];
  spillCosts_ = new InstCount[totInstCnt_];
  peakRegPressures_ = new InstCount[machMdl->GetRegTypeCnt()];
  dev_instInSlot_ = NULL;
  dev_slotForInst_ = NULL;
  dev_spillCosts_ = NULL;
  dev_peakRegPressures_ = NULL;

  InstCount i;

  for (i = 0; i < totInstCnt_; i++) {
    slotForInst_[i] = SCHD_UNSCHDULD;
    spillCosts_[i] = 0;
  }

  for (i = 0; i < totSlotCnt_; i++) {
    instInSlot_[i] = SCHD_UNSCHDULD;
  }

  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  maxSchduldInstCnt_ = 0;
  maxInstNumSchduld_ = -1;
  iterSlotNum_ = 0;
  cost_ = INVALID_VALUE;
  execCost_ = INVALID_VALUE;
  totSpillCost_ = 0;
  cnflctCnt_ = 0;
  spillCnddtCnt_ = 0;
  totalStalls_ = 0;
  unnecessaryStalls_ = 0;
  isZeroPerp_ = false;
}

InstSchedule::InstSchedule() {
  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  maxSchduldInstCnt_ = 0;
  maxInstNumSchduld_ = -1;
  iterSlotNum_ = 0;
  cost_ = INVALID_VALUE;
  execCost_ = INVALID_VALUE;
  totSpillCost_ = 0;
  cnflctCnt_ = 0;
  spillCnddtCnt_ = 0;
  totalStalls_ = 0;
  unnecessaryStalls_ = 0;
  isZeroPerp_ = false;
}

InstSchedule::~InstSchedule() {
  delete[] instInSlot_;
  delete[] slotForInst_;
  delete[] spillCosts_;
  delete[] peakRegPressures_;
}

bool InstSchedule::operator==(InstSchedule &b) const {
  if (b.totSlotCnt_ != totSlotCnt_)
    return false;
  if (b.crntSlotNum_ != crntSlotNum_)
    return false;
  for (InstCount i = 0; i < crntSlotNum_; i++) {
    if (b.instInSlot_[i] != instInSlot_[i])
      return false;
  }
  return true;
}

__host__ __device__
bool InstSchedule::AppendInst(InstCount instNum) {
#ifdef __HIP_DEVICE_COMPILE__
  assert(crntSlotNum_ < totSlotCnt_);
  dev_instInSlot_[crntSlotNum_] = instNum;

  if (vrfy_)
    if (instNum > maxInstNumSchduld_) {
      maxInstNumSchduld_ = instNum;
    }

  if (instNum != SCHD_STALL) {
    assert(instNum >= 0 && instNum < totInstCnt_);
    dev_slotForInst_[instNum] = crntSlotNum_;
    schduldInstCnt_++;
#ifdef IS_DEBUG_SCHED2

    if (schduldInstCnt_ > maxSchduldInstCnt_) {
      maxSchduldInstCnt_ = schduldInstCnt_;
      printf("INFO: Maximum # of Instructions Scheduled: %d\n",
                   maxSchduldInstCnt_);
    }

#endif
  }

#ifdef IS_DEBUG_SCHED2
  printf("INFO: Instructions Scheduled: %d\n", schduldInstCnt_);
#endif
  crntSlotNum_++;
  return true;

#else
  assert(crntSlotNum_ < totSlotCnt_);
  instInSlot_[crntSlotNum_] = instNum;

  if (vrfy_)
    if (instNum > maxInstNumSchduld_) {
      maxInstNumSchduld_ = instNum;
    }

  if (instNum != SCHD_STALL) {
    assert(instNum >= 0 && instNum < totInstCnt_);
    slotForInst_[instNum] = crntSlotNum_;
    schduldInstCnt_++;
#ifdef IS_DEBUG_SCHED2

    if (schduldInstCnt_ > maxSchduldInstCnt_) {
      maxSchduldInstCnt_ = schduldInstCnt_;
      Logger::Info("Maximum # of Instructions Scheduled: %d",
                   maxSchduldInstCnt_);
    }

#endif
  }

#ifdef IS_DEBUG_SCHED2
  Logger::Info("Instructions Scheduled: %d", schduldInstCnt_);
#endif
  crntSlotNum_++;
  return true;
#endif
}

bool InstSchedule::RemoveLastInst() {
  if (crntSlotNum_ == 0) { // an empty schedule
    return false;
  }

  crntSlotNum_--;
  InstCount instNum = instInSlot_[crntSlotNum_];
  instInSlot_[crntSlotNum_] = INVALID_VALUE;

  if (instNum != SCHD_STALL) {
    slotForInst_[instNum] = INVALID_VALUE;
    schduldInstCnt_--;
  }

#ifdef IS_DEBUG_SCHED2
  Logger::Info("Instructions Scheduled: %d", schduldInstCnt_);
#endif
  return true;
}

__host__ __device__
void InstSchedule::ResetInstIter() { iterSlotNum_ = 0; }

__host__ __device__
InstCount InstSchedule::GetFrstInst(InstCount &cycleNum, InstCount &slotNum) {
  iterSlotNum_ = 0;
  return GetNxtInst(cycleNum, slotNum);
}

__host__ __device__
InstCount InstSchedule::GetNxtInst(InstCount &cycleNum, InstCount &slotNum) {
  InstCount instNum;

  if (iterSlotNum_ == crntSlotNum_) {
    return INVALID_VALUE;
  }
  do {
#ifdef __HIP_DEVICE_COMPILE__
    instNum = dev_instInSlot_[iterSlotNum_];
#else
    instNum = instInSlot_[iterSlotNum_];
#endif
    iterSlotNum_++;
  } while (instNum == SCHD_STALL);

  assert(instNum != SCHD_STALL);
  GetCycleAndSlotNums_(iterSlotNum_ - 1, cycleNum, slotNum);
  return instNum;
}

__device__
InstCount InstSchedule::GetPrevInstNum(InstCount instNum) {
  int slot = dev_slotForInst_[instNum];
  if (slot > 0) { // not the first inst in schedule
    // iterate through instInSlot_ backwards and return first instNum reached
    for (int i = slot - 1; i >= 0; i--)
      if (dev_instInSlot_[i] != SCHD_STALL)
        return dev_instInSlot_[i];
  }
  return INVALID_VALUE;  
}

__host__ __device__
void InstSchedule::Reset() {
  InstCount i;
#ifdef __HIP_DEVICE_COMPILE__
  if (vrfy_) {
    for (i = 0; i <= maxInstNumSchduld_; i++) {
      dev_slotForInst_[i] = SCHD_UNSCHDULD;
    }

    for (i = 0; i <= crntSlotNum_; i++) {
      dev_instInSlot_[i] = SCHD_UNSCHDULD;
    }
  }
#else
  if (vrfy_) {
    for (i = 0; i < maxInstNumSchduld_; i++) {
      slotForInst_[i] = SCHD_UNSCHDULD;
    }

    for (i = 0; i <= crntSlotNum_; i++) {
      instInSlot_[i] = SCHD_UNSCHDULD;
    }
  }
#endif

  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  maxSchduldInstCnt_ = 0;
  maxInstNumSchduld_ = -1;
  cost_ = INVALID_VALUE;
}

__host__ __device__
void InstSchedule::Copy(InstSchedule *src) {
  Reset();
#ifdef __HIP_DEVICE_COMPILE__
  InstCount i;
  for (i = 0; i < totSlotCnt_ && src->dev_instInSlot_[i] != SCHD_UNSCHDULD; i++) {
    AppendInst(src->dev_instInSlot_[i]);
  }

  SetSpillCosts(src->dev_spillCosts_);
  SetPeakRegPressures(src->dev_peakRegPressures_);
#else
  InstCount i;
  for (i = 0; i < totSlotCnt_ && src->instInSlot_[i] != SCHD_UNSCHDULD; i++) {
    AppendInst(src->instInSlot_[i]);
  }

  SetSpillCosts(src->spillCosts_);
  SetPeakRegPressures(src->peakRegPressures_);
#endif
  cost_ = src->cost_;
  execCost_ = src->execCost_;
  NormSpillCost = src->NormSpillCost;
  spillCost_ = src->spillCost_;
  totalStalls_ = src->totalStalls_;
  unnecessaryStalls_ = src->unnecessaryStalls_;
  isZeroPerp_ = src->isZeroPerp_;
}

__host__ __device__
void InstSchedule::SetSpillCosts(InstCount spillCosts[]) {
#ifdef __HIP_DEVICE_COMPILE__
  totSpillCost_ = 0; 
  for (InstCount i = 0; i < totInstCnt_; i++) {
    dev_spillCosts_[i] = spillCosts[i];
    totSpillCost_ += spillCosts[i];
  }
#else
  totSpillCost_ = 0;
  for (InstCount i = 0; i < totInstCnt_; i++) {
    spillCosts_[i] = spillCosts[i];
    totSpillCost_ += spillCosts[i];
  }
#endif
}

__device__
void InstSchedule::Dev_SetSpillCosts(InstCount **spillCosts) {
  totSpillCost_ = 0;
  for (InstCount i = 0; i < totInstCnt_; i++) {
    dev_spillCosts_[i] = spillCosts[i][GLOBALTID];
    totSpillCost_ += spillCosts[i][GLOBALTID];
  }  
}

__host__ __device__
void InstSchedule::SetPeakRegPressures(InstCount peakRegPressures[]) {
#ifdef __HIP_DEVICE_COMPILE__
  for (InstCount i = 0; i < dev_machMdl_->GetRegTypeCnt(); i++) {
    dev_peakRegPressures_[i] = peakRegPressures[i];
  }
#else
  for (InstCount i = 0; i < machMdl_->GetRegTypeCnt(); i++) {
    peakRegPressures_[i] = peakRegPressures[i];
  }
#endif
}

__device__
void InstSchedule::Dev_SetPeakRegPressures(InstCount **peakRegPressures) {
  for (InstCount i = 0; i < dev_machMdl_->GetRegTypeCnt(); i++) {
    dev_peakRegPressures_[i] = peakRegPressures[i][GLOBALTID];
  }
}

InstCount
InstSchedule::GetPeakRegPressures(const InstCount *&regPressures) const {
  regPressures = peakRegPressures_;
  return machMdl_->GetRegTypeCnt();
}

__host__ __device__
InstCount InstSchedule::GetSpillCost(InstCount stepNum) {
  assert(stepNum >= 0 && stepNum < totInstCnt_);
#ifdef __HIP_DEVICE_COMPILE__
  return dev_spillCosts_[stepNum];
#else
  return spillCosts_[stepNum];
#endif
}

InstCount InstSchedule::GetTotSpillCost() { return totSpillCost_; }

int InstSchedule::GetConflictCount() { return cnflctCnt_; }

void InstSchedule::SetConflictCount(int cnflctCnt) { cnflctCnt_ = cnflctCnt; }

int InstSchedule::GetSpillCandidateCount() { return spillCnddtCnt_; }

void InstSchedule::SetSpillCandidateCount(int spillCnddtCnt) {
  spillCnddtCnt_ = spillCnddtCnt;
}

// TODO(austin) move logger print of schedule to different function
__host__ __device__
void InstSchedule::Print() {
  InstCount slotInCycle = 0;
  InstCount cycleNum = 0;
  InstCount i;
  for (i = 0; i < crntSlotNum_; i++) {
    if (slotInCycle == 0)
#ifdef __HIP_DEVICE_COMPILE__
      printf("Cycle# %d : %d\n", cycleNum, dev_instInSlot_[i]);
#else
      printf("Cycle# %d : %d\n", cycleNum, instInSlot_[i]);
#endif

    slotInCycle++;
    if (slotInCycle == issuRate_) {
      slotInCycle = 0;
      cycleNum++;
    }
  }
}

#if defined(IS_DEBUG_PEAK_PRESSURE) || defined(IS_DEBUG_OPTSCHED_PRESSURES)
void InstSchedule::PrintRegPressures() const {
  Logger::Info("OptSched max reg pressures:");
  InstCount i;
  for (i = 0; i < machMdl_->GetRegTypeCnt(); i++) {
    Logger::Info("%s: Peak %d Limit %d", machMdl_->GetRegTypeName(i),
                 peakRegPressures_[i], machMdl_->GetPhysRegCnt(i));
  }
}
#endif

void InstSchedule::PrintInstList(FILE *file, DataDepGraph *dataDepGraph,
                                 const char *label) const {
  fprintf(file, "\n%s Instruction Order:", label);

  for (InstCount i = 0; i < crntSlotNum_; i++) {
    if (instInSlot_[i] != SCHD_STALL) {
      InstCount instNum = instInSlot_[i];
      SchedInstruction *inst = dataDepGraph->GetInstByIndx(instNum);

      if (inst != dataDepGraph->GetRootInst()) {
        fprintf(file, "\nInst. #%d: %s", instNum, inst->GetName());
      }
    }
  }

#ifdef IS_DEBUG
  fflush(file);
#endif
}

bool InstSchedule::Verify(MachineModel *machMdl, DataDepGraph *dataDepGraph) {
  if (schduldInstCnt_ < totInstCnt_) {
    Logger::Error("Invalid schedule: too few scheduled instructions: %d of %d",
                  schduldInstCnt_, totInstCnt_);
    return false;
  }

  if (schduldInstCnt_ > totInstCnt_) {
    Logger::Error("Invalid schedule: Too many scheduled instructions: %d of %d",
                  schduldInstCnt_, totInstCnt_);
    return false;
  }

  for (InstCount i = 0; i < totInstCnt_; i++) {
    if (slotForInst_[i] == SCHD_UNSCHDULD) {
      Logger::Error("Invalid schedule: inst #%d unscheduled", i);
      return false;
    }
  }

  if (!VerifySlots_(machMdl, dataDepGraph))
    return false;
  if (!VerifyDataDeps_(dataDepGraph))
    return false;

#if defined(IS_DEBUG_PEAK_PRESSURE) || defined(IS_DEBUG_OPTSCHED_PRESSURES)
  PrintRegPressures();
#endif

#ifdef IS_DEBUG_PRINT_SCHEDULE
  Print(std::cout, "debug");
#endif

  Logger::Info("Schedule verified successfully");

  return true;
}

bool InstSchedule::VerifySlots_(MachineModel *machMdl,
                                DataDepGraph *dataDepGraph) {
  InstCount i;
  int slotsPerCycle[MAX_ISSUTYPE_CNT];
  int filledSlotsPerCycle[MAX_ISSUTYPE_CNT];
  int issuTypeCnt;
  InstCount cycleNum, localSlotNum, globalSlotNum;
  InstCount schduldInstCnt = 0;

  issuTypeCnt = machMdl->GetSlotsPerCycle(slotsPerCycle);

  for (cycleNum = 0, globalSlotNum = 0; globalSlotNum < crntSlotNum_;
       cycleNum++) {
    for (i = 0; i < issuTypeCnt; i++) {
      filledSlotsPerCycle[i] = 0;
    }

    for (localSlotNum = 0;
         localSlotNum < issuRate_ && globalSlotNum < crntSlotNum_;
         localSlotNum++, globalSlotNum++) {
      InstCount instNum = instInSlot_[globalSlotNum];

      if (instNum == SCHD_UNSCHDULD) {
        Logger::Error("Invalid schedule: slot #%d unscheduled", globalSlotNum);
        return false;
      }

      if (instNum != SCHD_STALL) {
        schduldInstCnt++;

        if (instNum >= totInstCnt_) {
          Logger::Error("Invalid schedule: Too large inst. number #%d",
                        instNum);
          return false;
        }

        SchedInstruction *inst = dataDepGraph->GetInstByIndx(instNum);
        IssueType issuType = inst->GetIssueType();

        if (issuType >= issuTypeCnt) {
          Logger::Error("Invalid schedule: Invalid issue type %d for inst #%d",
                        issuType, instNum);
          return false;
        }

        filledSlotsPerCycle[issuType]++;

        if (filledSlotsPerCycle[issuType] > slotsPerCycle[issuType]) {
          Logger::Error("Invalid schedule: Too many insts. of issue type #%d",
                        issuType);
          return false;
        }
      }
    }
  }

  if (schduldInstCnt != schduldInstCnt_) {
    Logger::Error("Invalid schedule: Invalid # of scheduled insts");
    return false;
  }

  return true;
}

bool InstSchedule::VerifyDataDeps_(DataDepGraph *dataDepGraph) {
  for (InstCount i = 0; i < totInstCnt_; i++) {
    if (slotForInst_[i] == SCHD_UNSCHDULD) {
      Logger::Error("Invalid schedule: inst #%d unscheduled", i);
      return false;
    }

    SchedInstruction *inst = dataDepGraph->GetInstByIndx(i);
    InstCount instCycle = GetSchedCycle(inst);

    UDT_GLABEL ltncy;
    DependenceType depType;
    bool IsArtificial;
    for (SchedInstruction *scsr = 
            inst->GetFrstScsr(NULL, &ltncy, &depType, NULL, &IsArtificial);
         scsr != NULL; 
         scsr = inst->GetNxtScsr(NULL, &ltncy, &depType, NULL, &IsArtificial)) {
      // Artificial edges are not required for the schedule to be correct
      if (IsArtificial)
        continue;

      InstCount scsrCycle = GetSchedCycle(scsr);
      if (scsrCycle < (instCycle + ltncy)) {
        Logger::Error("Invalid schedule: Latency from %d to %d not satisfied",
                      i, scsr->GetNum());
        return false;
      }
    }
  }

  return true;
}

void InstSchedule::PrintClassData() {
  Logger::Info("issuRate_=%d, totInstCnt_=%d, totSlotCnt_=%d", issuRate_,
               totInstCnt_, totSlotCnt_);
  Logger::Info("schduldInstCnt_=%d, maxSchduldInstCnt_=%d, "
               "maxInstNumSchduld_=%d",
               schduldInstCnt_, maxSchduldInstCnt_, maxInstNumSchduld_);
}

__host__ __device__
InstCount InstSchedule::GetCrntLngth() {
  return (crntSlotNum_ + issuRate_ - 1) / issuRate_;
}

InstCount InstSchedule::GetSchedCycle(SchedInstruction *inst) {
  assert(inst != NULL);
  return GetSchedCycle(inst->GetNum());
}

__host__ __device__
InstCount InstSchedule::GetSchedCycle(InstCount instNum) {
  assert(instNum < totInstCnt_);
  InstCount slotNum = slotForInst_[instNum];
  
  InstCount cycleNum;

  if (slotNum == SCHD_UNSCHDULD) {
    cycleNum = SCHD_UNSCHDULD;
  } else {
    cycleNum = slotNum / issuRate_;
  }

  return cycleNum;
}

__host__ __device__
bool InstSchedule::IsComplete() { return (schduldInstCnt_ == totInstCnt_); }

bool InstSchedule::AppendInst(SchedInstruction *inst) {
  InstCount instNum = inst == NULL ? SCHD_STALL : inst->GetNum();
  return AppendInst(instNum);
}

__host__ __device__
void InstSchedule::GetCycleAndSlotNums_(InstCount globSlotNum,
                                        InstCount &cycleNum,
                                        InstCount &slotNum) {
  cycleNum = globSlotNum / issuRate_;
  slotNum = globSlotNum % issuRate_;
}

__host__ __device__
void InstSchedule::SetCost(InstCount cost) { cost_ = cost; }

__host__ __device__
InstCount InstSchedule::GetCost() const { return cost_; }

__host__ __device__
void InstSchedule::SetExecCost(InstCount cost) { execCost_ = cost; }

__host__ __device__
InstCount InstSchedule::GetExecCost() const { return execCost_; }

__host__ __device__
void InstSchedule::SetSpillCost(InstCount cost) { spillCost_ = cost; }

__host__ __device__
InstCount InstSchedule::GetSpillCost() const { return spillCost_; }

// NOTE: ACO needs statically normalized costs.  These are statically normalized
// costs that don't use the dynamic SLIL lower bound.
__host__ __device__
void InstSchedule::SetNormSpillCost(InstCount cost) { NormSpillCost = cost; }

__host__ __device__
InstCount InstSchedule::GetNormSpillCost() const { return NormSpillCost; }

__host__ __device__
void InstSchedule::SetExtraSpillCost(SPILL_COST_FUNCTION Fn, InstCount cost) {
  storedSC[Fn] = cost;
}

__host__ __device__
InstCount InstSchedule::GetExtraSpillCost(SPILL_COST_FUNCTION Fn) const {
  return storedSC[Fn];
}

void InstSchedule::AllocateOnDevice(MachineModel *dev_machMdl) {
  // Alloc instInSlot_ on device
  size_t memSize = totSlotCnt_ * sizeof(InstCount);
  gpuErrchk(hipMalloc((void**)&dev_instInSlot_, memSize));
  // Alloc slotForInst_ on device
  memSize = totInstCnt_ * sizeof(InstCount);
  gpuErrchk(hipMalloc((void**)&dev_slotForInst_, memSize));
  // Alloc spillCosts_ on device
  gpuErrchk(hipMalloc((void**)&dev_spillCosts_, memSize));
  // Alloc peakRegPressures_ on device
  memSize = machMdl_->GetRegTypeCnt() * sizeof(InstCount);
  gpuErrchk(hipMalloc((void**)&dev_peakRegPressures_, memSize));
  dev_machMdl_ = dev_machMdl;
}

void InstSchedule::SetDevArrayPointers(MachineModel *dev_machMdl, 
                                       InstCount *dev_temp) {
  dev_machMdl_ = dev_machMdl;
  int index = 0;
  dev_instInSlot_ = &dev_temp[index];
  // Increment the index past the needed number of slots for dev_instInSlot_
  index += totSlotCnt_;
  dev_slotForInst_ = &dev_temp[index];
  // Increment the index past needed num of slots for dev_slotForInst_
  index += totInstCnt_;
  dev_spillCosts_ = &dev_temp[index];
  // Increment the index past needed num of slots for dev_spillCosts_
  index += totInstCnt_;
  dev_peakRegPressures_ = &dev_temp[index];
}

size_t InstSchedule::GetSizeOfDevArrays() {
  return totSlotCnt_ + totInstCnt_ * 2 + machMdl_->GetRegTypeCnt();
}

void InstSchedule::CopyArraysToDevice() {
  size_t memSize;
  // Copy instInSlot to device
  memSize = totSlotCnt_ * sizeof(InstCount);
  gpuErrchk(hipMemcpy(dev_instInSlot_, instInSlot_, memSize,
                       hipMemcpyHostToDevice));
  // Copy slotForInst_ to device
  memSize = totInstCnt_ * sizeof(InstCount);
  gpuErrchk(hipMemcpy(dev_slotForInst_, slotForInst_, memSize,
                       hipMemcpyHostToDevice));
  // Copy spillCosts to device
  gpuErrchk(hipMemcpy(dev_spillCosts_, spillCosts_, memSize,
                       hipMemcpyHostToDevice));
  // Copy peakRegPressures to device
  memSize = machMdl_->GetRegTypeCnt() * sizeof(InstCount);
  gpuErrchk(hipMemcpy(dev_peakRegPressures_, peakRegPressures_, memSize,
                       hipMemcpyHostToDevice));
}

void InstSchedule::CopyArraysToHost() {
  size_t memSize;
  // Copy instInSlot to host
  memSize = totSlotCnt_ * sizeof(InstCount);
  gpuErrchk(hipMemcpy(instInSlot_, dev_instInSlot_, memSize,
		       hipMemcpyDeviceToHost));
  // Copy slotForInst_ to host
  memSize = totInstCnt_ * sizeof(InstCount);
  gpuErrchk(hipMemcpy(slotForInst_, dev_slotForInst_, memSize, 
		       hipMemcpyDeviceToHost));
  // Copy spillCosts to host
  gpuErrchk(hipMemcpy(spillCosts_, dev_spillCosts_, memSize, 
		       hipMemcpyDeviceToHost));
  // Copy peakRegPressures to host
  memSize = machMdl_->GetRegTypeCnt() * sizeof(InstCount);
  gpuErrchk(hipMemcpy(peakRegPressures_, dev_peakRegPressures_, memSize,
		       hipMemcpyDeviceToHost));
}

void InstSchedule::FreeDeviceArrays() {
  hipFree(dev_instInSlot_);
  hipFree(dev_slotForInst_);
  hipFree(dev_spillCosts_);
  hipFree(dev_peakRegPressures_);  
}

__device__
void InstSchedule::Initialize() {
  InstCount i;

  for (i = 0; i < totInstCnt_; i++) {
    dev_slotForInst_[i] = SCHD_UNSCHDULD;
    dev_spillCosts_[i] = 0;
  }

  for (i = 0; i < totSlotCnt_; i++) {
    dev_instInSlot_[i] = SCHD_UNSCHDULD;
  }

  schduldInstCnt_ = 0;
  crntSlotNum_ = 0;
  maxSchduldInstCnt_ = 0;
  maxInstNumSchduld_ = -1;
  iterSlotNum_ = 0;
  cost_ = INVALID_VALUE;
  execCost_ = INVALID_VALUE;
  totSpillCost_ = 0;
  cnflctCnt_ = 0;
  spillCnddtCnt_ = 0;
}

/*******************************************************************************
 * Previously inlined functions
 ******************************************************************************/

DEP_GRAPH_TYPE DataDepStruct::GetType() { return type_; }

__host__ __device__
InstCount DataDepStruct::GetInstCnt() { return instCnt_; }

InstCount DataDepStruct::GetOrgnlInstCnt() { return instCnt_; }

__host__ __device__
InstCount DataDepStruct::GetAbslutSchedUprBound() { return schedUprBound_; }

void DataDepStruct::SetAbslutSchedUprBound(InstCount bound) {
  schedUprBound_ = bound;
}

void DataDepStruct::GetLwrBounds(InstCount *&frwrdLwrBounds,
                                 InstCount *&bkwrdLwrBounds) {
  frwrdLwrBounds = frwrdLwrBounds_;
  bkwrdLwrBounds = bkwrdLwrBounds_;
  assert(frwrdLwrBounds != NULL);
  assert(bkwrdLwrBounds != NULL);
}

__host__ __device__
SchedInstruction *DataDepGraph::GetRootInst() {
  return (SchedInstruction *)root_;
}

__host__ __device__
SchedInstruction *DataDepGraph::GetLeafInst() {
  return (SchedInstruction *)leaf_;
}

__host__
void DataDepGraph::CmputCrtclPathsFrmRoot_() {
  InstCount i;

  // Visit the nodes in topological order
  for (i = 0; i < instCnt_; i++) {
    ((SchedInstruction *)(tplgclOrdr_[i]))->CmputCrtclPathFrmRoot();
  }
}

__host__
void DataDepGraph::CmputCrtclPathsFrmLeaf_() {
  InstCount i;

  // Visit the nodes in reverse topological order
  for (i = instCnt_ - 1; i >= 0; i--) {
    ((SchedInstruction *)(tplgclOrdr_[i]))->CmputCrtclPathFrmLeaf();
  }
}

__host__
void DataDepGraph::CmputCrtclPaths_() {
  CmputCrtclPathsFrmRoot_();
  CmputCrtclPathsFrmLeaf_();
  //  crtclPathsCmputd_=true;
}

int DataDepGraph::GetBscBlkCnt() { return bscBlkCnt_; }

__host__ __device__
SchedInstruction *DataDepGraph::GetInstByIndx(InstCount instIndx) {
  if (instIndx >= 0 && instIndx < instCnt_)
    return &insts_[instIndx];
  else
    return NULL;
}

SchedInstruction *DataDepGraph::GetInstByTplgclOrdr(InstCount ordr) {
  assert(ordr >= 0 && ordr < instCnt_);
  return (SchedInstruction *)(tplgclOrdr_[ordr]);
}

SchedInstruction *DataDepGraph::GetInstByRvrsTplgclOrdr(InstCount ordr) {
  assert(ordr >= 0 && ordr < instCnt_);
  return (SchedInstruction *)(tplgclOrdr_[instCnt_ - 1 - ordr]);
}

void DataDepGraph::GetCrntLwrBounds(DIRECTION dir, InstCount crntLwrBounds[]) {
  InstCount i;

  for (i = 0; i < instCnt_; i++) {
    crntLwrBounds[i] = insts_[i].GetCrntLwrBound(dir);
  }
}

void DataDepGraph::SetCrntLwrBounds(DIRECTION dir, InstCount crntLwrBounds[]) {
  InstCount i;

  for (i = 0; i < instCnt_; i++) {
    insts_[i].SetCrntLwrBound(dir, crntLwrBounds[i]);
  }
}

__host__ __device__
UDT_GLABEL DataDepGraph::GetMaxLtncy() { return maxLtncy_; }

__host__ __device__
UDT_GLABEL DataDepGraph::GetMaxLtncySum() { return maxLtncySum_; }

InstCount DataDepGraph::GetSchedLwrBound() { return schedLwrBound_; }

const char *DataDepGraph::GetDagID() const { return dagID_; }

float DataDepGraph::GetWeight() const { return weight_; }

void DataDepGraph::GetFileSchedBounds(InstCount &lwrBound,
                                      InstCount &uprBound) const {
  lwrBound = fileSchedLwrBound_;
  uprBound = fileSchedUprBound_;
}

InstCount DataDepGraph::GetFileSchedTrgtUprBound() {
  return fileSchedTrgtUprBound_;
}

void DataDepGraph::GetFinalBounds(InstCount &lwrBound, InstCount &uprBound) {
  lwrBound = finalLwrBound_;
  uprBound = finalUprBound_;
}

void DataDepGraph::SetFinalBounds(InstCount lwrBound, InstCount uprBound) {
  finalLwrBound_ = lwrBound;
  finalUprBound_ = uprBound;
}

bool DataDepGraph::IsInGraph(SchedInstruction *) { return true; }

InstCount DataDepGraph::GetInstIndx(SchedInstruction *inst) {
  assert(inst != NULL);
  InstCount instNum = inst->GetNum();
  assert(0 <= instNum && instNum < instCnt_);
  return instNum;
}

void DataDepGraph::SetCrntFrwrdLwrBound(SchedInstruction *inst) {
  InstCount bound = inst->GetCrntLwrBound(DIR_FRWRD);
  frwrdLwrBounds_[inst->GetNum()] = bound;
}

InstCount DataDepGraph::GetDistFrmLeaf(SchedInstruction *inst) {
  return inst->GetLwrBound(DIR_BKWRD);
}

void DataDepGraph::SetPrblmtc() { isPrblmtc_ = true; }

bool DataDepGraph::IsPrblmtc() { return isPrblmtc_; }

bool DataDepGraph::DoesFeedUser(SchedInstruction *inst) {
#ifdef IS_DEBUG_RP_ONLY
  Logger::Info("Testing inst %d", inst->GetNum());
#endif
  ArrayList<InstCount> *rcrsvSuccs = inst->GetRcrsvNghbrLst(DIR_FRWRD);
  for (InstCount succNum = rcrsvSuccs->GetFrstElmnt(); succNum != END;
       succNum = rcrsvSuccs->GetNxtElmnt()) {
    SchedInstruction *succInst = &insts_[succNum];

    int curInstAdjUseCnt = succInst->GetAdjustedUseCnt();
    // Ignore successor instructions that does not close live intervals
    if (curInstAdjUseCnt == 0)
      continue;
    // Ignore instructions that open more live intervals than
    // it closes because it will increase register pressure instead.
    else if (curInstAdjUseCnt < succInst->GetDefCnt())
      continue;

    // If there is a successor instruction that decreases live intervals
    // or one that does not increase live intervals, then return true.
    return true;
  }
// Return false if there is no recursive successor of inst
// that uses a live register.
#ifdef IS_DEBUG_RP_ONLY
  Logger::Info("No recursive use for inst %d", inst->GetNum());
#endif
  return false;
}

int DataDepGraph::GetFileCostUprBound() { return fileCostUprBound_; }

void DataDepGraph::CopyPointersToDevice(DataDepGraph *dev_DDG, int numThreads) {
  // use to hold size of array
  size_t memSize;
  // Copy instCntPerType_ to device
  InstCount *dev_instCntPerType;
  memSize = sizeof(InstCount) * instTypeCnt_;
  gpuErrchk(hipMalloc(&dev_instCntPerType, memSize));
  gpuErrchk(hipMemcpy(dev_instCntPerType, instCntPerType_, memSize,
		       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_DDG->instCntPerType_, &dev_instCntPerType,
		       sizeof(InstCount *), hipMemcpyHostToDevice));
  // Copy instCntPerIssuType_
  InstCount *dev_instCntPerIssuType;
  memSize = sizeof(InstCount) * issuTypeCnt_;
  gpuErrchk(hipMalloc(&dev_instCntPerIssuType, memSize));
  gpuErrchk(hipMemcpy(dev_instCntPerIssuType, instCntPerIssuType_, memSize,
                       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_DDG->instCntPerIssuType_, &dev_instCntPerIssuType,
                       sizeof(InstCount *), hipMemcpyHostToDevice));
  // Copy frwrdLwrBounds_ to device
  InstCount *dev_frwrdLwrBounds;
  memSize = sizeof(InstCount) * instCnt_;
  gpuErrchk(hipMalloc(&dev_frwrdLwrBounds, memSize));
  gpuErrchk(hipMemcpy(dev_frwrdLwrBounds, frwrdLwrBounds_, memSize,
                       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_DDG->frwrdLwrBounds_, &dev_frwrdLwrBounds,
                       sizeof(InstCount *), hipMemcpyHostToDevice));
  // Copy bkwardLwrBounds_ to device
  InstCount *dev_bkwrdLwrBounds;
  memSize = sizeof(InstCount) * instCnt_;
  gpuErrchk(hipMalloc(&dev_bkwrdLwrBounds, memSize));
  gpuErrchk(hipMemcpy(dev_bkwrdLwrBounds, bkwrdLwrBounds_, memSize,
                       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_DDG->bkwrdLwrBounds_, &dev_bkwrdLwrBounds,
                       sizeof(InstCount *), hipMemcpyHostToDevice));
  // Copy insts_ to device
  SchedInstruction *dev_insts;
  memSize = sizeof(SchedInstruction) * instCnt_;
  gpuErrchk(hipMallocManaged(&dev_insts, memSize));
  gpuErrchk(hipMemcpy(dev_insts, insts_, memSize,
	               hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_DDG->insts_, &dev_insts, 
		       sizeof(SchedInstruction *), 
		       hipMemcpyHostToDevice));
  // update values of root_ and leaf_ on device
  SchedInstruction *dev_root = &dev_insts[root_->GetNum()];
  // set dev_IsRoot to be used on device to check if it is the root
  dev_root->SetDevIsRoot();
  memSize = sizeof(SchedInstruction *);
  gpuErrchk(hipMemcpy(&dev_DDG->root_, &dev_root, memSize,
	               hipMemcpyHostToDevice));
  SchedInstruction *dev_leaf = &dev_insts[leaf_->GetNum()];
  gpuErrchk(hipMemcpy(&dev_DDG->leaf_, &dev_leaf, memSize,
	               hipMemcpyHostToDevice));
  // Copy RegFiles
  RegisterFile *dev_regFiles;
  memSize = sizeof(RegisterFile) * machMdl_->GetRegTypeCnt();
  gpuErrchk(hipMallocManaged(&dev_regFiles, memSize));
  gpuErrchk(hipMemcpy(dev_regFiles, RegFiles, memSize,
		       hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(&dev_DDG->RegFiles, &dev_regFiles, 
		       sizeof(RegisterFile *), hipMemcpyHostToDevice));
  // Also copy each RegFile's pointers
  for (InstCount i = 0; i < machMdl_->GetRegTypeCnt(); i++)
    RegFiles[i].CopyPointersToDevice(&dev_DDG->RegFiles[i]);
  

  Logger::Info("Copying SchedInstructions to device");
  // count the number of elements in predecessor/successor lists
  // used to malloc the large elements array
  int lngthScsrElmnts = 0;
  int lengthLatencies = 0;
  for (InstCount i = 0; i < instCnt_; i++) {
    lngthScsrElmnts += insts_[i].GetScsrCnt();
    lengthLatencies += insts_[i].GetPrdcsrCnt();
  }

  memSize = sizeof(InstCount) * lengthLatencies;
  gpuErrchk(hipMallocManaged(&dev_latencies_, memSize));

  int scsrIndex = 0;
  int latencyIndex = 0;
  scsrs_ = new int[lngthScsrElmnts];
  latencies_ = new int[lngthScsrElmnts];
  predOrder_ = new int[lngthScsrElmnts];
  int indexOffset = 0;

  for (InstCount i = 0; i < instCnt_; i++) {
    DependenceType _dep;
    insts_[i].ddgIndex = indexOffset;
    int prdcsrNum, latency, toNodeNum;
    // Partition scsrs_, latencies_, predOrder_ for each SchedInstruction.
    for (SchedInstruction *crntScsr = insts_[i].GetFrstScsr(&prdcsrNum, 
                                              &latency,
                                              &_dep, 
                                              &toNodeNum);
      crntScsr != NULL; 
      crntScsr = insts_[i].GetNxtScsr(&prdcsrNum, &latency, &_dep, &toNodeNum)) {
        scsrs_[indexOffset] = toNodeNum;
        latencies_[indexOffset] = latency;
        predOrder_[indexOffset] = prdcsrNum;
        indexOffset += 1;
    }

    // Copy SchedInstruction/GraphNode pointers and link them to device inst
    // and update RegFiles pointer to dev_regFiles
    insts_[i].CopyPointersToDevice(&dev_DDG->insts_[i], dev_DDG->insts_,
                                   dev_regFiles, numThreads,
                                   dev_latencies_, latencyIndex);
  }
  memSize = sizeof(int) * lngthScsrElmnts;
  gpuErrchk(hipMalloc(&(dev_DDG->scsrs_), memSize));
  gpuErrchk(hipMalloc(&(dev_DDG->latencies_), memSize));
  gpuErrchk(hipMalloc(&(dev_DDG->predOrder_), memSize));

  gpuErrchk(hipMemcpy(dev_DDG->scsrs_, scsrs_, memSize, hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(dev_DDG->latencies_, latencies_, memSize, hipMemcpyHostToDevice));
  gpuErrchk(hipMemcpy(dev_DDG->predOrder_, predOrder_, memSize, hipMemcpyHostToDevice));

  memSize = sizeof(SchedInstruction) * instCnt_;
  gpuErrchk(hipMemPrefetchAsync(dev_insts, memSize, 0));
  memSize = sizeof(RegisterFile) * machMdl_->GetRegTypeCnt();
  gpuErrchk(hipMemPrefetchAsync(dev_regFiles, memSize, 0));
  memSize = sizeof(InstCount) * lengthLatencies;
  gpuErrchk(hipMemPrefetchAsync(dev_latencies_, memSize, 0));
}

void DataDepGraph::FreeDevicePointers(int numThreads) {
  hipFree(instCntPerType_);
  hipFree(instCntPerIssuType_);
  hipFree(frwrdLwrBounds_);
  hipFree(bkwrdLwrBounds_);
  hipFree(tplgclOrdr_);
  for (InstCount i = 0; i < machMdl_->GetRegTypeCnt(); i++)
    RegFiles[i].FreeDevicePointers();
  hipFree(RegFiles);
  for (InstCount i = 0; i < instCnt_; i++)
    insts_[i].FreeDevicePointers(numThreads);
  hipFree(insts_);
  hipFree(scsrs_);
  hipFree(latencies_);
  hipFree(predOrder_);
  // hipFree(dev_latencies_);
  // hipFree(dev_crntRange_);
  // (Josh) These frees are invalid but I am not sure why
  // hipFree(dev_scsrElmnts_);
  // hipFree(dev_keys_);
}

void DataDepGraph::FreeDevEdges() {
  if (dev_edges_)
    hipFree(dev_edges_);
}

/*
SchedInstruction *DataDepSubGraph::GetInstByTplgclOrdr(InstCount ordr) {
  assert(ordr >= 0 && ordr < instCnt_);
  return insts_[ordr];
}

SchedInstruction *DataDepSubGraph::GetInstByRvrsTplgclOrdr(InstCount ordr) {
  assert(ordr >= 0 && ordr < instCnt_);
  InstCount indx = instCnt_ - 1 - ordr;
  assert(indx >= 0 && indx < instCnt_);
  return insts_[indx];
}

__host__ __device__
SchedInstruction *DataDepSubGraph::GetRootInst() { return rootInst_; }

__host__ __device__
SchedInstruction *DataDepSubGraph::GetLeafInst() { return leafInst_; }

__host__ __device__
SchedInstruction *DataDepSubGraph::GetInstByIndx(InstCount indx) {
  assert(indx >= 0 && indx < instCnt_);
  return insts_[indx];
}

bool DataDepSubGraph::IsInGraph(SchedInstruction *inst) {
  assert(inst != NULL);

  if (inst == rootInst_ || inst == leafInst_) {
    return true;
  }

  InstCount instNum = inst->GetNum();
  assert(0 <= instNum && instNum < fullGraph_->GetInstCnt());
  bool isIn = numToIndx_[instNum] != INVALID_VALUE;
  return isIn;
}

InstCount DataDepSubGraph::GetInstIndx(SchedInstruction *inst) {
  assert(inst != NULL);

  if (inst == rootInst_) {
    return 0;
  }

  if (inst == leafInst_) {
    return instCnt_ - 1;
  }

  InstCount instNum = inst->GetNum();
  assert(0 <= instNum && instNum < fullGraph_->GetInstCnt());
  InstCount indx = numToIndx_[instNum];
  assert(indx != INVALID_VALUE);
  return indx;
}

InstCount DataDepSubGraph::GetRealInstCnt_() {
  assert(instCnt_ >= 2);
  return instCnt_ - 2;
}

void DataDepSubGraph::GetLwrBounds(InstCount *&frwrdLwrBounds,
                                   InstCount *&bkwrdLwrBounds) {
  if (dynmcLwrBoundsSet_) {
    frwrdLwrBounds = dynmcFrwrdLwrBounds_;
    bkwrdLwrBounds = dynmcBkwrdLwrBounds_;
  } else {
    frwrdLwrBounds = frwrdLwrBounds_;
    bkwrdLwrBounds = bkwrdLwrBounds_;
  }

  assert(frwrdLwrBounds != NULL);
  assert(bkwrdLwrBounds != NULL);
}

InstCount DataDepSubGraph::GetOrgnlInstCnt() {
  return instCnt_ - extrnlInstCnt_;
}

InstCount DataDepSubGraph::GetLwrBound() { return schedLwrBound_ - 2; }

InstCount DataDepSubGraph::GetLostInstCnt_() {
  return lostInsts_->GetElmntCnt();
}
*/
__host__ __device__
bool DataDepStruct::IncludesUnpipelined() { return includesUnpipelined_; }

bool DataDepGraph::IncludesUnsupported() { return includesUnsupported_; }

bool DataDepGraph::IncludesNonStandardBlock() {
  return includesNonStandardBlock_;
}

bool DataDepGraph::IncludesCall() { return includesCall_; }

InstCount DataDepGraph::GetRealInstCnt() { return realInstCnt_; }

InstCount DataDepGraph::GetCodeSize() { return realInstCnt_ - 2; }

void DataDepGraph::SetHard(bool isHard) { isHard_ = isHard; }
