/*******************************************************************************
Description:  Defines the data dependence graph class, which is one of the
              central components of the instruction scheduler.
Author:       Ghassan Shobaki
Created:      Apr. 2002
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_BASIC_DATA_DEP_H
#define OPTSCHED_BASIC_DATA_DEP_H

#include "opt-sched/Scheduler/OptSchedDDGWrapperBase.h"
#include "opt-sched/Scheduler/buffers.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace llvm {
namespace opt_sched {

// The algorithm to use for determining the lower bound.
enum LB_ALG {
  // Rim and Jain's Algorithm.
  LBA_RJ,
  // Langevin and Cerny's Algorithm.
  LBA_LC
};

// Precision  of latencies
enum LATENCY_PRECISION {
  // precise: as passed by the compiler
  LTP_PRECISE,
  // rough: map all large latencies (>=10) to 10 and all small latencies to 1
  LTP_ROUGH,
  // unity: set all latencies to unity, thus ignoring ILP and scheduling only
  // for RP
  LTP_UNITY
};

// Filters to select which DAGs to write to the output file.
enum OUTPUT_DAGS {
  // All DAGs are written.
  ODG_ALL,
  // Only optimal DAGs are written.
  ODG_OPT,
  // Only improved DAGs (whether optimal or not) are written.
  ODG_IMP,
  // Only hard DAGs are written.
  ODG_HARD,
  // No DAGs are written.
  ODG_NONE
};

// The format of the DAG description file to be read.
enum DAG_FILE_FORMAT {
  // Basic block.
  DFF_BB,
  // Superblock.
  DFF_SB,
  // Trace.
  DFF_TR
};

// The type of the dependence graph.
// TODO(max): Elaborate.
enum DEP_GRAPH_TYPE {
  // A full dependence graph.
  DGT_FULL,
  // A subgraph.
  DGT_SUB
};

// The subgraph type.
// TODO(max): Elaborate.
enum SUB_GRAPH_TYPE {
  // Continuous.
  SGT_CONT,
  // Discontinuous.
  SGT_DISC
};

// TODO(max): Document.
const size_t MAX_INSTNAME_LNGTH = 160;

// An extra to be added to the absolute upper bound. This is a hack to fix some
// elusive legacy bug which may or may not exist anymore.
// TODO(ghassan): Eliminate.
const int SCHED_UB_EXTRA = 20;

const int MAX_LATENCY_VALUE = 10;

// The total number of possible graph transformations.
const int NUM_GRAPH_TRANS = 1;

// Forward declarations used to reduce the number of #includes.
class MachineModel;
class SpecsBuffer;
class RelaxedScheduler;
class RJ_RelaxedScheduler;
class LC_RelaxedScheduler;
class Register;
class RegisterFile;
class InstSchedule;
class GraphTrans;

// TODO(max): Document.
class DataDepStruct {
public:
  // TODO(max): Document.
  DataDepStruct(MachineModel *machMdl);
  // TODO(max): Document.
  virtual ~DataDepStruct();

  virtual InstCount GetInstCnt();
  virtual InstCount GetOrgnlInstCnt();
  virtual SchedInstruction *GetInstByIndx(InstCount instIndx) = 0;
  virtual SchedInstruction *GetInstByTplgclOrdr(InstCount ordr) = 0;
  virtual SchedInstruction *GetInstByRvrsTplgclOrdr(InstCount ordr) = 0;

  virtual SchedInstruction *GetRootInst() = 0;
  virtual SchedInstruction *GetLeafInst() = 0;

  void GetInstCntPerIssuType(InstCount instCntPerIssuType[]);
  bool IncludesUnpipelined();

  virtual bool IsInGraph(SchedInstruction *inst) = 0;
  virtual InstCount GetInstIndx(SchedInstruction *inst) = 0;
  DEP_GRAPH_TYPE GetType();
  InstCount GetAbslutSchedUprBound();
  void SetAbslutSchedUprBound(InstCount bound);
  virtual void GetLwrBounds(InstCount *&frwrdLwrBounds,
                            InstCount *&bkwrdLwrBounds);
  virtual InstCount GetRltvCrtclPath(SchedInstruction *ref,
                                     SchedInstruction *inst, DIRECTION dir) = 0;
  virtual InstCount GetDistFrmLeaf(SchedInstruction *inst) = 0;

protected:
  // A pointer to the machine which this graph uses.
  MachineModel *machMdl_;

  DEP_GRAPH_TYPE type_;

  // The total number of instructions in the graph.
  InstCount instCnt_;

  // An array of pointers to instructions.
  // TODO(max): Elaborate.
  SchedInstruction **insts_;

  // The number of issue types of the machine which this graph uses.
  int16_t issuTypeCnt_;

  // An array holding the number of instructions of each issue type.
  InstCount *instCntPerIssuType_;

  InstCount schedLwrBound_;
  InstCount schedUprBound_;

  InstCount *frwrdLwrBounds_;
  InstCount *bkwrdLwrBounds_;

  bool includesUnpipelined_;

  InstCount CmputRsrcLwrBound_();
  virtual InstCount CmputAbslutUprBound_();
};

// TODO(max): Find out what this really is.
// The Data Dependence Graph is a special case of a DAG and a special case of
// a Data Dependence Structure as well
class DataDepGraph : public llvm::opt_sched::OptSchedDDGWrapperBase,
                     public DirAcycGraph,
                     public DataDepStruct {
public:
  DataDepGraph(MachineModel *machMdl, LATENCY_PRECISION ltncyPcsn);
  virtual ~DataDepGraph();

  // Reads the data dependence graph from a text file.
  FUNC_RESULT ReadFrmFile(SpecsBuffer *buf, bool &endOfFileReached);
  // Continues reading until the end of the current graph definition,
  // discarding the data.
  FUNC_RESULT SkipGraph(SpecsBuffer *buf, bool &endOfFileReached);
  // Writes the data dependence graph to a text file.
  FUNC_RESULT WriteToFile(FILE *file, FUNC_RESULT rslt, InstCount imprvmnt,
                          long number);
  // Returns the string ID of the graph as read from the input file.
  const char *GetDagID() const;
  // Returns the weight of the graph, as read from the input file.
  float GetWeight() const;

  // Given an instruction number, return a pointer to the instruction object.
  SchedInstruction *GetInstByIndx(InstCount instIndx);

  SchedInstruction *GetInstByTplgclOrdr(InstCount ordr);
  SchedInstruction *GetInstByRvrsTplgclOrdr(InstCount ordr);

  // Setup the Dep. Graph for scheduling by doing a topological sort
  // followed by critical path computation
  FUNC_RESULT SetupForSchdulng(bool cmputTrnstvClsr);
  // Update the Dep after applying graph transformations
  FUNC_RESULT UpdateSetupForSchdulng(bool cmputTrnstvClsr);

  // Returns transformations that we will apply to the graph
  SmallVector<std::unique_ptr<GraphTrans>, 0> *GetGraphTrans() {
    return &graphTrans_;
  }

  void EnableBackTracking();

  void GetCrntLwrBounds(DIRECTION dir, InstCount crntlwrBounds[]);
  void SetCrntLwrBounds(DIRECTION dir, InstCount crntlwrBounds[]);

  SchedInstruction *GetRootInst();
  SchedInstruction *GetLeafInst();

  UDT_GLABEL GetMaxLtncySum();
  UDT_GLABEL GetMaxLtncy();

  bool DoesFeedUser(SchedInstruction *inst);

  // Get a lower bound on the schedule length
  InstCount GetSchedLwrBound();

  // Get the lower and upper bounds read from the input file
  void GetFileSchedBounds(InstCount &lwrBound, InstCount &uprBound) const;

  InstCount GetFileSchedLngth() { return fileSchedLngth_; }
  InstCount GetAdjustedFileSchedCycle(InstCount instNum);

  // Get the target upper bound from the input file
  InstCount GetFileSchedTrgtUprBound();

  // Get the final lower and upper bounds
  void GetFinalBounds(InstCount &lwrBound, InstCount &uprBound);

  // Set the final lower and upper bounds
  void SetFinalBounds(InstCount lwrBound, InstCount uprBound);

  int GetFileCostUprBound();

  // Add edges to enforce the original program order, assuming that
  // it is represented by the instruction numbers
  void EnforceProgOrder();

  bool UseFileBounds();
  void PrintLwrBounds(DIRECTION dir, std::ostream &out,
                      const char *const title);
  void RestoreAbsoluteBounds();

  void PrintInstTypeInfo(FILE *file);

  // Count dependencies and cross-dependencies
  void CountDeps(InstCount &totDepCnt, InstCount &crossDepCnt);

  int GetBscBlkCnt();
  bool IsInGraph(SchedInstruction *inst);
  InstCount GetInstIndx(SchedInstruction *inst);
  InstCount GetRltvCrtclPath(SchedInstruction *ref, SchedInstruction *inst,
                             DIRECTION dir);
  void SetCrntFrwrdLwrBound(SchedInstruction *inst);
  void SetSttcLwrBounds();
  void SetDynmcLwrBounds();
  void CreateEdge(SchedInstruction *frmNode, SchedInstruction *toNode,
                  int ltncy, DependenceType depType);
  InstCount GetDistFrmLeaf(SchedInstruction *inst);

  void SetPrblmtc();
  bool IsPrblmtc();

  bool IncludesUnsupported();
  bool IncludesNonStandardBlock();
  bool IncludesCall();

  InstCount GetRealInstCnt();
  InstCount GetCodeSize();
  void SetHard(bool isHard);
  bool IsHard() { return isHard_; }

  int GetEntryInstCnt() { return entryInstCnt_; }
  int GetExitInstCnt() { return exitInstCnt_; }

  InstCount GetMaxFileSchedOrder() { return maxFileSchedOrder_; }
  void PrintEdgeCntPerLtncyInfo();

  int16_t GetMaxUseCnt() { return maxUseCnt_; }
  int16_t GetRegTypeCnt() { return machMdl_->GetRegTypeCnt(); }
  int GetPhysRegCnt(int16_t regType) {
    return machMdl_->GetPhysRegCnt(regType);
  }

  RegisterFile *getRegFiles() { return RegFiles.get(); }

protected:
  // TODO(max): Get rid of this.
  // Number of basic blocks
  int32_t bscBlkCnt_;

  // How many instruction types are supported
  int16_t instTypeCnt_;

  // An array holding the number of instructions of each type
  InstCount *instCntPerType_;

  // The total sum of latencies in the entire graph
  // Will be useful to compute an absolute upper bound on the schedule length
  //  InstCount totLtncySum_;

  // The maximum latency in the graph
  UDT_GLABEL maxLtncy_;

  // The maximum sum of latencies from a single instruction
  UDT_GLABEL maxLtncySum_;

  // Upper and lower bounds read from the input file
  InstCount fileSchedLwrBound_;
  InstCount fileSchedUprBound_;
  InstCount fileSchedTrgtUprBound_;
  InstCount fileCostUprBound_;

  InstCount fileSchedLngth_;
  InstCount minFileSchedCycle_;
  InstCount maxFileSchedOrder_;
  int16_t maxUseCnt_;

  // Final upper and lower bounds when the solver completes or times out
  InstCount finalLwrBound_;
  InstCount finalUprBound_;

  // The length of some known schedule that has been computed by some other
  // program ,e.g., gcc when the input DAGs come from gcc
  InstCount knwnSchedLngth_;

  // A list of DDG mutations
  SmallVector<std::unique_ptr<GraphTrans>, 0> graphTrans_;

  MachineModel *machMdl_;

  bool backTrackEnbl_;

  char dagID_[MAX_NAMESIZE];
  char compiler_[MAX_NAMESIZE];
  float weight_;

  // Override the machine model by using the latencies in the input DAG file
  bool useFileLtncs_;

  DAG_FILE_FORMAT dagFileFormat_;
  bool isTraceFormat_;

  bool wasSetupForSchduling_;

  int32_t lastBlkNum_;

  bool isPrblmtc_;

  OUTPUT_DAGS outptDags_;
  InstCount maxOutptDagSize_;

  bool includesCall_;
  bool includesUnsupported_;
  bool includesNonStandardBlock_;

  InstCount realInstCnt_;
  bool isHard_;

  int entryInstCnt_;
  int exitInstCnt_;

  LATENCY_PRECISION ltncyPrcsn_;
  int edgeCntPerLtncy_[MAX_LATENCY_VALUE + 1];

  // Tracks all registers in the scheduling region. Each RegisterFile
  // object holds all registers for a given register type.
  std::unique_ptr<RegisterFile[]> RegFiles;

  void AllocArrays_(InstCount instCnt);
  FUNC_RESULT ParseF2Nodes_(SpecsBuffer *specsBuf, MachineModel *machMdl);
  FUNC_RESULT ParseF2Edges_(SpecsBuffer *specsBuf, MachineModel *machMdl);
  FUNC_RESULT ParseF2Blocks_(SpecsBuffer *buf);

  FUNC_RESULT ReadInstName_(SpecsBuffer *buf, int i, char *instName,
                            char *prevInstName, char *opCode,
                            InstCount &nodeNum, InstType &instType,
                            NXTLINE_TYPE &nxtLine);

  SchedInstruction *CreateNode_(InstCount instNum, const char *const instName,
                                InstType instType, const char *const opCode,
                                int nodeID, InstCount fileSchedOrder,
                                InstCount fileSchedCycle, InstCount fileLB,
                                InstCount fileUB, int blkNum);
  FUNC_RESULT FinishNode_(InstCount nodeNum, InstCount edgeCnt = -1);
  void CreateEdge_(InstCount frmInstNum, InstCount toInstNum, int ltncy,
                   DependenceType depType, bool IsArtificial = false);

  FUNC_RESULT Finish_();

  void CmputCrtclPaths_();
  void CmputCrtclPathsFrmRoot_();
  void CmputCrtclPathsFrmLeaf_();
  void CmputCrtclPathsFrmRcrsvScsr_(SchedInstruction *ref);
  void CmputCrtclPathsFrmRcrsvPrdcsr_(SchedInstruction *ref);
  void CmputRltvCrtclPaths_(DIRECTION dir);
  void CmputBasicLwrBounds_();

  void WriteNodeInfoToF2File_(FILE *file);
  void WriteDepInfoToF2File_(FILE *file);

  void AdjstFileSchedCycles_();
};
/*****************************************************************************/

class DataDepSubGraph : public DataDepStruct {
protected:
  DataDepGraph *fullGraph_;

  InstCount maxInstCnt_;
  InstCount extrnlInstCnt_;
  InstCount cmpnstdInstCnt_;
  bool instsChngd_;
  bool instAdded_;
  SUB_GRAPH_TYPE subType_;

  SchedInstruction *rootInst_;
  SchedInstruction *leafInst_;

  InstCount *frwrdCrtclPaths_;
  InstCount *bkwrdCrtclPaths_;

  BitVector *rootVctr_;
  BitVector *leafVctr_;

  InstCount *numToIndx_;

  RJ_RelaxedScheduler *RJRlxdSchdulr_;
  RJ_RelaxedScheduler *RJRvrsRlxdSchdulr_;
  LC_RelaxedScheduler *LCRlxdSchdulr_;
  LC_RelaxedScheduler *LCRvrsRlxdSchdulr_;
  RJ_RelaxedScheduler *dynmcRlxdSchdulr_;
  bool dynmcLwrBoundsSet_;
  InstCount *dynmcFrwrdLwrBounds_;
  InstCount *dynmcBkwrdLwrBounds_;
  LinkedList<SchedInstruction> *fxdLst_;

  typedef struct {
    SchedInstruction *inst;
    InstCount indx;
  } LostInst;

  Stack<LostInst> *lostInsts_;

  // The total lower bound that also includes the gap size
  InstCount totLwrBound_;
  InstCount unstsfidLtncy_;
  InstCount rejoinCycle_;

#ifdef IS_DEBUG
  int errorCnt_;
#endif

#ifdef IS_DEBUG_TRACE_ENUM
  bool smplDynmcLB_;
#endif

  void CreateRootAndLeafInsts_();
  void DelRootAndLeafInsts_(bool isFinal);
  void Clear_();
  void InitForLwrBounds_();
  void InitForDynmcLwrBounds_();
  void UpdtSttcLwrBounds_();
  void RmvExtrnlInsts_();

  void SetRootsAndLeaves_();
  bool IsRoot_(SchedInstruction *inst);
  bool IsLeaf_(SchedInstruction *inst);
  InstCount CmputCrtclPath_();
  void CreateEdge_(SchedInstruction *frmInst, SchedInstruction *toInst);
  void RmvEdge_(SchedInstruction *frmInst, SchedInstruction *toInst);
  void CmputCrtclPaths_(DIRECTION dir);
  void CmputCrtclPaths_();
  void FindFrstCycleRange_(InstCount &minFrstCycle, InstCount &maxFrstCycle);
  InstCount GetRealInstCnt_();

  bool TightnDynmcLwrBound_(InstCount frstCycle, InstCount minLastCycle,
                            InstCount maxLastCycle, InstCount trgtLwrBound,
                            InstCount &dynmcLwrBound);
  bool SetDynmcLwrBounds_(InstCount frstCycle, InstCount lastCycle,
                          InstCount shft, InstCount trgtLwrBound,
                          bool useDistFrmLeaf, bool &trgtFsbl);
  bool SetDynmcFrwrdLwrBounds_(InstCount frstCycle, InstCount lastCycle,
                               InstCount shft);
  bool SetDynmcBkwrdLwrBounds_(InstCount lastCycle, InstCount shft,
                               bool useDistFrmLeaf);

  bool ChkInstRanges_(InstCount lastCycle);
  bool ChkInstRange_(SchedInstruction *inst, InstCount indx,
                     InstCount lastCycle);
  bool CmputSmplDynmcLwrBound_(InstCount &dynmcLwrBound, InstCount trgtLwrBound,
                               bool &trgtFsbl);
  InstCount CmputTwoInstDynmcLwrBound_();
  InstCount CmputIndpndntInstDynmcLwrBound_();

  void AddRoot_(SchedInstruction *inst);
  void AddLeaf_(SchedInstruction *inst);
  void RmvLastRoot_(SchedInstruction *inst);
  void RmvLastLeaf_(SchedInstruction *inst);

  void PropagateFrwrdLwrBounds_(InstCount frmIndx, InstCount toIndx,
                                InstCount LwrBounds[], bool reset);
  void PropagateBkwrdLwrBounds_(InstCount frmIndx, InstCount toIndx,
                                InstCount LwrBounds[], bool reset);
  void TightnLwrBound_(DIRECTION dir, InstCount indx, InstCount lwrBounds[]);
  void AllocSttcData_();
  void AllocDynmcData_();
  InstCount CmputMaxReleaseTime_();
  InstCount CmputMaxDeadline_();
  bool CmputEntTrmnlDynmcLwrBound_(InstCount &dynmcLwrBound,
                                   InstCount trgtLwrBound);
  InstCount GetLostInstCnt_();

  //  void CmputExtrnlLtncs_(InstCount rejoinCycle);
  InstCount CmputExtrnlLtncs_(InstCount rejoinCycle, SchedInstruction *inst);

  InstCount CmputExtrnlLtncy_(SchedInstruction *pred, SchedInstruction *scsr,
                              InstCount rejoinCycle, InstCount scsrCycle,
                              bool isSchduld, bool tightnLwrBound);
  InstCount CmputUnstsfidLtncy_();
  void AllocRlxdSchdulr_(LB_ALG lbAlg, RelaxedScheduler *&rlxdSchdulr,
                         RelaxedScheduler *&rvrsRlxdSchdulr);
  void FreeRlxdSchdulr_(LB_ALG lbAlg);
  InstCount CmputAbslutUprBound_();

public:
  DataDepSubGraph(DataDepGraph *fullGraph, InstCount maxInstCnt,
                  MachineModel *machMdl);
  virtual ~DataDepSubGraph();
  void InitForSchdulng(bool clearAll);
  void SetupForDynmcLwrBounds(InstCount schedUprBound);
  void AddInst(SchedInstruction *inst);
  void RmvInst(SchedInstruction *inst);

  InstCount CmputLwrBound(LB_ALG lbAlg, bool addExtrnlLtncs,
                          InstCount rejoinCycle, SchedInstruction *inst,
                          InstCount &instGapSize);

  void CmputTotLwrBound(LB_ALG lbAlg, InstCount rejoinCycle,
                        SchedInstruction *inst, InstCount &lwrBound,
                        InstCount &unstsfidLtncy, bool &crtnRejoin,
                        InstCount &instGapSize);

  InstCount GetLwrBound();
  SchedInstruction *GetInstByIndx(InstCount instIndx);
  SchedInstruction *GetInstByTplgclOrdr(InstCount ordr);
  SchedInstruction *GetInstByRvrsTplgclOrdr(InstCount ordr);

  SchedInstruction *GetRootInst();
  SchedInstruction *GetLeafInst();
  bool IsInGraph(SchedInstruction *inst);
  InstCount GetInstIndx(SchedInstruction *inst);
  InstCount GetRltvCrtclPath(SchedInstruction *ref, SchedInstruction *inst,
                             DIRECTION dir);

  void AddExtrnlInst(SchedInstruction *inst);
  void RmvExtrnlInst(SchedInstruction *inst);
  InstCount GetDistFrmLeaf(SchedInstruction *inst);
  void GetLwrBounds(InstCount *&frwrdLwrBounds, InstCount *&bkwrdLwrBounds);
  InstCount GetOrgnlInstCnt();

  void InstLost(SchedInstruction *inst);
  void UndoInstLost(SchedInstruction *inst);
  InstCount GetAvlblSlots(IssueType issuType);
};
/*****************************************************************************/

// An instance of this class holds all the necessary information about an
// instruction schedule. A scheduler starts with an empty object of this
// class and fills it in as it progresses until it ends up with a complete
// schedule. Different schedulers differ in the order in which they fill it in.
// The numbering of slots in this class is zero-based and linear, not on a
// per-cycle basis. For instance, the second slot in the third cycle of a
// 4-issue machine is slot #9
class InstSchedule {
private:
  int issuRate_;

  // The total number of instructions to be scheduled
  InstCount totInstCnt_;

  // The total number of slots available
  InstCount totSlotCnt_;

  // The number of instructions that have been scheduled so far
  // When this is equal to totInstCnt_ we have a complete schedule
  InstCount schduldInstCnt_;

  // The maximum number of instructions scheduled. Useful in backtracking mode
  InstCount maxSchduldInstCnt_;

  // The maximum instruction number scheduled
  InstCount maxInstNumSchduld_;

  // An absolute upper bound on the schedule length to determine the array
  // sizes
  InstCount schedUprBound_;

  // The number of the next available slot
  InstCount crntSlotNum_;

  // An array indexed by linear slot number which contains the instruction
  // number scheuled in that slot
  InstCount *instInSlot_;

  // An array indexed by instruction number which contains the linear slot
  // number in which that instruction has been scheduled
  InstCount *slotForInst_;

  // The current slot number for the iterator
  InstCount iterSlotNum_;

  InstCount cost_;
  InstCount execCost_;

  // An array of spill costs at all points in the schedule
  InstCount *spillCosts_;

  // Tot spill cost across the entire schedule
  InstCount totSpillCost_;

  // The schedule's spill cost according to the cost function used
  InstCount spillCost_;

  // The normalized spill cost (absolute Spill Cost - lower bound of spill cost)
  InstCount NormSpillCost;

  // Stores the spill cost of other spill cost functions
  InstCount storedSC[MAX_SCF_TYPES];

  // An array of peak reg pressures for all reg types in the schedule
  InstCount *peakRegPressures_;

  // The number of conflicts among live ranges
  int cnflctCnt_;

  // The number of live ranges involved in high-reg pressure sections
  // exceeding the physical reg limit. These live ranges may get spilled
  // by the reg allocator
  int spillCnddtCnt_;

  MachineModel *machMdl_;

  bool vrfy_;

  bool VerifySlots_(MachineModel *machMdl, DataDepGraph *dataDepGraph);
  bool VerifyDataDeps_(DataDepGraph *dataDepGraph);
  void GetCycleAndSlotNums_(InstCount globSlotNum, InstCount &cycleNum,
                            InstCount &slotNum);

public:
  InstSchedule(MachineModel *machMdl, DataDepGraph *dataDepGraph, bool vrfy);
  ~InstSchedule();
  bool operator==(InstSchedule &b) const;

  InstCount GetCrntLngth();
  void Reset();

  // Add an instruction sequentially to the current issue slot.
  // If the current slot contains a fixed instruction this function fails
  bool AppendInst(InstCount instNum);

  bool AppendInst(SchedInstruction *inst);

  // Remove the last instruction scheduled
  bool RemoveLastInst();

  // Get the cycle in which the given instruction (by number) is scheduled
  InstCount GetSchedCycle(InstCount instNum);
  InstCount GetSchedCycle(SchedInstruction *inst);

  void SetCost(InstCount cost);
  InstCount GetCost() const;
  void SetExecCost(InstCount cost);
  InstCount GetExecCost() const;
  void SetSpillCost(InstCount cost);
  InstCount GetSpillCost() const;
  void SetNormSpillCost(InstCount cost);
  InstCount GetNormSpillCost() const;
  void SetExtraSpillCost(SPILL_COST_FUNCTION Fn, InstCount cost);
  InstCount GetExtraSpillCost(SPILL_COST_FUNCTION Fn) const;

  void ResetInstIter();
  InstCount GetFrstInst(InstCount &cycleNum, InstCount &slotNum);
  InstCount GetNxtInst(InstCount &cycleNum, InstCount &slotNum);

  bool IsComplete();

  // Copy schedule src into the current schedule
  void Copy(InstSchedule *src);

  void SetSpillCosts(InstCount *spillCosts);
  void SetPeakRegPressures(InstCount *regPressures);
  InstCount GetPeakRegPressures(const InstCount *&regPressures) const;
  InstCount GetSpillCost(InstCount stepNum);
  InstCount GetTotSpillCost();
  int GetConflictCount();
  void SetConflictCount(int cnflctCnt);
  int GetSpillCandidateCount();
  void SetSpillCandidateCount(int cnflctCnt);

  void Print(std::ostream &out, const char *const title);
  void PrintInstList(FILE *file, DataDepGraph *dataDepGraph,
                     const char *title) const;
  void PrintRegPressures() const;
  bool Verify(MachineModel *machMdl, DataDepGraph *dataDepGraph);
  void PrintClassData();
};
/*****************************************************************************/

} // namespace opt_sched
} // namespace llvm

#endif
