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
#include <hip/hip_runtime.h>

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

// Values for Uninitiated nodes
const InstCount UNINITIATED_NUM = -1;
const InstType UNINITIATED_TYPE = -1;

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
  __host__ __device__
  DataDepStruct(MachineModel *machMdl);
  // TODO(max): Document.
  __host__
  virtual ~DataDepStruct();

  __host__ __device__
  InstCount GetInstCnt();
  virtual InstCount GetOrgnlInstCnt();
  //__host__ __device__
  //virtual SchedInstruction *GetInstByIndx(InstCount instIndx) = 0;
  virtual SchedInstruction *GetInstByTplgclOrdr(InstCount ordr) = 0;
  virtual SchedInstruction *GetInstByRvrsTplgclOrdr(InstCount ordr) = 0;

  //__host__ __device__
  //virtual SchedInstruction *GetRootInst() = 0;
  //__host__ __device__
  //virtual SchedInstruction *GetLeafInst() = 0;

  __host__ __device__
  void GetInstCntPerIssuType(InstCount instCntPerIssuType[]);
  __host__ __device__
  bool IncludesUnpipelined();

  virtual bool IsInGraph(SchedInstruction *inst) = 0;
  virtual InstCount GetInstIndx(SchedInstruction *inst) = 0;
  DEP_GRAPH_TYPE GetType();
  __host__ __device__
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

  // An array of instructions
  SchedInstruction *insts_;

  // If compiling on device, keep track of the pointers for all edges added
  // to facilitate a fast copy of edges to device
  std::vector<GraphEdge *> *edges_;
  // Device array of all GraphEdges
  GraphEdge *dev_edges_;

  // The number of issue types of the machine which this graph uses.
  int16_t issuTypeCnt_;

  // An array holding the number of instructions of each issue type.
  InstCount *instCntPerIssuType_;

  InstCount schedLwrBound_;
  InstCount schedUprBound_;

  InstCount *frwrdLwrBounds_;
  InstCount *bkwrdLwrBounds_;

  bool includesUnpipelined_;

  __host__
  InstCount CmputRsrcLwrBound_();
  __host__
  virtual InstCount CmputAbslutUprBound_();
};

// TODO(max): Find out what this really is.
// The Data Dependence Graph is a sepcial case of a DAG and a special case of
// a Data Dependence Structure as well
class DataDepGraph : public llvm::opt_sched::OptSchedDDGWrapperBase,
                     public DirAcycGraph,
                     public DataDepStruct {
public:
  __host__
  DataDepGraph(MachineModel *machMdl, LATENCY_PRECISION ltncyPcsn);
  __host__
  virtual ~DataDepGraph();

  //Prevent DDG from being abstract, these should not actually be invoked
  virtual void convertSUnits(bool IgnoreRealEdges, bool IgnoreArtificialEdges) {
    Logger::Fatal("Wrong convertSUnits called");
  }

  virtual void convertRegFiles() {
    Logger::Fatal("Wrong convertRegFiles called");
  }

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
  __host__ __device__
  SchedInstruction *GetInstByIndx(InstCount instIndx);

  SchedInstruction *GetInstByTplgclOrdr(InstCount ordr);
  SchedInstruction *GetInstByRvrsTplgclOrdr(InstCount ordr);

  // Setup the Dep. Graph for scheduling by doing a topological sort
  // followed by critical path computation
  __host__
  FUNC_RESULT SetupForSchdulng(bool cmputTrnstvClsr);
  // Parallelized device version of SetupForSchdulng
  // __device__
  // FUNC_RESULT Dev_SetupForSchdulng(bool cmputTrnstvClsr);
  // Update the Dep after applying graph transformations
  FUNC_RESULT UpdateSetupForSchdulng(bool cmputTrnstvClsr);

  // Returns transformations that we will apply to the graph
  SmallVector<std::unique_ptr<GraphTrans>, 0> *GetGraphTrans() {
    return graphTrans_;
  }

  void EnableBackTracking();

  void GetCrntLwrBounds(DIRECTION dir, InstCount crntlwrBounds[]);
  void SetCrntLwrBounds(DIRECTION dir, InstCount crntlwrBounds[]);

  __host__ __device__
  SchedInstruction *GetRootInst();
  __host__ __device__
  SchedInstruction *GetLeafInst();

  __host__ __device__
  UDT_GLABEL GetMaxLtncySum();
  __host__ __device__
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

  LATENCY_PRECISION GetLtncyPrcsn() const { return ltncyPrcsn_;}

  // Add edges to enforce the original program order, assuming that
  // it is represented by the instruction numbers
  void EnforceProgOrder();

  bool UseFileBounds();
  void PrintLwrBounds(DIRECTION dir, std::ostream &out,
                      const char *const title);
  void RestoreAbsoluteBounds();

  void PrintInstTypeInfo(FILE *file);

  // Count dependences and cross-dependences
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

  __host__ __device__
  InstCount GetMaxFileSchedOrder() { return maxFileSchedOrder_; }
  void PrintEdgeCntPerLtncyInfo();

  __host__ __device__
  int16_t GetMaxUseCnt() { return maxUseCnt_; }
  int16_t GetRegTypeCnt() { return machMdl_->GetRegTypeCnt(); }
  __host__ __device__
  int GetPhysRegCnt(int16_t regType) {
    return machMdl_->GetPhysRegCnt(regType);
  }

  __host__ __device__
  InstCount GetMaxIndependentInstructions() { return maxIndependentInstructions_; }
  __host__ __device__
  void SetMaxIndependentInstructions(InstCount maxIndependentInstructions) { maxIndependentInstructions_ = maxIndependentInstructions; }

  __host__ __device__
  RegisterFile *getRegFiles() { return RegFiles; }
  __host__ __device__
  Register *getRegByTuple(RegIndxTuple *tuple) { 
    return RegFiles[tuple->regType_].GetReg(tuple->regNum_); 
  }

  int* scsrs_;
  int* latencies_;
  int* predOrder_;
  int* ltncyPerPrdcsr_;

  // Tracks all registers in the scheduling region. Each RegisterFile
  // object holds all registers for a given register type.
  RegisterFile *RegFiles;

  // Deep Copies DDG's arrays to device and links them to device DDG pointer
  void CopyPointersToDevice(DataDepGraph *dev_DDG, int numThreads = 0);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  void FreeDevicePointers(int numThreads);
  // frees the dev_edges_ array, for some reason did not work in the destructor
  void FreeDevEdges();

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

  // The maximum sum of latencies from a sinlge instruction
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
  SmallVector<std::unique_ptr<GraphTrans>, 0> *graphTrans_;
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

  InstCount maxIndependentInstructions_;

  __host__
  void AllocArrays_(InstCount instCnt);
  FUNC_RESULT ParseF2Nodes_(SpecsBuffer *specsBuf, MachineModel *machMdl);
  FUNC_RESULT ParseF2Edges_(SpecsBuffer *specsBuf, MachineModel *machMdl);
  FUNC_RESULT ParseF2Blocks_(SpecsBuffer *buf);

  FUNC_RESULT ReadInstName_(SpecsBuffer *buf, int i, char *instName,
                            char *prevInstName, char *opCode,
                            InstCount &nodeNum, InstType &instType,
                            NXTLINE_TYPE &nxtLine);

  __host__ __device__
  SchedInstruction *CreateNode_(InstCount instNum, const char *const instName,
                                InstType instType, const char *const opCode,
                                int nodeID, InstCount fileSchedOrder,
                                InstCount fileSchedCycle, InstCount fileLB,
                                InstCount fileUB, int blkNum);
  FUNC_RESULT FinishNode_(InstCount nodeNum, InstCount edgeCnt = -1);
  __host__
  void CreateEdge_(InstCount frmInstNum, InstCount toInstNum, int ltncy,
                   DependenceType depType, bool IsArtificial = false);

  FUNC_RESULT Finish_();

  __host__
  void CmputCrtclPaths_();
  __host__
  void CmputCrtclPathsFrmRoot_();
  __host__
  void CmputCrtclPathsFrmLeaf_();
  __host__
  void CmputCrtclPathsFrmRcrsvScsr_(SchedInstruction *ref);
  __host__
  void CmputCrtclPathsFrmRcrsvPrdcsr_(SchedInstruction *ref);
  __host__
  void CmputRltvCrtclPaths_(DIRECTION dir);
  __host__
  void CmputBasicLwrBounds_();

  void WriteNodeInfoToF2File_(FILE *file);
  void WriteDepInfoToF2File_(FILE *file);

  void AdjstFileSchedCycles_();
};
/*****************************************************************************

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

  struct LostInst{
    SchedInstruction *inst;
    InstCount indx;
  };

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
  __host__ __device__
  InstCount CmputAbslutUprBound_();

public:
  DataDepSubGraph(DataDepGraph *fullGraph, InstCount maxInstCnt,
                  MachineModel *machMdl);
  __host__ __device__
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
  __host__ __device__
  SchedInstruction *GetInstByIndx(InstCount instIndx);
  SchedInstruction *GetInstByTplgclOrdr(InstCount ordr);
  SchedInstruction *GetInstByRvrsTplgclOrdr(InstCount ordr);

  __host__ __device__
  SchedInstruction *GetRootInst();
  __host__ __device__
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
*****************************************************************************/

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
  InstCount *dev_instInSlot_;

  // An array indexed by instruction number which contains the linear slot
  // number in which that instruction has been scheduled
  InstCount *slotForInst_;
  InstCount *dev_slotForInst_;

  // The current slot number for the iterator
  InstCount iterSlotNum_;

  InstCount cost_;
  InstCount execCost_;

  // An array of spill costs at all points in the schedule
  InstCount *spillCosts_;
  InstCount *dev_spillCosts_;

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
  InstCount *dev_peakRegPressures_;

  // The number of conflicts among live ranges
  int cnflctCnt_;

  // The number of live ranges involved in high-reg pressure sections
  // exceeding the physical reg limit. These live ranges may get spilled
  // by the reg allocator
  int spillCnddtCnt_;

  MachineModel *machMdl_;
  MachineModel *dev_machMdl_;

  bool vrfy_;

  int totalStalls_, unnecessaryStalls_;
  bool isZeroPerp_;

  bool VerifySlots_(MachineModel *machMdl, DataDepGraph *dataDepGraph);
  bool VerifyDataDeps_(DataDepGraph *dataDepGraph);
  __host__ __device__
  void GetCycleAndSlotNums_(InstCount globSlotNum, InstCount &cycleNum,
                            InstCount &slotNum);

public:
  InstSchedule(MachineModel *machMdl, DataDepGraph *dataDepGraph, bool vrfy);
  // dummy constructor
  InstSchedule();
  ~InstSchedule();
  bool operator==(InstSchedule &b) const;

  __host__ __device__
  InstCount GetCrntLngth();
  __host__ __device__
  void Reset();

  // Add an instruction sequentially to the current issue slot.
  // If the current slot contains a fixed instruction this function fails
  __host__ __device__
  bool AppendInst(InstCount instNum);

  bool AppendInst(SchedInstruction *inst);

  // Remove the last instruction scheduled
  bool RemoveLastInst();

  // Get the cycle in which the given instruction (by number) is scheduled
  __host__ __device__
  InstCount GetSchedCycle(InstCount instNum);
  InstCount GetSchedCycle(SchedInstruction *inst);

  __host__ __device__
  void SetCost(InstCount cost);
  __host__ __device__
  InstCount GetCost() const;
  __host__ __device__
  void SetExecCost(InstCount cost);
  __host__ __device__
  InstCount GetExecCost() const;
  __host__ __device__
  void SetSpillCost(InstCount cost);
  __host__ __device__
  InstCount GetSpillCost() const;
  __host__ __device__
  void SetNormSpillCost(InstCount cost);
  __host__ __device__
  InstCount GetNormSpillCost() const;
  __host__ __device__
  void SetExtraSpillCost(SPILL_COST_FUNCTION Fn, InstCount cost);
  __host__ __device__
  InstCount GetExtraSpillCost(SPILL_COST_FUNCTION Fn) const;

  __host__ __device__
  void ResetInstIter();
  __host__ __device__
  InstCount GetFrstInst(InstCount &cycleNum, InstCount &slotNum);
  __host__ __device__
  InstCount GetNxtInst(InstCount &cycleNum, InstCount &slotNum);
  // Returns instNum of instruction that is scheduled immediately
  // before the instNum that is passed in. Used for parallel UpdatePheremone
  __device__
  InstCount GetPrevInstNum(InstCount instNum);

  __host__ __device__
  bool IsComplete();

  // Copy schedule src into the current schedule
  __host__ __device__
  void Copy(InstSchedule *src);

  __host__ __device__
  void SetSpillCosts(InstCount *spillCosts);
  // Device version of set spill costs
  __device__
  void Dev_SetSpillCosts(InstCount **spillCosts);
  __host__ __device__
  void SetPeakRegPressures(InstCount *regPressures);
  // Device version of PeakRegPressures
  __device__
  void Dev_SetPeakRegPressures(InstCount **regPressures);
  InstCount GetPeakRegPressures(const InstCount *&regPressures) const;
  __host__ __device__
  InstCount GetSpillCost(InstCount stepNum);
  InstCount GetTotSpillCost();
  int GetConflictCount();
  void SetConflictCount(int cnflctCnt);
  int GetSpillCandidateCount();
  void SetSpillCandidateCount(int cnflctCnt);

  __host__ __device__
  void Print();
  void PrintInstList(FILE *file, DataDepGraph *dataDepGraph,
                     const char *title) const;
  void PrintRegPressures() const;
  bool Verify(MachineModel *machMdl, DataDepGraph *dataDepGraph);
  void PrintClassData();
  // Allocates arrays on device
  void AllocateOnDevice(MachineModel *dev_machMdl);
  // Divide up passed dev array and set the dev pointers to pieces
  // of the passed array
  void SetDevArrayPointers(MachineModel *dev_machMdl, InstCount *dev_temp);
  // Returns size needed for all dev arrays for a schedule
  // used to preallocate memory to be passed to SetDevArrayPointers
  size_t GetSizeOfDevArrays();
  // Copies host arrays to device
  void CopyArraysToDevice();
  // Copies device arrays to host
  void CopyArraysToHost();
  void FreeDeviceArrays();
  // Initializes schedules on device, used between iterations of ACO
  __device__
  void Initialize();
  __host__ __device__
  inline void incrementTotalStalls() {totalStalls_++;}
  __host__ __device__
  inline void incrementUnnecessaryStalls() {unnecessaryStalls_++;}
  __host__ __device__
  inline void resetTotalStalls() {totalStalls_ = 0;}
  __host__ __device__
  inline void resetUnnecessaryStalls() {unnecessaryStalls_ = 0;}
  __host__ __device__
  inline int getTotalStalls() const {return totalStalls_;}
  __host__ __device__
  inline int getUnnecessaryStalls() const {return unnecessaryStalls_;}
  __host__ __device__
  void setIsZeroPerp(bool isZeroPerp) { isZeroPerp_ = isZeroPerp; }
  __host__ __device__
  bool getIsZeroPerp() { return isZeroPerp_; }
};
/*****************************************************************************/

} // namespace opt_sched
} // namespace llvm

#endif
