/*******************************************************************************
Description:  Defines basic data structures needed for instruction scheduling.
Author:       Ghassan Shobaki
Created:      Apr. 2002
Last Update:  Sept. 2013
*******************************************************************************/

#ifndef OPTSCHED_BASIC_SCHED_BASIC_DATA_H
#define OPTSCHED_BASIC_SCHED_BASIC_DATA_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/graph.h"
#include "opt-sched/Scheduler/hash_table.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace llvm {
namespace opt_sched {

using std::string;

// List scheduling heuristic.
enum LISTSCHED_HEURISTIC {
  // Critical path.
  LSH_CP = 0,

  // Last use count:
  // the number of virtual regs for which this instruction is the last user
  LSH_LUC = 1,

  // Use count:
  // the number of virtual regs for which this instruction is a user
  LSH_UC = 2,

  // Node ID
  LSH_NID = 3,

  // Critical path with resources taken into account
  LSH_CPR = 4,

  // Input scheduling order: scheduling order of the input instruction stream
  LSH_ISO = 5,

  // Successor count
  LSH_SC = 6,

  // Latency sum
  LSH_LS = 7,

  // LLVM list scheduler order
  LSH_LLVM = 8
};

#define MAX_SCHED_PRIRTS 10

// An ordered vector of priority schemes based on which scheduling
// will be performed
struct SchedPriorities {
  int cnt;
  bool isDynmc;
  LISTSCHED_HEURISTIC vctr[MAX_SCHED_PRIRTS];
};

enum SPILL_COST_FUNCTION {
  // peak excess reg pressure at one point in the schedule
  SCF_PERP,
  // peak reg pressure at one point in the schedule
  SCF_PRP,
  // peak excess reg pressure per type, that is, the peak is
  // taken for each type individually, thus different types
  // may have peaks at different points in the schedule
  SCF_PEAK_PER_TYPE,
  // sum of excess reg pressures at all points in the block
  SCF_SUM,
  // peak excess reg pressure plus the avg excess reg pressure across the block
  SCF_PEAK_PLUS_AVG,
  // (Chris) Sum of live interval lengths.
  SCF_SLIL,
  // Run a register allocator and count the spills
  SCF_SPILLS,
  // Get target specific RP cost (e.g. GCN Occupancy)
  SCF_TARGET
};

#define MAX_SCF_TYPES 10

// The type of instruction signatures, used by the enumerator's history table to
// keep track of partial schedules.
typedef UDT_HASHKEY InstSignature;

// The cycle or slot number of an instruction node which has not been scheduled.
const int SCHD_UNSCHDULD = -1;
// TODO(ghassan): Document.
const int SCHD_STALL = -2;

// TODO(max): Eliminate these limits.
// The maximum number of register definition per instruction node.
const int MAX_DEFS_PER_INSTR = 4096;
// The maximum number of register usages per instruction node.
const int MAX_USES_PER_INSTR = 4096;

// function for parsing cost function names to enum values
SPILL_COST_FUNCTION ParseSCFName(const std::string &name);

// Forward declarations used to reduce the number of #includes.
class DataDepGraph;
class Register;

// There is a circular dependence between SchedInstruction and SchedRange.
class SchedRange;

// An object of this class contains all the information that a scheduler
// needs to keep track of for an instruction. This class is derived from
// GraphNode, since, from the scheduler's point of view, an instruction is a
// node in the data dependence graph. During scheduling it is critical for each
// instruction object to be linked to its predecessors and successors in the
// graph so that it can receive/propagate scheduling information from/to them.
class SchedInstruction : public GraphNode {
public:
  // Creates a new instruction node for scheduling. Parameters are as follows:
  //   num: The number of this graph node.
  //   name: The name of this instruction's type, e.g. iarith, move, etc. Only
  //     stored, not used by this class.
  //   instType: The instruction type (defined in its machine model).
  //   opCode: The mnemonic for this instruction, e.g. "add" or "jmp". Only
  //     stored, not used by this class.
  //   maxInstCnt: The maximum number of instructions in the graph. Passed to
  //     the graph node constructor which uses it for calculating memory
  //     allocation bounds.
  //   nodeID: The ID of this node. Only stored, not used by this class.
  //   fileSchedCycle: The scheduled cycle for this instruction as provided in
  //     the input file. Only stored, not used by this class.
  //   fileSchedOrder: The scheduling order of this instruction as provided in
  //     the input file. Only stored, not used by this class.
  //   fileLB: The static lower bound on this instruction's scheduling as
  //     provided in the input file.
  //   fileUB: The static upper bound on this instruction's scheduling as
  //     provided in the input file.
  //   model: The machine model used by this instruction.
  SchedInstruction(InstCount num, const string &name, InstType instType,
                   const string &opCode, InstCount maxInstCnt, int nodeID,
                   InstCount fileSchedCycle, InstCount fileSchedOrder,
                   InstCount fileLB, InstCount fileUB, MachineModel *model);
  // Deallocates the memory used by the instruction and destroys the object.
  ~SchedInstruction();

  // Prepares the instruction for scheduling. Should be called only once in
  // the lifetime of an instruction object.
  void SetupForSchdulng(InstCount instCnt, bool isCP_FromScsr,
                        bool isCP_FromPrdcsr);

  // Sets the instruction's bounds to the ones specified in the input file.
  bool UseFileBounds();

  // Initializes the instruction for a new scheduling iteration. Sometimes,
  // early infeasibility might get detected by this function, in which case it
  // returns false. Otherwise, returns true.
  bool InitForSchdulng(InstCount schedLngth = INVALID_VALUE,
                       LinkedList<SchedInstruction> *fxdLst = NULL);

  // Returns the name of the instruction.
  const char *GetName() const;
  // Returns the opcode of the instruction.
  const char *GetOpCode() const;
  // Returns the ID of the node as specified in the constructor.
  int GetNodeID() const;
  void SetNodeID(int nodeID);
  // Returns the sum of latencies from this instruction.
  int GetLtncySum() const;
  // Returns the maximum latency from this instruction.
  int GetMaxLtncy() const;
  // Returns the scheduling order for this instruction as provided in the
  // input file.
  InstCount GetFileSchedOrder() const;
  // Returns the scheduled cycle for this instruction as provided in the input
  // file.
  InstCount GetFileSchedCycle() const;
  // Returns the instruction's forward or backward lower bound.
  InstCount GetLwrBound(DIRECTION dir) const;

  /***************************************************************************
   * Iterators                                                               *
   ***************************************************************************/
  // Returns the first predecessor of this instruction node and resets the
  // predecessor iterator. Writes edge properties into the parameters if
  // provided:
  //   scsrNum: this instruction's number (order) in the predecessor's
  //     successor list.
  //   ltncy: the latency from the predecessor to this instruction.
  //   depType: the type of dependence between this node and the predecessor.
  SchedInstruction *GetFrstPrdcsr(InstCount *scsrNum = NULL,
                                  UDT_GLABEL *ltncy = NULL,
                                  DependenceType *depType = NULL);
  // Returns the next predecessor of this instruction node and moves the
  // predecessor iterator forward. Fills parameters as above.
  SchedInstruction *GetNxtPrdcsr(InstCount *scsrNum = NULL,
                                 UDT_GLABEL *ltncy = NULL,
                                 DependenceType *depType = NULL);

  // Returns the first successor of this instruction node and resets the
  // successor iterator. Writes edge properties into the parameters if
  // provided:
  //   prdcsrNum: this instruction's number (order) in the successor's
  //     predecessor list.
  //   ltncy: the latency from this instruction to the successor.
  //   depType: the type of dependence between this node and the successor.
  SchedInstruction *GetFrstScsr(InstCount *prdcsrNum = NULL,
                                UDT_GLABEL *ltncy = NULL,
                                DependenceType *depType = NULL,
                                bool *IsArtificial = nullptr);
  // Returns the next successor of this instruction node and moves the
  // successor iterator forward. Fills parameters as above.
  SchedInstruction *GetNxtScsr(InstCount *prdcsrNum = NULL,
                               UDT_GLABEL *ltncy = NULL,
                               DependenceType *depType = NULL,
                               bool *IsArtificial = nullptr);

  // Returns the last successor of this instruction node and moves the
  // successor iterator to the end of the list. If prdcsrNum is provided, this
  // instruction's number (order) in the successor's predecessor list is
  // written to it.
  SchedInstruction *GetLastScsr(InstCount *prdcsrNum = NULL);
  // Returns the previous predecessor of this instruction node and moves the
  // predecessor iterator backward. Fills prdcsrNum as above.
  SchedInstruction *GetPrevScsr(InstCount *prdcsrNum = NULL);

  // Returns the first predecessor or successor of this instruction node,
  // depending on the value of dir, filling in the latency from the
  // predecessor to this instruction into ltncy, if provided. Resets the
  // predecessor or successor iterator (the two iterator are independent).
  SchedInstruction *GetFrstNghbr(DIRECTION dir, UDT_GLABEL *ltncy = NULL);
  // Returns the next predecessor or successor of this instruction node,
  // depending on the value of dir, filling in the latency from the
  // predecessor to this instruction into ltncy, if provided. Moves the
  // predecessor iterator forward (the two iterator are independent).
  SchedInstruction *GetNxtNghbr(DIRECTION dir, UDT_GLABEL *ltncy = NULL);
  /***************************************************************************/

  // Sets the instruction's current forward and backward lower bounds to the
  // specified values.
  void SetBounds(InstCount flb, InstCount blb);
  // Sets one of the instruction's lower bounds.
  void SetLwrBound(DIRECTION dir, InstCount bound, bool isAbslut = true);
  // Sets the instruction's lower bounds to the absolute lower bounds.
  void RestoreAbsoluteBounds();

  // Returns whether this instruction is flagged as being ready.
  bool IsInReadyList() const;
  // Flags this instruction as being ready.
  void PutInReadyList();
  // Flags this instruction as NOT being ready.
  void RemoveFromReadyList();

  // Calculates the instruction's critical path distance from the root,
  // assuming that the critical paths of all of its predecessors have been
  // calculated.
  InstCount CmputCrtclPathFrmRoot();

  // Calculates the instruction's critical path distance from the leaf,
  // assuming that the critical paths of all of its successors have been
  // calculated.
  InstCount CmputCrtclPathFrmLeaf();

  // Returns the critical path distance of this instruction from the root of
  // leaf, depending on dir. Assumes that the path has already been
  // calculated.
  InstCount GetCrtclPath(DIRECTION dir) const;

  // Notifies this instruction that one of its predecessors has been scheduled
  // in a certain cycle. If that was the last predecessor to schedule, this
  // function will return true and set rdyCycle to the cycle in which this
  // instruction will become ready. Otherwise it will return false and set
  // rdyCycle to -1, indicating that it isn't yet known when it will be ready.
  bool PrdcsrSchduld(InstCount prdcsrNum, InstCount cycle, InstCount &rdyCycle);
  // Undoes the effect of PrdcsrSchduld().
  bool PrdcsrUnSchduld(InstCount prdcsrNum, InstCount &rdyCycle);

  // Notifies this instruction that one of its successors has been scheduled.
  // Returns true if that was the last successor to schedule.
  bool ScsrSchduld();

  // Schedules the instruction to a given cycle and clot number.
  void Schedule(InstCount cycleNum, InstCount slotNum);
  // Mark this instruction as unscheduled.
  void UnSchedule();

  // Sets the instruction type to a given value.
  void SetInstType(InstType type);
  // Returns the type of the instruction.
  InstType GetInstType() const;
  // Sets the instruction issue type to a given value.
  void SetIssueType(IssueType type);
  // Returns the issue type of the instruction.
  IssueType GetIssueType() const;

  // Returns whether the instruction has been scheduled. If the cycle argument
  // is provided, it is filled with the cycle to which this instruction has
  // been scheduled.
  bool IsSchduld(InstCount *cycle = NULL) const;

  // Returns the cycle to which this instruction has been scheduled.
  InstCount GetSchedCycle() const;
  // Returns the slot to which this instruction has been scheduled.
  InstCount GetSchedSlot() const;

  // Returns the number of the deadline cycle for this instruction.
  InstCount GetCrntDeadline() const;
  // Returns the release time for this instruction.
  InstCount GetCrntReleaseTime() const;
  // Returns the relaxed cycle number for this instruction.
  // TODO(ghassan): Elaborate.
  InstCount GetRlxdCycle() const;
  // Sets the relaxed cycle number for this instruction.
  // TODO(ghassan): Elaborate.
  void SetRlxdCycle(InstCount cycle);

  // Returns the instruction's current lower bound in the given direction.
  InstCount GetCrntLwrBound(DIRECTION dir) const;
  // Sets the instruction's current lower bound in the given direction.
  void SetCrntLwrBound(DIRECTION dir, InstCount bound);

  // Tightens the lower bound of this instruction to the given new lower bound
  // if it is greater than the current lower bound. Any tightened instruction
  // is added to the given list to be used for efficient untightening. This
  // function returns with false as soon as infeasibility (w.r.t the given
  // schedule length) is detected, otherwise it returns true.
  bool TightnLwrBound(DIRECTION dir, InstCount newLwrBound,
                      LinkedList<SchedInstruction> *tightndLst,
                      LinkedList<SchedInstruction> *fxdLst, bool enforce);
  // Like TightnLwrBound(), but also recursively propagates tightening through
  // the subgraph rooted at this instruction.
  bool TightnLwrBoundRcrsvly(DIRECTION dir, InstCount newLwrBound,
                             LinkedList<SchedInstruction> *tightndLst,
                             LinkedList<SchedInstruction> *fxdLst,
                             bool enforce);
  // Untightens any tightened lower bound.
  void UnTightnLwrBounds();
  // Marks the instruction as not tightened.
  void CmtLwrBoundTightnng();

  // Sets the instruction's signature.
  void SetSig(InstSignature sig);
  // Returns the instruction's signature.
  InstSignature GetSig() const;

  // TODO(ghassan): Document.
  InstCount GetFxdCycle() const;
  // TODO(ghassan): Document.
  bool IsFxd() const;

  // Tightens the lower bound and deadline and recursively propagates these
  // tightenings to the neighbors and checking for feasibility at each point.
  bool ApplyPreFxng(LinkedList<SchedInstruction> *tightndLst,
                    LinkedList<SchedInstruction> *fxdLst);

  // TODO(ghassan): Document.
  InstCount GetPreFxdCycle() const;

  // Probes the successors of this instruction to see if their current lower
  // bounds will get tightened (delayed) if this instruction was scheduled in
  // the given cycle.
  bool ProbeScsrsCrntLwrBounds(InstCount cycle);

  /***************************************************************************
   * Entry/exit-related methods                                              *
   ***************************************************************************/
  // TODO(max): Verify that these are indeed entry/exit-related.
  // TODO(ghassan): Document.
  InstCount GetRltvCrtclPath(DIRECTION dir, SchedInstruction *ref);

  // Calculates the instruction's critical path distance from the given entry
  // node assuming that the critical paths of all of its predecessors have
  // been calculated.
  InstCount CmputCrtclPathFrmRcrsvPrdcsr(SchedInstruction *ref);

  // Calculates the instruction's critical path distance from the given exit
  // node assuming that the critical paths of all of its successors have been
  // calculated.
  InstCount CmputCrtclPathFrmRcrsvScsr(SchedInstruction *ref);
  /***************************************************************************/

  // Returns whether the instruction blocks a scheduling cycle, i.e. prevents
  // any other instructions from running during the same cycle.
  bool BlocksCycle() const;
  // Returns whether the instruction is pipelined.
  bool IsPipelined() const;

  bool MustBeInBBEntry() const;
  bool MustBeInBBExit() const;
  void SetMustBeInBBEntry(bool val);
  void SetMustBeInBBExit(bool val);

  // Add a register definition to this instruction node.
  void AddDef(Register *reg);
  // Add a register usage to this instruction node.
  void AddUse(Register *reg);
  // Returns whether this instruction defines the specified register.
  bool FindDef(Register *reg) const;
  // Returns whether this instruction uses the specified register.
  bool FindUse(Register *reg) const;

  int16_t NumDefs() { return defCnt_; }
  int16_t NumUses() { return useCnt_; }

  llvm::ArrayRef<Register *> GetDefs() {
    return llvm::makeArrayRef(defs_, static_cast<size_t>(defCnt_));
  }

  llvm::ArrayRef<const Register *> GetDefs() const {
    return llvm::makeArrayRef(defs_, static_cast<size_t>(defCnt_));
  }

  llvm::ArrayRef<Register *> GetUses() {
    return llvm::makeArrayRef(uses_, static_cast<size_t>(useCnt_));
  }

  llvm::ArrayRef<const Register *> GetUses() const {
    return llvm::makeArrayRef(uses_, static_cast<size_t>(useCnt_));
  }

  // Return the adjusted use count. The number of uses minus live-out uses.
  int16_t GetAdjustedUseCnt() { return adjustedUseCnt_; }
  // Computer the adjusted use count. Update "adjustedUseCnt_".
  void ComputeAdjustedUseCnt(SchedInstruction *inst);

  int16_t CmputLastUseCnt();
  int16_t GetLastUseCnt() { return lastUseCnt_; }

  InstType GetCrtclPathFrmRoot() { return crtclPathFrmRoot_; }

  friend class SchedRange;

protected:
  // The "name" of this instruction. Usually a string indicating its type.
  string name_;
  // The mnemonic of this instruction, e.g. "add" or "jmp".
  string opCode_;
  // A numberical ID for this instruction.
  int nodeID_;
  // The type of this instruction.
  InstType instType_;
  // The issue type of this instruction.
  IssueType issuType_;
  // The order of this instruction in the input file's schedule.
  InstCount fileSchedOrder_;
  // The issue cycle of this instruction in the input file's schedule.
  InstCount fileSchedCycle_;

  // The number of predecessors of this instruction.
  InstCount prdcsrCnt_;
  // The number of successors of this instruction.
  InstCount scsrCnt_;

  // The minimum cycle in which this instruction can be scheduled, given its
  // data and resource constraints.
  InstCount frwrdLwrBound_;

  // The maximum cycle (measured backwards relative to the leaf node) in which
  // this instruction can be scheduled, given its data and resource
  // constraints. The leaf node's upper bound in this scheme is always zero
  // and the root node's upper bound is always equal to the leaf node's lower
  // bound.
  InstCount bkwrdLwrBound_;

  // The absolute forward lower bound of this instruction.
  InstCount abslutFrwrdLwrBound_;
  //  The absolute backward lower bound of this instruction.
  InstCount abslutBkwrdLwrBound_;

  // The critical-path distance of this instruction from the root.
  InstCount crtclPathFrmRoot_;

  // The critical-path distance of this instruction from the leaf.
  InstCount crtclPathFrmLeaf_;

  // Whether memory has been allocated for this instruction's data structures.
  bool memAllocd_;

  // The priority list of this instruction's predecessors, sorted by deadline
  // for relaxed scheduling.
  PriorityList<SchedInstruction> *sortedPrdcsrLst_;
  // The priority list of this instruction's successors, sorted by deadline
  // for relaxed scheduling.
  PriorityList<SchedInstruction> *sortedScsrLst_;

  /***************************************************************************
   * Used during scheduling                                                  *
   ***************************************************************************/
  // Whether the instruction is currently in the Ready List.
  bool ready_;
  // Each entry in this array holds the cycle in which this instruction will
  // become partially ready by satisfying the dependence of one predecessor.
  // For a predecessor that has not been scheduled the corresponding entry is
  // set to -1.
  InstCount *rdyCyclePerPrdcsr_;
  // A lower bound on the cycle in which this instruction will be ready. This
  // is the maximum entry in the "readyCyclePerPrdcsr_" array. When all
  // predecessors have been scheduled, this value gives the cycle in which
  // this instruction will actually become ready.
  InstCount minRdyCycle_;
  // The previous value of the minRdyCycle_, saved before the scheduling of a
  // predecessor to enable backtracking if this predecessor is unscheduled.
  InstCount *prevMinRdyCyclePerPrdcsr_;
  // An array of predecessor latencies indexed by predecessor number.
  InstCount *ltncyPerPrdcsr_;
  // The number of unscheduled predecessors.
  InstCount unschduldPrdcsrCnt_;
  // The number of unscheduled successors.
  InstCount unschduldScsrCnt_;
  /***************************************************************************/

  // The cycle in which this instruction is currently scheduled.
  InstCount crntSchedCycle_;
  // The slot in which this instruction is currently scheduled.
  InstCount crntSchedSlot_;
  // TODO(ghassan): Document.
  InstCount crntRlxdCycle_;

  // The lower bound, as read from the input file (if any).
  InstCount fileLwrBound_;
  // The upper bound, as read from the input file (if any).
  InstCount fileUprBound_;

  // The range of dynamic lower bounds in both directions at the current time
  // during the scheduling process, considering the predecessors that have
  // been scheduled already. These bounds should be larger than the permanent
  // lower bound (range tightening).
  SchedRange *crntRange_;

  // The instruction's signature, used by the enumerator's history table to
  // keep track of partial schedules.
  InstSignature sig_;

  // The cycle (if any) in which the instruction had been fixed before the
  // scheduling process started.
  InstCount preFxdCycle_;

  /***************************************************************************
   * Recursive lower bounds                                                  *
   ***************************************************************************/
  // The critical-path distances from recursive successors to be used in
  // recursive lower bound computations.
  InstCount *crtclPathFrmRcrsvScsr_;
  // The critical-path distances from recursive predecessors to be used in
  // recursive lower bound computations.
  InstCount *crtclPathFrmRcrsvPrdcsr_;
  /***************************************************************************/

  /***************************************************************************
   * Used for BB-Spill scheduling                                            *
   ***************************************************************************/
  // The registers defined by this instruction node.
  Register *defs_[MAX_DEFS_PER_INSTR];
  // The number of elements in defs.
  int16_t defCnt_;
  // The registers used by this instruction node.
  Register *uses_[MAX_USES_PER_INSTR];
  // The number of elements in uses.
  int16_t useCnt_;
  // The number of uses minus live-out registers. Live-out registers are uses
  // in the artifical leaf instruction.
  int16_t adjustedUseCnt_;
  // The number of live virtual registers for which this instruction is
  // the last use. This value changes dynamically during scheduling
  int16_t lastUseCnt_;
  /***************************************************************************/

  // Whether this instruction blocks its cycle, i.e. does not allow other
  // instructions to be executed during the same cycle.
  bool blksCycle_;
  // Whether this instruction is pipelined.
  bool pipelined_;

  bool mustBeInBBEntry_;
  bool mustBeInBBExit_;

  // TODO(ghassan): Document.
  InstCount CmputCrtclPath_(DIRECTION dir, SchedInstruction *ref = NULL);
  // Allocate the memory needed for data structures used in this node.
  // Arguments as follows:
  //   instCnt: The maximum number of instructions in the graph.
  //   isCP_FromScsr: Whether this instruction will keep track of critical
  //     paths from successors.
  //   isCP_FromPrdcsr: Whether this instruction will keep track of critical
  //     paths from predecessors.
  void AllocMem_(InstCount instCnt, bool isCP_FromScsr, bool isCP_FromPrdcsr);
  // Deallocates the memory used by the node's data structures.
  void DeAllocMem_();
  // Sets the predecessor order numbers on the edges between this node and its
  // predecessors.
  void SetPrdcsrNums_();
  // Sets the successor order numbers on the edges between this node and its
  // successors.
  void SetScsrNums_();
  // Computer the adjusted use count. Update "adjustedUseCnt_".
  void ComputeAdjustedUseCnt_();
};

// A class to keep track of dynamic SchedInstruction lower bounds, i.e. lower
// bounds which are tightened during enumeration based on the constraints
// imposed by the enumerator's decisions. This differs from bounds defined in
// SchedInstruction, which are static lower bounds computed before enumerations
// starts.
class SchedRange {
public:
  // Creates a scheduling range for a given instruction.
  SchedRange(SchedInstruction *inst);

  // Sets the range's boudns to the given values. If the range is then
  // "fixed" with respect to schedLngth, adds its instruction to fxdLst.
  bool SetBounds(InstCount frwrdLwrBound, InstCount bkwrdLwrBound,
                 InstCount schedLngth, LinkedList<SchedInstruction> *fxdLst);
  // Sets the range's boudns to the given values.
  void SetBounds(InstCount frwrdLwrBound, InstCount bkwrdLwrBound);

  // Sets the forward bound of the range.
  void SetFrwrdBound(InstCount bound);
  // Sets the backward bound of the range.
  void SetBkwrdBound(InstCount bound);

  // Tightens the lower bound of this range to the given new lower bound if it
  // is greater than the current lower bound. Any tightened instruction is
  // added to the given list to be used for efficient untightening. This
  // function returns with false as soon as infeasiblity (w.r.t the given
  // schedule length) is detected, otherwise it returns true.
  bool TightnLwrBound(DIRECTION dir, InstCount newLwrBound,
                      LinkedList<SchedInstruction> *tightndLst,
                      LinkedList<SchedInstruction> *fxdLst, bool enforce);
  // Like TightnLwrBound(), but also recursively propagates tightening through
  // the subgraph rooted at the instruction using this range.
  bool TightnLwrBoundRcrsvly(DIRECTION dir, InstCount newLwrBound,
                             LinkedList<SchedInstruction> *tightndLst,
                             LinkedList<SchedInstruction> *fxdLst,
                             bool enforce);

  // Returns the forward or backward lower bound of this range.
  InstCount GetLwrBound(DIRECTION dir) const;
  // Sets the forward or backward lower bound of this range.
  void SetLwrBound(DIRECTION dir, InstCount bound);
  // Returns the deadline cycle for this range.
  InstCount GetDeadline() const;
  // TODO(ghassan): Document.
  bool IsFxd() const;
  // Untightens any tightened lower bound.
  void UnTightnLwrBounds();
  // Marks the range as not tightened.
  void CmtLwrBoundTightnng();
  // TODO(ghassan): Document.
  bool Fix(InstCount cycle, LinkedList<SchedInstruction> *tightndLst,
           LinkedList<SchedInstruction> *fxdLst);
  // Returns whether the range is tightened in the given direction.
  bool IsTightnd(DIRECTION dir) const;

protected:
  // The forward lower bound.
  // TODO(ghassan): Elaborate.
  InstCount frwrdLwrBound_;
  // The backward lower bound.
  // TODO(ghassan): Elaborate.
  InstCount bkwrdLwrBound_;

  // The last cycle number in the current target length. It is equal to the
  // target schedule length minus one.
  InstCount lastCycle_;

  // The previous values of the forward lower bound. Used for backtracking.
  InstCount prevFrwrdLwrBound_;
  // The previous values of the backward lower bound. Used for backtracking.
  InstCount prevBkwrdLwrBound_;

  // A flag indicating whether the forward lower bound has been tightened.
  bool isFrwrdTightnd_;
  // A flag indicating whether the backward lower bound has been tightened.
  bool isBkwrdTightnd_;

  // Whether the current range is equal to exactly one cycle, or,
  // equivalently, whether the sum of bounds in both directions is equal to
  // the current target length.
  bool isFxd_;

  // A pointer to the instruction that owns this range.
  SchedInstruction *inst_;

  // Returns the sum of the range's forward and backward lower bounds.
  InstCount GetLwrBoundSum_() const;
  // Returns whether the bounds of this range may produce a feasible schedule.
  bool IsFsbl_() const;
  // Initializes the range members to a default state.
  void InitVars_();
};

} // namespace opt_sched
} // namespace llvm

#endif
