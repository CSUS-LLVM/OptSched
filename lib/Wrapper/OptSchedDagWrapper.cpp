/*******************************************************************************
Description:  A wrapper that convert an LLVM ScheduleDAG to an OptSched
              DataDepGraph.
*******************************************************************************/

#include "OptSchedDagWrapper.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdio>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

#define DEBUG_TYPE "optsched"

using namespace opt_sched;
using namespace llvm;

static std::unique_ptr<LLVMRegTypeFilter> createLLVMRegTypeFilter(
    const MachineModel *MM, const llvm::TargetRegisterInfo *TRI,
    const std::vector<unsigned> &RegionPressure, float RegFilterFactor = .7f) {
  return std::unique_ptr<LLVMRegTypeFilter>(
      new LLVMRegTypeFilter(MM, TRI, RegionPressure, RegFilterFactor));
}

LLVMDataDepGraph::LLVMDataDepGraph(
    MachineSchedContext *context, ScheduleDAGOptSched *llvmDag,
    LLVMMachineModel *machMdl, LATENCY_PRECISION ltncyPrcsn,
    MachineBasicBlock *BB, GraphTransTypes graphTransTypes,
    const std::vector<unsigned> &RegionPressure, bool treatOrderDepsAsDataDeps,
    int maxDagSizeForPrcisLtncy, int regionNum)
    : DataDepGraph(machMdl, ltncyPrcsn, graphTransTypes),
      llvmNodes_(llvmDag->SUnits), context_(context), schedDag_(llvmDag),
      target_(llvmDag->TM), RegionPressure(RegionPressure), RTFilter(nullptr) {
  llvmMachMdl_ = machMdl;
  dagFileFormat_ = DFF_BB;
  isTraceFormat_ = false;
  ltncyPrcsn_ = ltncyPrcsn;
  treatOrderDepsAsDataDeps_ = treatOrderDepsAsDataDeps;
  maxDagSizeForPrcisLtncy_ = maxDagSizeForPrcisLtncy;
  includesNonStandardBlock_ = false;
  includesUnsupported_ = false;
  includesCall_ = false;
  ShouldFilterRegisterTypes = SchedulerOptions::getInstance().GetBool(
      "FILTER_REGISTERS_TYPES_WITH_LOW_PRP", false);
  ShouldGenerateMM =
      SchedulerOptions::getInstance().GetBool("GENERATE_MACHINE_MODEL", false);

  includesUnpipelined_ = true;

  if (ShouldFilterRegisterTypes)
    RTFilter = createLLVMRegTypeFilter(machMdl, schedDag_->TRI, RegionPressure);

  // The extra 2 are for the artifical root and leaf nodes.
  instCnt_ = nodeCnt_ = llvmNodes_.size() + 2;

  std::snprintf(dagID_, MAX_NAMESIZE, "%s:%d",
                context_->MF->getFunction().getName().data(), regionNum);
  std::snprintf(compiler_, MAX_NAMESIZE, "LLVM");

  AllocArrays_(instCnt_);

  ConvertLLVMNodes_();

  if (Finish_() == RES_ERROR)
    Logger::Fatal("DAG Finish_() failed.");
}

void LLVMDataDepGraph::ConvertLLVMNodes_() {
  LLVM_DEBUG(dbgs() << "Building opt_sched DAG\n");

  // Create nodes.
  for (size_t i = 0; i < llvmNodes_.size(); i++) {
    const SUnit &SU = llvmNodes_[i];
    assert(SU.NodeNum == i && "Nodes must be numbered sequentially!");

    convertSUnit(SU);
  }

  // Create edges.
  for (const auto &SU : llvmNodes_) {
    convertEdges(SU);
  }

  // Add artificial root and leaf nodes and edges.
  setupRoot();
  setupLeaf();
}

void LLVMDataDepGraph::CountDefs(RegisterFile regFiles[]) {
  std::vector<int> regDefCounts(machMdl_->GetRegTypeCnt());
  // Track all regs that are defined.
  std::set<unsigned> defs;
  // Should we add uses that have no definition.
  bool addUsedAndNotDefined =
      SchedulerOptions::getInstance().GetBool("ADD_USED_AND_NOT_DEFINED_REGS");
  // Should we add live-out registers that have no definition.
  bool addLiveOutAndNotDefined = SchedulerOptions::getInstance().GetBool(
      "ADD_LIVE_OUT_AND_NOT_DEFINED_REGS");

  // count live-in as defs in root node
  for (const RegisterMaskPair &L : schedDag_->getRegPressure().LiveInRegs) {
    unsigned resNo = L.RegUnit;

    std::vector<int> regTypes = GetRegisterType_(resNo);
    for (int regType : regTypes)
      regDefCounts[regType]++;

    if (addUsedAndNotDefined)
      defs.insert(resNo);
  }

  for (std::vector<SUnit>::iterator it = llvmNodes_.begin();
       it != llvmNodes_.end(); ++it) {
    MachineInstr *MI = it->getInstr();
    // Get all defs for this instruction
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *schedDag_->TRI, schedDag_->MRI, true, false);

    // If a register is used but not defined prepare to add def as live-in.
    if (addUsedAndNotDefined) {
      for (const RegisterMaskPair &U : RegOpers.Uses) {
        unsigned resNo = U.RegUnit;
        if (!defs.count(resNo)) {
          std::vector<int> regTypes = GetRegisterType_(resNo);
          for (int regType : regTypes)
            regDefCounts[regType]++;

          defs.insert(resNo);
        }
      }
    }

    // Allocate defs
    for (const RegisterMaskPair &D : RegOpers.Defs) {
      unsigned resNo = D.RegUnit;
      std::vector<int> regTypes = GetRegisterType_(resNo);
      for (int regType : regTypes)
        regDefCounts[regType]++;

      if (addUsedAndNotDefined)
        defs.insert(resNo);
    }
  }

  // Get region end instruction if it is not a sentinel value
  const MachineInstr *MI = schedDag_->getRegionEnd();
  if (MI)
    countBoundaryLiveness(regDefCounts, defs, addUsedAndNotDefined, MI);

  if (addLiveOutAndNotDefined) {
    for (const RegisterMaskPair &O : schedDag_->getRegPressure().LiveOutRegs) {
      unsigned resNo = O.RegUnit;
      if (!defs.count(resNo)) {
        std::vector<int> regTypes = GetRegisterType_(resNo);
        for (int regType : regTypes)
          regDefCounts[regType]++;
      }
    }
  }

  for (int i = 0; i < machMdl_->GetRegTypeCnt(); i++) {
#ifndef NDEBUG
    if (regDefCounts[i])
      dbgs() << "Reg Type " << llvmMachMdl_->GetRegTypeName(i).c_str() << "->"
             << regDefCounts[i] << " registers\n";
#endif

    regFiles[i].SetRegCnt(regDefCounts[i]);
  }
}

void LLVMDataDepGraph::AddDefsAndUses(RegisterFile regFiles[]) {
  // The index of the last "assigned" register for each register type.
  regIndices_.resize(machMdl_->GetRegTypeCnt());

  // Add live in regs as defs for artificial root
  for (const RegisterMaskPair &I : schedDag_->getRegPressure().LiveInRegs) {
    AddLiveInReg_(I.RegUnit, regFiles);
  }

  std::vector<SUnit>::iterator startNode;
  for (startNode = llvmNodes_.begin(); startNode != llvmNodes_.end();
       ++startNode) {
    // The machine instruction we are processing
    MachineInstr *MI = startNode->getInstr();

    // Collect def/use information for this machine instruction
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *schedDag_->TRI, schedDag_->MRI, true, false);

    // add uses
    for (const RegisterMaskPair &U : RegOpers.Uses) {
      AddUse_(U.RegUnit, startNode->NodeNum, regFiles);
    }

    // add defs
    for (const RegisterMaskPair &D : RegOpers.Defs) {
      AddDef_(D.RegUnit, startNode->NodeNum, regFiles);
    }
  }

  // Get region end instruction if it is not a sentinel value
  const MachineInstr *MI = schedDag_->getRegionEnd();
  if (MI)
    discoverBoundaryLiveness(regFiles, MI);

  // add live-out registers as uses in artificial leaf instruction
  for (const RegisterMaskPair &O : schedDag_->getRegPressure().LiveOutRegs) {
    AddLiveOutReg_(O.RegUnit, regFiles);
  }

  // Check for any registers that are not used but are also not in LLVM's
  // live-out set.
  // Optionally, add these registers as uses in the aritificial leaf node.
  if (SchedulerOptions::getInstance().GetBool(
          "ADD_DEFINED_AND_NOT_USED_REGS")) {
    for (int16_t i = 0; i < machMdl_->GetRegTypeCnt(); i++) {
      for (int j = 0; j < regFiles[i].GetRegCnt(); j++) {
        Register *reg = regFiles[i].GetReg(j);
        if (reg->GetUseCnt() == 0) {
          AddDefAndNotUsed_(reg, regFiles);
        }
      }
    }
  }

  LLVM_DEBUG(schedDag_->dumpLLVMRegisters());
  LLVM_DEBUG(dumpRegisters(regFiles));
}

void LLVMDataDepGraph::AddUse_(unsigned resNo, InstCount nodeIndex,
                               RegisterFile regFiles[]) {
  bool addUsedAndNotDefined =
      SchedulerOptions::getInstance().GetBool("ADD_USED_AND_NOT_DEFINED_REGS");
  std::vector<int> resTypes = GetRegisterType_(resNo);

  if (addUsedAndNotDefined && lastDef_.find(resNo) == lastDef_.end()) {
    AddLiveInReg_(resNo, regFiles);
#ifdef IS_DEBUG_DEFS_AND_USES
    Logger::Info("Adding register that is used-and-not-defined.");
#endif
  }

  std::vector<Register *> regs = lastDef_[resNo];
  for (Register *reg : regs) {
    if (!insts_[nodeIndex]->FindUse(reg)) {
      insts_[nodeIndex]->AddUse(reg);
      reg->AddUse(insts_[nodeIndex]);
#ifdef IS_DEBUG_DEFS_AND_USES
      Logger::Info("Adding use for OptSched register: type: %lu number: "
                   "%lu  NodeNum: %lu",
                   reg->GetType(), reg->GetNum(), nodeIndex);
#endif
    }
  }
#ifdef IS_DEBUG_DEFS_AND_USES
  Logger::Info("Adding use for LLVM register: %lu NodeNum: %lu", resNo,
               nodeIndex);
#endif
}

void LLVMDataDepGraph::AddDef_(unsigned resNo, InstCount nodeIndex,
                               RegisterFile regFiles[]) {
  int weight = GetRegisterWeight_(resNo);
  std::vector<int> regTypes = GetRegisterType_(resNo);

  std::vector<Register *> regs;
  for (int regType : regTypes) {
    Register *reg = regFiles[regType].GetReg(regIndices_[regType]++);
    insts_[nodeIndex]->AddDef(reg);
    reg->SetWght(weight);
    reg->AddDef(insts_[nodeIndex]);
#ifdef IS_DEBUG_DEFS_AND_USES
    Logger::Info("Adding def for OptSched register: type: %lu number: %lu "
                 "NodeNum: %lu",
                 reg->GetType(), reg->GetNum(), nodeIndex);
#endif
    regs.push_back(reg);
  }
  lastDef_[resNo] = regs;

#ifdef IS_DEBUG_DEFS_AND_USES
  Logger::Info("Adding def for LLVM register: %lu NodeNum: %lu", resNo,
               nodeIndex);
#endif
}

void LLVMDataDepGraph::AddLiveInReg_(unsigned resNo, RegisterFile regFiles[]) {
  // index of root node in insts_
  int rootIndex = llvmNodes_.size();
  int weight = GetRegisterWeight_(resNo);
  std::vector<int> regTypes = GetRegisterType_(resNo);

  std::vector<Register *> regs;
  for (int regType : regTypes) {
    Register *reg = regFiles[regType].GetReg(regIndices_[regType]++);
    insts_[rootIndex]->AddDef(reg);
    reg->SetWght(weight);
    reg->AddDef(insts_[rootIndex]);
    reg->SetIsLiveIn(true);
#ifdef IS_DEBUG_DEFS_AND_USES
    Logger::Info("Adding live-in def for OptSched register: type: %lu "
                 "number: %lu NodeNum: %lu",
                 reg->GetType(), reg->GetNum(), rootIndex);
#endif
    regs.push_back(reg);
  }
  lastDef_[resNo] = regs;

#ifdef IS_DEBUG_DEFS_AND_USES
  Logger::Info("Adding live-in def for LLVM register: %lu NodeNum: %lu", resNo,
               rootIndex);
#endif
}

void LLVMDataDepGraph::AddLiveOutReg_(unsigned resNo, RegisterFile regFiles[]) {
  // Should we add live-out registers that have no definition.
  bool addLiveOutAndNotDefined = SchedulerOptions::getInstance().GetBool(
      "ADD_LIVE_OUT_AND_NOT_DEFINED_REGS");
  // index of leaf node in insts_
  int leafIndex = llvmNodes_.size() + 1;
  std::vector<int> regTypes = GetRegisterType_(resNo);

  if (addLiveOutAndNotDefined && lastDef_.find(resNo) == lastDef_.end()) {
    AddLiveInReg_(resNo, regFiles);
#ifdef IS_DEBUG_DEFS_AND_USES
    Logger::Info("Adding register that is live-out-and-not-defined.");
#endif
  }

  std::vector<Register *> regs = lastDef_[resNo];
  for (Register *reg : regs) {
    if (!insts_[leafIndex]->FindUse(reg)) {
      insts_[leafIndex]->AddUse(reg);
      reg->AddUse(insts_[leafIndex]);
      reg->SetIsLiveOut(true);
#ifdef IS_DEBUG_DEFS_AND_USES
      Logger::Info("Adding live-out use for OptSched register: type: %lu "
                   "number: %lu NodeNum: %lu",
                   reg->GetType(), reg->GetNum(), leafIndex);
#endif
    }
#ifdef IS_DEBUG_DEFS_AND_USES
    Logger::Info("Adding live-out use for register: %lu NodeNum: %lu", resNo,
                 leafIndex);
#endif
  }
}

void LLVMDataDepGraph::AddDefAndNotUsed_(Register *reg,
                                         RegisterFile regFiles[]) {
  // index of leaf node in insts_
  int leafIndex = llvmNodes_.size() + 1;
  if (!insts_[leafIndex]->FindUse(reg)) {
    insts_[leafIndex]->AddUse(reg);
    reg->AddUse(insts_[leafIndex]);
    reg->SetIsLiveOut(true);
#ifdef IS_DEBUG_DEFS_AND_USES
    Logger::Info("Adding live-out use for OptSched register: type: %lu "
                 "This register is not in the live-out set from LLVM"
                 "number: %lu NodeNum: %lu",
                 reg->GetType(), reg->GetNum(), leafIndex);
#endif
  }
}

int LLVMDataDepGraph::GetRegisterWeight_(const unsigned resNo) const {
  bool useSimpleTypes =
      SchedulerOptions::getInstance().GetBool("USE_SIMPLE_REGISTER_TYPES");
  // If using simple register types ignore PSet weight.
  if (useSimpleTypes)
    return 1;
  else {
    PSetIterator PSetI = schedDag_->MRI.getPressureSets(resNo);
    return PSetI.getWeight();
  }
}

// A register type is an int value that corresponds to a register type in our
// scheduler.
// We assign multiple register types to each register from LLVM to account
// for all register pressure sets associated with the register class for resNo.
std::vector<int>
LLVMDataDepGraph::GetRegisterType_(const unsigned resNo) const {
  bool useSimpleTypes =
      SchedulerOptions::getInstance().GetBool("USE_SIMPLE_REGISTER_TYPES");
  const TargetRegisterInfo &TRI = *schedDag_->TRI;
  std::vector<int> pSetTypes;

  PSetIterator PSetI = schedDag_->MRI.getPressureSets(resNo);

  // If we want to use simple register types return the first PSet.
  if (useSimpleTypes) {
    if (!PSetI.isValid())
      return pSetTypes;

    const char *pSetName = TRI.getRegPressureSetName(*PSetI);
    bool FilterOutRegType = ShouldFilterRegisterTypes && (*RTFilter)[pSetName];

    if (!FilterOutRegType) {
      int type = llvmMachMdl_->GetRegTypeByName(pSetName);
      pSetTypes.push_back(type);
    }

  } else {
    for (; PSetI.isValid(); ++PSetI) {
      const char *pSetName = TRI.getRegPressureSetName(*PSetI);
      bool FilterOutRegType =
          ShouldFilterRegisterTypes && (*RTFilter)[pSetName];

      if (!FilterOutRegType) {
        int type = llvmMachMdl_->GetRegTypeByName(pSetName);
        pSetTypes.push_back(type);
      }
    }
  }

  return pSetTypes;
}

SUnit *LLVMDataDepGraph::GetSUnit(size_t index) const {
  if (index < llvmNodes_.size()) {
    return &llvmNodes_[index];
  } else {
    // Artificial entry/exit node.
    return NULL;
  }
}

// Check if this is a root SU
bool LLVMDataDepGraph::isRootNode(const SUnit &SU) {
  for (SUnit::const_pred_iterator I = SU.Preds.begin(), E = SU.Preds.end();
       I != E; ++I) {
    if (I->getSUnit()->isBoundaryNode())
      continue;
    else
      return false;
  }
  return true;
}

// Check if this is a leaf SU
bool LLVMDataDepGraph::isLeafNode(const SUnit &SU) {
  for (SUnit::const_succ_iterator I = SU.Succs.begin(), E = SU.Succs.end();
       I != E; ++I) {
    if (I->getSUnit()->isBoundaryNode())
      continue;
    else
      return false;
  }
  return true;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void LLVMDataDepGraph::dumpRegisters(const RegisterFile regFiles[]) const {
  dbgs() << "Optsched Regsiters\n";

  auto RegTypeCount = machMdl_->GetRegTypeCnt();
  for (int16_t RegTypeNum = 0; RegTypeNum < RegTypeCount; RegTypeNum++) {
    const auto &RegFile = regFiles[RegTypeNum];
    // Skip register types that are not used/defined in the region
    if (RegFile.GetRegCnt() == 0)
      continue;

    const auto &RegTypeName = machMdl_->GetRegTypeName(RegTypeNum);
    for (int RegNum = 0; RegNum < RegFile.GetRegCnt(); RegNum++) {
      const auto *Reg = RegFile.GetReg(RegNum);
      dbgs() << "Register: " << '%' << Reg->GetNum() << " (" << RegTypeName
             << '/' << RegTypeNum << ")\n";

      typedef SmallPtrSet<const SchedInstruction *, 8>::const_iterator
          const_iterator;

      // Definitions for this register
      const auto &DefList = Reg->GetDefList();
      dbgs() << "\t--Defs:";
      for (const_iterator I = DefList.begin(), E = DefList.end(); I != E; ++I)
        dbgs() << " (" << (*I)->GetNodeID() << ") " << (*I)->GetOpCode();

      dbgs() << '\n';

      // Uses for this register
      const auto &UseList = Reg->GetUseList();
      dbgs() << "\t--Uses:";
      for (const_iterator I = UseList.begin(), E = UseList.end(); I != E; ++I)
        dbgs() << " (" << (*I)->GetNodeID() << ") " << (*I)->GetOpCode();

      dbgs() << "\n\n";
    }
  }
}
#endif

inline void LLVMDataDepGraph::setupRoot() {
  // Create artificial root.
  int rootNum = llvmNodes_.size();
  root_ =
      CreateNode_(rootNum, "artificial",
                  machMdl_->GetInstTypeByName("artificial"), "__optsched_entry",
                  rootNum, // nodeID
                  rootNum, // fileSchedOrder
                  rootNum, // fileSchedCycle
                  0,       // fileInstLwrBound
                  0,       // fileInstUprBound
                  0);      // blkNum
  // Add edges between root nodes in graph and optsched artificial root.
  for (size_t i = 0; i < llvmNodes_.size(); i++) {
    if (insts_[i]->GetPrdcsrCnt() == 0)
      CreateEdge_(rootNum, i, 0, DEP_OTHER);
  }
}

inline void LLVMDataDepGraph::setupLeaf() {
  // Create artificial leaf.
  int leafNum = llvmNodes_.size() + 1;
  CreateNode_(leafNum, "artificial", machMdl_->GetInstTypeByName("artificial"),
              "__optsched_exit",
              leafNum, // nodeID
              leafNum, // fileSchedOrder
              leafNum, // fileSchedCycle
              0,       // fileInstLwrBound
              0,       // fileInstUprBound
              0);      // blkNum

  // Add edges between leaf nodes in graph and optsched artificial leaf.
  for (size_t i = 0; i < llvmNodes_.size(); i++)
    if (insts_[i]->GetScsrCnt() == 0)
      CreateEdge_(i, leafNum, 0, DEP_OTHER);
}

void LLVMDataDepGraph::convertEdges(const SUnit &SU) {
  const MachineInstr *instr = SU.getInstr();
  SUnit::const_succ_iterator I, E;
  for (I = SU.Succs.begin(), E = SU.Succs.end(); I != E; ++I) {
    if (I->getSUnit()->isBoundaryNode())
      continue;

    DependenceType depType;
    switch (I->getKind()) {
    case SDep::Data:
      depType = DEP_DATA;
      break;
    case SDep::Anti:
      depType = DEP_ANTI;
      break;
    case SDep::Output:
      depType = DEP_OUTPUT;
      break;
    case SDep::Order:
      depType = treatOrderDepsAsDataDeps_ ? DEP_DATA : DEP_OTHER;
      break;
    }

    LATENCY_PRECISION prcsn = ltncyPrcsn_;
    if (prcsn == LTP_PRECISE && maxDagSizeForPrcisLtncy_ > 0 &&
        llvmNodes_.size() > static_cast<size_t>(maxDagSizeForPrcisLtncy_))
      prcsn = LTP_ROUGH; // use rough latencies if DAG is too large

    int16_t Latency;
    if (prcsn == LTP_PRECISE) { // get precise latency from the machine model
      const auto &InstName = schedDag_->TII->getName(instr->getOpcode());
      const auto &InstType = machMdl_->GetInstTypeByName(InstName);
      Latency = machMdl_->GetLatency(InstType, depType);
    } else if (prcsn == LTP_ROUGH) // rough latency = llvm latency
      Latency = I->getLatency();
    else
      Latency = 1;

    CreateEdge_(SU.NodeNum, I->getSUnit()->NodeNum, Latency, depType);
  }
}

void LLVMDataDepGraph::convertSUnit(const SUnit &SU) {
  InstType instType;
  std::string instName;
  std::string opCode;

  if (SU.isBoundaryNode() || !SU.isInstr())
    return;

  const MachineInstr *MI = SU.getInstr();
  instName = opCode = schedDag_->TII->getName(MI->getOpcode());

  // Search in the machine model for an instType with this OpCode name
  instType = machMdl_->GetInstTypeByName(instName.c_str());

  // If the machine model does not have an instruction type with this OpCode
  // name generate one. Alternatively if not generating types, use a default
  // type.
  if (instType == INVALID_INST_TYPE) {
    if (ShouldGenerateMM) {
      llvmMachMdl_->getMMGen()->generateInstrType(MI);
    } else {
      instName = "Default";
      instType = machMdl_->getDefaultInstType();
    }
  }

  CreateNode_(SU.NodeNum, instName.c_str(), instType, opCode.c_str(),
              SU.NodeNum, // nodeID
              SU.NodeNum, // fileSchedOrder
              SU.NodeNum, // fileSchedCycle
              0,          // fileInstLwrBound
              0,          // fileInstUprBound
              0);         // blkNum
}

void LLVMDataDepGraph::discoverBoundaryLiveness(RegisterFile registerFiles[],
                                                const MachineInstr *MI) {
  RegisterOperands RegOpers;
  RegOpers.collect(*MI, *schedDag_->TRI, schedDag_->MRI, true, false);

  int leafIndex = llvmNodes_.size() + 1;
  for (auto &U : RegOpers.Uses)
    AddUse_(U.RegUnit, leafIndex, registerFiles);

  for (auto &D : RegOpers.Defs)
    AddDef_(D.RegUnit, leafIndex, registerFiles);
}

void LLVMDataDepGraph::countBoundaryLiveness(std::vector<int> &RegDefCounts,
                                             std::set<unsigned> &Defs,
                                             bool AddUsedAndNotDefined,
                                             const MachineInstr *MI) {
  RegisterOperands RegOpers;
  RegOpers.collect(*MI, *schedDag_->TRI, schedDag_->MRI, true, false);

  for (auto &D : RegOpers.Defs) {
    std::vector<int> RegTypes = GetRegisterType_(D.RegUnit);
    for (int RegType : RegTypes)
      RegDefCounts[RegType]++;

    if (AddUsedAndNotDefined)
      Defs.insert(D.RegUnit);
  }
}

LLVMRegTypeFilter::LLVMRegTypeFilter(
    const MachineModel *MM, const llvm::TargetRegisterInfo *TRI,
    const std::vector<unsigned> &RegionPressure, float RegFilterFactor)
    : MM(MM), TRI(TRI), RegionPressure(RegionPressure),
      RegFilterFactor(RegFilterFactor) {
  FindPSetsToFilter();
}

void LLVMRegTypeFilter::FindPSetsToFilter() {
  for (unsigned i = 0, e = RegionPressure.size(); i < e; ++i) {
    const char *RegTypeName = TRI->getRegPressureSetName(i);
    int16_t RegTypeID = MM->GetRegTypeByName(RegTypeName);

    int RPLimit = MM->GetPhysRegCnt(RegTypeID);
    int MAXPR = RegionPressure[i];

    bool ShouldFilterType = MAXPR < RegFilterFactor * RPLimit;

    RegTypeIDFilteredMap[RegTypeID] = ShouldFilterType;
    RegTypeNameFilteredMap[RegTypeName] = ShouldFilterType;
  }
}

bool LLVMRegTypeFilter::operator[](const int16_t RegTypeID) const {
  assert(RegTypeIDFilteredMap.find(RegTypeID) != RegTypeIDFilteredMap.end() &&
         "Could not find RegTypeID!");

  return RegTypeIDFilteredMap.find(RegTypeID)->second;
}

bool LLVMRegTypeFilter::operator[](const char *RegTypeName) const {
  assert(RegTypeNameFilteredMap.find(RegTypeName) !=
             RegTypeNameFilteredMap.end() &&
         "Could not find RegTypeName!");

  return RegTypeNameFilteredMap.find(RegTypeName)->second;
}

bool LLVMRegTypeFilter::shouldFilter(int16_t RegTypeID) const {
  return (*this)[RegTypeID];
}

bool LLVMRegTypeFilter::shouldFilter(const char *RegTypeName) const {
  return (*this)[RegTypeName];
}

void LLVMRegTypeFilter::setRegFilterFactor(float RegFilterFactor) {
  this->RegFilterFactor = RegFilterFactor;
}
