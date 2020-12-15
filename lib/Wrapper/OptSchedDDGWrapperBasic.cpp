//===- OptSchedDDGWrapperBasic.cpp - Basic DDG Wrapper --------------------===//
//
// Target independent conversion from LLVM ScheduleDAG to OptSched DDG.
//
//===----------------------------------------------------------------------===//

#include "OptSchedDDGWrapperBasic.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdio>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>

#define DEBUG_TYPE "optsched-ddg-wrapper"

using namespace llvm;
using namespace llvm::opt_sched;

#ifndef NDEBUG
static Printable printOptSchedReg(const Register *Reg,
                                  const std::string &RegTypeName,
                                  int16_t RegTypeNum);
#endif

static std::unique_ptr<LLVMRegTypeFilter> createLLVMRegTypeFilter(
    const MachineModel *MM, const llvm::TargetRegisterInfo *TRI,
    const std::vector<unsigned> &RegionPressure, float RegFilterFactor = .7f) {

  return std::unique_ptr<LLVMRegTypeFilter>(
      new LLVMRegTypeFilter(MM, TRI, RegionPressure, RegFilterFactor));
}

OptSchedDDGWrapperBasic::OptSchedDDGWrapperBasic(
    MachineSchedContext *Context, ScheduleDAGOptSched *DAG,
    OptSchedMachineModel *MM, LATENCY_PRECISION LatencyPrecision,
    const std::string &RegionID)
    : DataDepGraph(MM, LatencyPrecision), MM(MM), Contex(Context), DAG(DAG),
      RTFilter(nullptr) {
  dagFileFormat_ = DFF_BB;
  isTraceFormat_ = false;
  TreatOrderDepsAsDataDeps =
      SchedulerOptions::getInstance().GetBool("TREAT_ORDER_DEPS_AS_DATA_DEPS");
  ShouldFilterRegisterTypes = SchedulerOptions::getInstance().GetBool(
      "FILTER_REGISTERS_TYPES_WITH_LOW_PRP", false);
  ShouldGenerateMM =
      SchedulerOptions::getInstance().GetBool("GENERATE_MACHINE_MODEL", false);
  includesNonStandardBlock_ = false;
  includesUnsupported_ = false;
  includesCall_ = false;
  includesUnpipelined_ = true;
  strncpy(dagID_, RegionID.c_str(), sizeof(dagID_));
  strncpy(compiler_, "LLVM", sizeof(compiler_));

  if (ShouldFilterRegisterTypes)
    RTFilter = createLLVMRegTypeFilter(MM, DAG->TRI,
                                       DAG->getRegPressure().MaxSetPressure);
}

void OptSchedDDGWrapperBasic::convertSUnits(bool IgnoreRealEdges,
                                            bool IgnoreArtificialEdges) {
  LLVM_DEBUG(dbgs() << "Building opt_sched DAG\n");
  // The extra 2 are for the artifical root and leaf nodes.
  instCnt_ = nodeCnt_ = DAG->SUnits.size() + 2;
  AllocArrays_(instCnt_);

  // Create nodes.
  for (size_t i = 0; i < DAG->SUnits.size(); i++) {
    const SUnit &SU = DAG->SUnits[i];
    assert(SU.NodeNum == i && "Nodes must be numbered sequentially!");

    convertSUnit(SU);
  }

  // Create edges.
  for (const auto &SU : DAG->SUnits) {
    convertEdges(SU, IgnoreRealEdges, IgnoreArtificialEdges);
  }

  // Add artificial root and leaf nodes and edges.
  setupRoot();
  setupLeaf();

  if (Finish_() == RES_ERROR)
    llvm::report_fatal_error("DAG Finish_() failed.", false);
}

void OptSchedDDGWrapperBasic::convertRegFiles() {
  for (int i = 0; i < MM->GetRegTypeCnt(); i++)
    RegFiles[i].SetRegType(i);

  countDefs();
  addDefsAndUses();
}

void OptSchedDDGWrapperBasic::countDefs() {
  std::vector<int> RegDefCounts(MM->GetRegTypeCnt());
  // Track all regs that are defined.
  std::set<unsigned> Defs;

  // count live-in as defs in root node
  for (const auto &L : DAG->getRegPressure().LiveInRegs) {
    for (int Type : getRegisterType(L.RegUnit))
      RegDefCounts[Type]++;
    Defs.insert(L.RegUnit);
  }

  std::vector<SUnit>::const_iterator I, E;
  for (I = DAG->SUnits.begin(), E = DAG->SUnits.end(); I != E; ++I) {
    MachineInstr *MI = I->getInstr();
    // Get all defs for this instruction
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *DAG->TRI, DAG->MRI, true, false);

    for (const auto &U : RegOpers.Uses) {
      // If this register is not defined, add it as live-in.
      if (!Defs.count(U.RegUnit)) {
        for (int Type : getRegisterType(U.RegUnit))
          RegDefCounts[Type]++;

        Defs.insert(U.RegUnit);
      }
    }

    // Allocate defs
    for (const auto &D : RegOpers.Defs) {
      for (int Type : getRegisterType(D.RegUnit))
        RegDefCounts[Type]++;
      Defs.insert(D.RegUnit);
    }
  }

  // Get region end instruction if it is not a sentinel value
  const MachineInstr *MI = DAG->getRegionEnd();
  if (MI)
    countBoundaryLiveness(RegDefCounts, Defs, MI);

  for (const RegisterMaskPair &O : DAG->getRegPressure().LiveOutRegs)
    if (!Defs.count(O.RegUnit))
      for (int Type : getRegisterType(O.RegUnit))
        RegDefCounts[Type]++;

  for (int i = 0; i < MM->GetRegTypeCnt(); i++) {
    LLVM_DEBUG(if (RegDefCounts[i]) dbgs()
                   << "Reg Type " << MM->GetRegTypeName(i).c_str() << "->"
                   << RegDefCounts[i] << " registers\n";);

    RegFiles[i].SetRegCnt(RegDefCounts[i]);
  }
}

void OptSchedDDGWrapperBasic::addDefsAndUses() {
  // The index of the last "assigned" register for each register type.
  RegIndices.resize(MM->GetRegTypeCnt());

  // Add live in regs as defs for artificial root
  for (const auto &I : DAG->getRegPressure().LiveInRegs)
    addLiveInReg(I.RegUnit);

  std::vector<SUnit>::const_iterator I, E;
  for (I = DAG->SUnits.begin(), E = DAG->SUnits.end(); I != E; ++I) {
    MachineInstr *MI = I->getInstr();
    RegisterOperands RegOpers;
    RegOpers.collect(*MI, *DAG->TRI, DAG->MRI, true, false);

    for (const auto &U : RegOpers.Uses)
      addUse(U.RegUnit, I->NodeNum);

    for (const auto &D : RegOpers.Defs)
      addDef(D.RegUnit, I->NodeNum);
  }

  // Get region end instruction if it is not a sentinel value
  const MachineInstr *MI = DAG->getRegionEnd();
  if (MI)
    discoverBoundaryLiveness(MI);

  // add live-out registers as uses in artificial leaf instruction
  for (const RegisterMaskPair &O : DAG->getRegPressure().LiveOutRegs)
    addLiveOutReg(O.RegUnit);

  // Check for any registers that are not used but are also not in LLVM's
  // live-out set.
  // Optionally, add these registers as uses in the artificial leaf node.
  for (int16_t i = 0; i < MM->GetRegTypeCnt(); i++)
    for (int j = 0; j < RegFiles[i].GetRegCnt(); j++) {
      Register *Reg = RegFiles[i].GetReg(j);
      if (Reg->GetUseCnt() == 0)
        addDefAndNotUsed(Reg);
    }

  LLVM_DEBUG(DAG->dumpLLVMRegisters());
  LLVM_DEBUG(dumpOptSchedRegisters());
}

void OptSchedDDGWrapperBasic::addUse(unsigned RegUnit, InstCount Index) {
  if (LastDef.find(RegUnit) == LastDef.end()) {
    addLiveInReg(RegUnit);

    LLVM_DEBUG(dbgs() << "Adding register that is used-and-not-defined: ");
    LLVM_DEBUG(TargetRegisterInfo::dumpReg(RegUnit, 0, DAG->TRI));
  }

  for (Register *Reg : LastDef[RegUnit]) {
    insts_[Index]->AddUse(Reg);
    Reg->AddUse(insts_[Index]);
  }
}

void OptSchedDDGWrapperBasic::addDef(unsigned RegUnit, InstCount Index) {
  std::vector<Register *> Regs;
  for (int Type : getRegisterType(RegUnit)) {
    Register *Reg = RegFiles[Type].GetReg(RegIndices[Type]++);
    insts_[Index]->AddDef(Reg);
    Reg->SetWght(getRegisterWeight(RegUnit));
    Reg->AddDef(insts_[Index]);
    Regs.push_back(Reg);
  }
  LastDef[RegUnit] = Regs;
}

void OptSchedDDGWrapperBasic::addLiveInReg(unsigned RegUnit) {
  std::vector<Register *> Regs;
  for (int Type : getRegisterType(RegUnit)) {
    Register *Reg = RegFiles[Type].GetReg(RegIndices[Type]++);
    GetRootInst()->AddDef(Reg);
    Reg->SetWght(getRegisterWeight(RegUnit));
    Reg->AddDef(GetRootInst());
    Reg->SetIsLiveIn(true);
    Regs.push_back(Reg);
  }
  LastDef[RegUnit] = Regs;
}

void OptSchedDDGWrapperBasic::addLiveOutReg(unsigned RegUnit) {
  // Add live-out registers that have no definition.
  if (LastDef.find(RegUnit) == LastDef.end()) {
    addLiveInReg(RegUnit);

    LLVM_DEBUG(dbgs() << "Adding register that is live-out-and-not-defined: ");
    LLVM_DEBUG(TargetRegisterInfo::dumpReg(RegUnit, 0, DAG->TRI));
  }

  auto LeafInstr = insts_[DAG->SUnits.size() + 1];
  std::vector<Register *> Regs = LastDef[RegUnit];
  for (Register *Reg : Regs) {
    LeafInstr->AddUse(Reg);
    Reg->AddUse(LeafInstr);
    Reg->SetIsLiveOut(true);
  }
}

void OptSchedDDGWrapperBasic::addDefAndNotUsed(Register *Reg) {
  int LeafIndex = DAG->SUnits.size() + 1;
  auto LeafInstr = insts_[LeafIndex];
  if (!LeafInstr->FindUse(Reg)) {
    LeafInstr->AddUse(Reg);
    Reg->AddUse(LeafInstr);
    Reg->SetIsLiveOut(true);

    LLVM_DEBUG(dbgs() << "Adding register that is defined and not used: ");
    LLVM_DEBUG(dbgs() << printOptSchedReg(
                   Reg, MM->GetRegTypeName(Reg->GetType()), Reg->GetType()));
  }
}

int OptSchedDDGWrapperBasic::getRegisterWeight(unsigned RegUnit) const {
  bool useSimpleTypes =
      SchedulerOptions::getInstance().GetBool("USE_SIMPLE_REGISTER_TYPES");
  if (useSimpleTypes)
    return 1;
  else {
    PSetIterator PSetI = DAG->MRI.getPressureSets(RegUnit);
    return PSetI.getWeight();
  }
}

// A register type is an int value that corresponds to a register type in our
// scheduler.
// We assign multiple register types to each register from LLVM to account
// for all register pressure sets associated with the register class for resNo.
std::vector<int>
OptSchedDDGWrapperBasic::getRegisterType(unsigned RegUnit) const {
  std::vector<int> RegTypes;
  PSetIterator PSetI = DAG->MRI.getPressureSets(RegUnit);

  bool UseSimpleTypes =
      SchedulerOptions::getInstance().GetBool("USE_SIMPLE_REGISTER_TYPES");
  // If we want to use simple register types return the first PSet.
  if (UseSimpleTypes) {
    if (!PSetI.isValid())
      return RegTypes;

    const char *PSetName = DAG->TRI->getRegPressureSetName(*PSetI);
    bool FilterOutRegType = ShouldFilterRegisterTypes && (*RTFilter)[PSetName];
    if (!FilterOutRegType)
      RegTypes.push_back(MM->GetRegTypeByName(PSetName));
  } else {
    for (; PSetI.isValid(); ++PSetI) {
      const char *PSetName = DAG->TRI->getRegPressureSetName(*PSetI);
      bool FilterOutRegType =
          ShouldFilterRegisterTypes && (*RTFilter)[PSetName];
      if (!FilterOutRegType)
        RegTypes.push_back(MM->GetRegTypeByName(PSetName));
    }
  }
  return RegTypes;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static Printable printOptSchedReg(const Register *Reg,
                                  const std::string &RegTypeName,
                                  int16_t RegTypeNum) {
  return Printable([Reg, &RegTypeName, RegTypeNum](raw_ostream &OS) {
    OS << "Register: " << '%' << Reg->GetNum() << " (" << RegTypeName << '/'
       << RegTypeNum << ")\n";

    typedef SmallPtrSet<const SchedInstruction *, 8>::const_iterator
        const_iterator;

    // Definitions for this register
    const auto &DefList = Reg->GetDefList();
    OS << "\t--Defs:";
    for (const_iterator I = DefList.begin(), E = DefList.end(); I != E; ++I)
      OS << " (" << (*I)->GetNodeID() << ") " << (*I)->GetOpCode();

    OS << '\n';

    // Uses for this register
    const auto &UseList = Reg->GetUseList();
    OS << "\t--Uses:";
    for (const_iterator I = UseList.begin(), E = UseList.end(); I != E; ++I)
      OS << " (" << (*I)->GetNodeID() << ") " << (*I)->GetOpCode();

    OS << "\n\n";
  });
}

LLVM_DUMP_METHOD
void OptSchedDDGWrapperBasic::dumpOptSchedRegisters() const {
  dbgs() << "Optsched Regsiters\n";

  auto RegTypeCount = MM->GetRegTypeCnt();
  for (int16_t RegTypeNum = 0; RegTypeNum < RegTypeCount; RegTypeNum++) {
    const auto &RegFile = RegFiles[RegTypeNum];
    // Skip register types that have no registers in the region
    if (RegFile.GetRegCnt() == 0)
      continue;

    const auto &RegTypeName = MM->GetRegTypeName(RegTypeNum);
    for (int RegNum = 0; RegNum < RegFile.GetRegCnt(); RegNum++) {
      auto *Reg = RegFile.GetReg(RegNum);
      dbgs() << printOptSchedReg(Reg, RegTypeName, RegTypeNum);
    }
  }
}
#endif

inline void OptSchedDDGWrapperBasic::setupRoot() {
  // Create artificial root.
  int RootNum = DAG->SUnits.size();
  root_ = CreateNode_(RootNum, "artificial",
                      MM->GetInstTypeByName("artificial"), "__optsched_entry",
                      RootNum, // nodeID
                      RootNum, // fileSchedOrder
                      RootNum, // fileSchedCycle
                      0,       // fileInstLwrBound
                      0,       // fileInstUprBound
                      0);      // blkNum

  // Add edges between root nodes in graph and optsched artificial root.
  for (size_t i = 0; i < DAG->SUnits.size(); i++)
    if (insts_[i]->GetPrdcsrCnt() == 0)
      CreateEdge_(RootNum, i, 0, DEP_OTHER);
}

inline void OptSchedDDGWrapperBasic::setupLeaf() {
  // Create artificial leaf.
  int LeafNum = DAG->SUnits.size() + 1;
  CreateNode_(LeafNum, "artificial", MM->GetInstTypeByName("artificial"),
              "__optsched_exit",
              LeafNum, // nodeID
              LeafNum, // fileSchedOrder
              LeafNum, // fileSchedCycle
              0,       // fileInstLwrBound
              0,       // fileInstUprBound
              0);      // blkNum

  // Add edges between leaf nodes in graph and optsched artificial leaf.
  for (size_t i = 0; i < DAG->SUnits.size(); i++)
    if (insts_[i]->GetScsrCnt() == 0)
      CreateEdge_(i, LeafNum, 0, DEP_OTHER);
}

void OptSchedDDGWrapperBasic::addArtificialEdges() {
  for (const auto &SU : DAG->SUnits) {
    convertEdges(SU, true, false);
  }
}

void OptSchedDDGWrapperBasic::convertEdges(const SUnit &SU,
                                           bool IgnoreRealEdges,
                                           bool IgnoreArtificialEdges) {
  const MachineInstr *instr = SU.getInstr();
  SUnit::const_succ_iterator I, E;
  for (I = SU.Succs.begin(), E = SU.Succs.end(); I != E; ++I) {
    if (I->getSUnit()->isBoundaryNode())
      continue;

    bool IsArtificial = I->isArtificial() || I->isCluster();
    if (IgnoreArtificialEdges && IsArtificial)
      continue;
    else if (IgnoreRealEdges && !IsArtificial)
      continue;

    DependenceType DepType;
    switch (I->getKind()) {
    case SDep::Data:
      DepType = DEP_DATA;
      break;
    case SDep::Anti:
      DepType = DEP_ANTI;
      break;
    case SDep::Output:
      DepType = DEP_OUTPUT;
      break;
    case SDep::Order:
      DepType = TreatOrderDepsAsDataDeps ? DEP_DATA : DEP_OTHER;
      break;
    }

    int16_t Latency;
    if (ltncyPrcsn_ == LTP_PRECISE) { // get latency from the machine model
      const auto &InstName = DAG->TII->getName(instr->getOpcode());
      const auto &InstType = MM->GetInstTypeByName(InstName);
      Latency = MM->GetLatency(InstType, DepType);
    } else if (ltncyPrcsn_ == LTP_ROUGH) // rough latency = llvm latency
      Latency = I->getLatency();
    else
      Latency = 1; // unit latency = ignore ilp

    CreateEdge_(SU.NodeNum, I->getSUnit()->NodeNum, Latency, DepType,
                IsArtificial);
  }
}

void OptSchedDDGWrapperBasic::convertSUnit(const SUnit &SU) {
  InstType InstType;
  std::string InstName;
  if (SU.isBoundaryNode() || !SU.isInstr())
    return;

  const MachineInstr *MI = SU.getInstr();
  InstName = DAG->TII->getName(MI->getOpcode());

  // Search in the machine model for an instType with this OpCode name
  InstType = MM->GetInstTypeByName(InstName.c_str());

  // If the machine model does not have an instruction type with this OpCode
  // name generate one. Alternatively if not generating types, use a default
  // type.
  if (InstType == INVALID_INST_TYPE) {
    if (ShouldGenerateMM)
      InstType = MM->getMMGen()->generateInstrType(MI);
    else
      InstType = MM->getDefaultInstType();
  }

  CreateNode_(SU.NodeNum, InstName.c_str(), InstType, InstName.c_str(),
              SU.NodeNum, // nodeID
              SU.NodeNum, // fileSchedOrder
              SU.NodeNum, // fileSchedCycle
              0,          // fileInstLwrBound
              0,          // fileInstUprBound
              0);         // blkNum
}

void OptSchedDDGWrapperBasic::discoverBoundaryLiveness(const MachineInstr *MI) {
  int LeafIndex = DAG->SUnits.size() + 1;
  RegisterOperands RegOpers;
  RegOpers.collect(*MI, *DAG->TRI, DAG->MRI, true, false);

  for (auto &U : RegOpers.Uses)
    addUse(U.RegUnit, LeafIndex);

  for (auto &D : RegOpers.Defs)
    addDef(D.RegUnit, LeafIndex);
}

void OptSchedDDGWrapperBasic::countBoundaryLiveness(
    std::vector<int> &RegDefCounts, std::set<unsigned> &Defs,
    const MachineInstr *MI) {
  RegisterOperands RegOpers;
  RegOpers.collect(*MI, *DAG->TRI, DAG->MRI, true, false);

  for (auto &D : RegOpers.Defs) {
    for (int Type : getRegisterType(D.RegUnit))
      RegDefCounts[Type]++;
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

bool LLVMRegTypeFilter::operator[](int16_t RegTypeID) const {
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
