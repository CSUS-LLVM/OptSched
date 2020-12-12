/*******************************************************************************
Description:  A wrapper that convert an LLVM target to an OptSched MachineModel.
*******************************************************************************/

#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

#define DEBUG_TYPE "optsched-machine-model"

using namespace llvm;
using namespace llvm::opt_sched;

namespace {

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void dumpInstType(InstTypeInfo &instType, MachineModel *mm) {
  Logger::Info("Adding new instruction type.\n \
                Name: %s\n \
                Is Context Dependent: %s\n \
                IssueType: %s\n \
                Latency: %d\n \
                Is Pipelined %s\n \
                Supported: %s\n \
                Blocks Cycle: %s\n",
               instType.name.c_str(), instType.isCntxtDep ? "True" : "False",
               mm->GetIssueTypeNameByCode(instType.issuType), instType.ltncy,
               instType.pipelined ? "True" : "False",
               instType.sprtd ? "True" : "False",
               instType.blksCycle ? "True" : "False");
}
#endif

std::unique_ptr<MachineModelGenerator>
createCortexA7MMGenerator(const llvm::ScheduleDAGInstrs *dag,
                          MachineModel *mm) {
  return make_unique<CortexA7MMGenerator>(dag, mm);
}

std::unique_ptr<MachineModelGenerator>
createCortexA53MMGenerator(const llvm::ScheduleDAGInstrs *dag,
                           MachineModel *mm) {
  return make_unique<CortexA53MMGenerator>(dag, mm);
}

} // end anonymous namespace

OptSchedMachineModel::OptSchedMachineModel(const char *configFile)
    : MachineModel(configFile), shouldGenerateMM(false), MMGen(nullptr) {}

void OptSchedMachineModel::convertMachineModel(
    const ScheduleDAGInstrs &Dag, const RegisterClassInfo *RegClassInfo) {
  const TargetMachine &Target = Dag.TM;
  const TargetSchedModel *LLVMSchedModel = Dag.getSchedModel();
  const TargetSubtargetInfo *SubtargetInfo = LLVMSchedModel->getSubtargetInfo();
  std::string ExactCPU = SubtargetInfo->getCPU().str();

  mdlName_ = Target.getTarget().getName();

  LLVM_DEBUG(dbgs() << "Machine model: " << mdlName_.c_str() << '\n');

  // Should we try to generate a machine model using LLVM itineraries.
  shouldGenerateMM =
      SchedulerOptions::getInstance().GetBool("GENERATE_MACHINE_MODEL", false);

  if (shouldGenerateMM) {
    if (mdlName_ == "ARM-Cortex-A7")
      MMGen = createCortexA7MMGenerator(&Dag, this);
    else if (ExactCPU == "cortex-a53")
      MMGen = createCortexA53MMGenerator(&Dag, this);
    else
      llvm::report_fatal_error(
          "Could not find machine model generator for target " + mdlName_,
          false);
  }

  InstTypeInfo instType;

  instType.name = "Default";
  instType.isCntxtDep = false;
  instType.issuType = 0;
  instType.ltncy = 1;
  instType.pipelined = true;
  instType.sprtd = true;
  instType.blksCycle = false;
  instTypes_.push_back(instType);

  instType.name = "artificial";
  instType.isCntxtDep = false;
  instType.issuType = 0;
  instType.ltncy = 1;
  instType.pipelined = true;
  instType.sprtd = true;
  instType.blksCycle = false;
  instTypes_.push_back(instType);

  // Clear The registerTypes list to read registers limits from the LLVM machine
  // model
  registerTypes_.clear();

  if (mdlName_ == "amdgcn") {
    RegTypeInfo SGPR32;
    SGPR32.name = "SGPR32";
    SGPR32.count = 80;
    registerTypes_.push_back(SGPR32);

    RegTypeInfo VGPR32;
    VGPR32.name = "VGPR32";
    VGPR32.count = 24;
    registerTypes_.push_back(VGPR32);
  } else {
    const auto *TRI = Dag.TRI;
    for (unsigned pSet = 0; pSet < TRI->getNumRegPressureSets(); ++pSet) {
      RegTypeInfo regType;
      regType.name = TRI->getRegPressureSetName(pSet);
      int pressureLimit = RegClassInfo->getRegPressureSetLimit(pSet);
      // set registers with 0 limit to 1 to support flags and special cases
      if (pressureLimit > 0)
        regType.count = pressureLimit;
      else
        regType.count = 1;
      registerTypes_.push_back(regType);
    }
  }

  // initialize other MM non-instruction specific info if appropriate
  if (shouldGenerateMM && MMGen->generatesAllData())
    MMGen->generateProcessorData(&mdlName_, &issueRate_);
}

CortexA7MMGenerator::CortexA7MMGenerator(const llvm::ScheduleDAGInstrs *dag,
                                         MachineModel *mm)
    : DAG(dag), MM(mm) {
  IID = dag->getSchedModel()->getInstrItineraries();
}

bool CortexA7MMGenerator::isMIPipelined(const MachineInstr *inst,
                                        unsigned idx) const {
  const InstrStage *IS = IID->beginStage(idx);
  const InstrStage *E = IID->endStage(idx);

  for (; IS != E; ++IS)
    if (IS->getCycles() > 1)
      // Instruction contains a non-pipelined stage
      return false;

  // All stages can be pipelined
  return true;
}

IssueType CortexA7MMGenerator::generateIssueType(const InstrStage *E) const {
  IssueType type = INVALID_ISSUE_TYPE;
  // Get functional units for the last stage in the itinerary
  const unsigned units = E->getUnits();

  if (units & NLSPipe)
    type = MM->GetIssueTypeByName("NLSPipe");
  else if (units & NPipe)
    type = MM->GetIssueTypeByName("NPipe");
  else if (units & LSPipe)
    type = MM->GetIssueTypeByName("LSPipe");
  else if (units & Pipe0 || units & Pipe1)
    type = MM->GetIssueTypeByName("ALUPipe");
  else
    type = MM->getDefaultIssueType();

  assert(type != INVALID_ISSUE_TYPE && "Could not find issue type for "
                                       "instruction, is the correct machine "
                                       "model file loaded?");
  return type;
}

InstType CortexA7MMGenerator::generateInstrType(const MachineInstr *instr) {
  // Search in the machine model for an instType with this OpCode
  const std::string instrName = DAG->TII->getName(instr->getOpcode());
  const InstType InstType = MM->GetInstTypeByName(instrName);

  // If the machine model does not have instType with this OpCode name,
  // generate a type for the instruction.
  if (InstType != INVALID_INST_TYPE)
    return InstType;
  else {
    LLVM_DEBUG(dbgs() << "Generating instr type for " << instrName.c_str());

    SUnit *SU = DAG->getSUnit(const_cast<MachineInstr *>(instr));
    const MCInstrDesc *instDesc = DAG->getInstrDesc(SU);
    unsigned IDX = instDesc->getSchedClass();

    if (!IID || IID->isEmpty() || IID->isEndMarker(IDX))
      return MM->getDefaultInstType();

    // Create the new instruction type
    InstTypeInfo InstTypeI;
    InstTypeI.name = instrName;
    const InstrStage *E = IID->endStage(IDX);
    InstTypeI.issuType = generateIssueType(--E);
    InstTypeI.isCntxtDep = false;
    InstTypeI.ltncy = 1; // Assume using "rough" llvm latencies
    InstTypeI.pipelined = isMIPipelined(instr, IDX);
    InstTypeI.sprtd = true;
    InstTypeI.blksCycle = false; // TODO: Find a more precise value for this.

    LLVM_DEBUG(dumpInstType(InstTypeI, MM));

    // Add the new instruction type
    MM->AddInstType(InstTypeI);
    return MM->GetInstTypeByName(instrName);
  }
}

void CortexA53MMGenerator::generateProcessorData(std::string *mdlName_,
                                                 int *issueRate_) {
  const TargetSchedModel *LLVMSchedModel = DAG->getSchedModel();
  const TargetSubtargetInfo *SubtargetInfo = LLVMSchedModel->getSubtargetInfo();
  std::string ExactCPU = SubtargetInfo->getCPU().str();

  *mdlName_ = ExactCPU;
  *issueRate_ = LLVMSchedModel->getIssueWidth();

  unsigned FUTypesCount = LLVMSchedModel->getNumProcResourceKinds();
  ResourceIdToIssueType.resize(FUTypesCount);
  for (unsigned FUIdx = 1; FUIdx < FUTypesCount; FUIdx++) {
    const MCProcResourceDesc *FUResource =
        LLVMSchedModel->getProcResource(FUIdx);
    ResourceIdToIssueType[FUIdx] = FUResource->Name;
    IssueTypeInfo ITI{FUResource->Name, static_cast<int>(FUResource->NumUnits)};
    MM->addIssueType(ITI);
  }
}

InstType
CortexA53MMGenerator::generateInstrType(const llvm::MachineInstr *instr) {
  // Search in the machine model for an instType with this OpCode
  const std::string InstrName = DAG->TII->getName(instr->getOpcode());
  const InstType InstrType = MM->GetInstTypeByName(InstrName);

  // If the machine model does not have instType with this OpCode name,
  // generate a type for the instruction.
  if (InstrType != INVALID_INST_TYPE)
    return InstrType;
  else {
    const TargetSchedModel *LLVMSchedModel = DAG->getSchedModel();
    const MCSchedClassDesc *MCDesc = LLVMSchedModel->resolveSchedClass(instr);

    InstTypeInfo InstTypeI;
    InstTypeI.name = InstrName;
    // Most of the A53's instructions only use one FU (one uses 3 FUs and some
    // use 0 FUs) we use the default issue type for instructions which use no FU
    // If we were to see the 3 FU instruction then we assign it its first issue
    // type (but I think the 3 FU inst is a type of branch isntruction that we
    // never see)
    const MCWriteProcResEntry *ResEntry =
        LLVMSchedModel->getWriteProcResBegin(MCDesc);
    if (ResEntry != LLVMSchedModel->getWriteProcResEnd(MCDesc)) {
      std::string IssueTypeName =
          ResourceIdToIssueType[ResEntry->ProcResourceIdx];
      InstTypeI.issuType = MM->GetIssueTypeByName(IssueTypeName.c_str());
      InstTypeI.pipelined = (ResEntry->Cycles == 1);
    } else {
      InstTypeI.issuType = MM->getDefaultIssueType();
      InstTypeI.pipelined = true;
    }
    InstTypeI.ltncy = 1; // This field is never used
    InstTypeI.isCntxtDep = false;
    InstTypeI.sprtd = true;
    InstTypeI.blksCycle = false;

    Logger::Info(
        "Adding new instruction type.\n"
        "Name: %s\n"
        "Is Context Dependent: %s\n"
        "IssueType: %s (%d)\n"
        "Latency: %d\n"
        "Is Pipelined: %s\n"
        "Supported: %s\n"
        "Blocks Cycle: %s\n",
        InstTypeI.name.c_str(), InstTypeI.isCntxtDep ? "True" : "False",
        MM->GetIssueTypeNameByCode(InstTypeI.issuType), InstTypeI.issuType,
        InstTypeI.ltncy, InstTypeI.pipelined ? "True" : "False",
        InstTypeI.sprtd ? "True" : "False",
        InstTypeI.blksCycle ? "True" : "False");

    // Add the new instruction type
    MM->AddInstType(InstTypeI);
    return MM->GetInstTypeByName(InstrName);
  }
}
