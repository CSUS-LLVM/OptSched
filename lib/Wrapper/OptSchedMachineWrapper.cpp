/*******************************************************************************
Description:  A wrapper that convert an LLVM target to an OptSched MachineModel.
*******************************************************************************/

#include "llvm/CodeGen/MachineInstr.h"
#include "OptSchedMachineWrapper.h"
#include "opt-sched/Scheduler/machine_model.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/logger.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <memory>

#define DEBUG_TYPE "optsched"

using namespace opt_sched;
using namespace llvm;

namespace {

#ifdef IS_DEBUG_MM
void dumpInstType(InstTypeInfo &instType, MachineModel *mm) const {
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

} // end anonymous namespace


LLVMMachineModel::LLVMMachineModel(const char *configFile)
    : MachineModel(configFile), registerInfo(nullptr), shouldGenerateMM(false),
      MMGen(nullptr) {}

void LLVMMachineModel::convertMachineModel(
    const ScheduleDAGInstrs &dag, const RegisterClassInfo *regClassInfo) {
  const TargetMachine &target = dag.TM;

  mdlName_ = target.getTarget().getName();

  Logger::Info("Machine model: %s", mdlName_.c_str());

  // Should we try to generate a machine model using LLVM itineraries.
  shouldGenerateMM =
      SchedulerOptions::getInstance().GetBool("GENERATE_MACHINE_MODEL", false);

  if (shouldGenerateMM) {
    if (mdlName_ == "ARM-Cortex-A7")
      MMGen = createCortexA7MMGenerator(&dag, this);
    else
      Logger::Error("Could not find machine model generator for target \"%s\"",
                    mdlName_.c_str());
  }

  // Clear The registerTypes list to read registers limits from the LLVM machine
  // model
  registerTypes_.clear();

  registerInfo = dag.TRI;
  for (int pSet = 0; pSet < registerInfo->getNumRegPressureSets(); ++pSet) {
    RegTypeInfo regType;
    regType.name = registerInfo->getRegPressureSetName(pSet);
    int pressureLimit = regClassInfo->getRegPressureSetLimit(pSet);
    // set registers with 0 limit to 1 to support flags and special cases
    if (pressureLimit > 0)
      regType.count = pressureLimit;
    else
      regType.count = 1;

    registerTypes_.push_back(regType);
#ifdef IS_DEBUG_MM
    Logger::Info("Pressure set %s has a limit of %d", regType.name.c_str(),
                 regType.count);
#endif
  }

#ifdef IS_DEBUG_REG_CLASS
  Logger::Info("LLVM register class info");

  for (TargetRegisterClass::sc_iterator cls = registerInfo->regclass_begin();
       cls != registerInfo->regclass_end(); cls++) {
    RegTypeInfo regType;
    const char *clsName = registerInfo->getRegClassName(*cls);
    unsigned weight = registerInfo->getRegClassWeight(*cls).RegWeight;
    Logger::Info("For the register class %s getNumRegs is %d", clsName,
                 (*cls)->getNumRegs());
    Logger::Info("For the register class %s getRegPressureLimit is %d", clsName,
                 registerInfo->getRegPressureLimit((*cls), dag.MF));
    Logger::Info("This register has a weight of %lu", weight);
    Logger::Info("Pressure Sets for this register class");
    for (const int *pSet = registerInfo->getRegClassPressureSets(*cls);
         *pSet != -1; ++pSet) {
      Logger::Info(
          "Pressure set %s has member %s and a pressure set limit of %d",
          registerInfo->getRegPressureSetName(*pSet), clsName,
          registerInfo->getRegPressureSetLimit(dag.MF, *pSet));
    }
  }
#endif

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

// Print the machine model parameters.
#ifdef IS_DEBUG_MM
  Logger::Info(
      "######################## THE MACHINE MODEL #######################");
  Logger::Info("Issue Rate: %d.", issueRate_);
  Logger::Info("Issue Types Count: %d", issueTypes_.size());
  for (int x = 0; x < issueTypes_.size(); x++)
    Logger::Info("Type %s has %d slots", issueTypes_[x].name.c_str(),
                 issueTypes_[x].slotsCount);

  Logger::Info("Instructions Type Count: %d", instTypes_.size());
  for (int y = 0; y < instTypes_.size(); y++)
    Logger::Info("Instruction %s is of issue type %s and has a latency of %d",
                 instTypes_[y].name.c_str(),
                 issueTypes_[instTypes_[y].issuType].name.c_str(),
                 instTypes_[y].ltncy);
#endif
}

CortexA7MMGenerator::CortexA7MMGenerator(const llvm::ScheduleDAGInstrs *dag,
                                         MachineModel *mm)
    : dag(dag), mm(mm) {
  iid = dag->getSchedModel()->getInstrItineraries();
}

bool CortexA7MMGenerator::isMIPipelined(const MachineInstr *inst,
                                        unsigned idx) const {
  const InstrStage *IS = iid->beginStage(idx);
  const InstrStage *E = iid->endStage(idx);

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
    type = mm->GetIssueTypeByName("NLSPipe");
  else if (units & NPipe)
    type = mm->GetIssueTypeByName("NPipe");
  else if (units & LSPipe)
    type = mm->GetIssueTypeByName("LSPipe");
  else if (units & Pipe0 || units & Pipe1)
    type = mm->GetIssueTypeByName("ALUPipe");
  else
    type = mm->GetInstTypeByName("Default");

  assert(type != INVALID_ISSUE_TYPE && "Could not find issue type for "
                                       "instruction, is the correct machine "
                                       "model file loaded?");
  return type;
}

void CortexA7MMGenerator::generateInstrType(const MachineInstr *instr) {
  const std::string instrName = dag->TII->getName(instr->getOpcode());

  // Search in the machine model for an instType with this OpCode name
  const InstType instType = mm->GetInstTypeByName(instrName);

  // If the machine model does not have instType with this OpCode name,
  // generate a type for the instruction.
  if (instType == INVALID_INST_TYPE) {
#ifdef IS_DEBUG_MM
    Logger::Info("Generating instr type for \'%s\'", instrName.c_str());
#endif
    SUnit *su = dag->getSUnit(const_cast<MachineInstr *>(instr));
    const MCInstrDesc *instDesc = dag->getInstrDesc(su);
    unsigned idx = instDesc->getSchedClass();

    if (!iid || iid->isEmpty() || iid->isEndMarker(idx)) {
#ifdef IS_DEBUG_MM
      Logger::Info("No itinerary data for instr \'%s\'", instrName.c_str());
#endif
      return;
    }

    // Create the new instruction type
    InstTypeInfo instType;
    instType.name = instrName;
    // Endstage is the last stage+1 so decrement the iterator to get final stage
    const InstrStage *E = iid->endStage(idx);
    instType.issuType = generateIssueType(--E);
    instType.isCntxtDep = false;
    instType.ltncy = 1; // Assume using "rough" llvm latencies
    instType.pipelined = isMIPipelined(instr, idx);
    instType.sprtd = true;
    instType.blksCycle = false; // TODO: Find a more precise value for this.

#ifdef IS_DEBUG_MM
    dumpInstType(instType, mm);
#endif

    // Add the new instruction type
    mm->AddInstType(instType);
  }
}
