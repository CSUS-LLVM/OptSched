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
#include "llvm/MC/TargetRegistry.h"
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
               instType.name, instType.isCntxtDep ? "True" : "False",
               mm->GetIssueTypeNameByCode(instType.issuType), instType.ltncy,
               instType.pipelined ? "True" : "False",
               instType.sprtd ? "True" : "False",
               instType.blksCycle ? "True" : "False");
}
#endif

std::unique_ptr<MachineModelGenerator>
createCortexA7MMGenerator(const llvm::ScheduleDAGInstrs *dag,
                          MachineModel *mm) {
  return std::make_unique<CortexA7MMGenerator>(dag, mm);
}

} // end anonymous namespace

OptSchedMachineModel::OptSchedMachineModel(const char *configFile)
    : MachineModel(configFile), shouldGenerateMM(false), MMGen(nullptr) {}

void OptSchedMachineModel::convertMachineModel(
    const ScheduleDAGInstrs &dag, const RegisterClassInfo *regClassInfo) {
  
  const TargetMachine &target = dag.TM;

  mdlName_ = target.getTarget().getName();

  LLVM_DEBUG(dbgs() << "Machine model: " << mdlName_.c_str() << '\n');

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

/* old vector based method
  InstTypeInfo instType;

  instType.name = "Default";
  instType.isCntxtDep = false;
  instType.issuType = 0;
  instType.ltncy = 1;
  instType.pipelined = true;
  instType.sprtd = true;
  instType.blksCycle = false;
  //instTypes_.push_back(instType); replace vectors with arrays
*/

  if (instTypes_size_ == instTypes_alloc_) {  //double size of array if full
    if (instTypes_alloc_ > 0)
      instTypes_alloc_ *= 2;
    else
      instTypes_alloc_ = 2;
    InstTypeInfo *newArray = new InstTypeInfo[instTypes_alloc_];
    //copy old array to new array
    for (int i = 0; i < instTypes_size_; i++) {
      strncpy((char *)newArray[i].name, (const char *)instTypes_[i].name, 50);

      newArray[i].isCntxtDep = instTypes_[i].isCntxtDep;
      newArray[i].issuType = instTypes_[i].issuType;
      newArray[i].ltncy = instTypes_[i].ltncy;
      newArray[i].pipelined = instTypes_[i].pipelined;
      newArray[i].sprtd = instTypes_[i].sprtd;
      newArray[i].blksCycle = instTypes_[i].blksCycle;
    }
    //delete old array
    if (instTypes_size_ > 0)
      delete[] instTypes_;
    //set new array as instTypes_
    instTypes_ = newArray;
  }

  //copy element to next open slot
  strncpy((char *)instTypes_[instTypes_size_].name, "Default", 50);
  
  instTypes_[instTypes_size_].isCntxtDep = false;
  instTypes_[instTypes_size_].issuType = 0;
  instTypes_[instTypes_size_].ltncy = 1;
  instTypes_[instTypes_size_].pipelined = true;
  instTypes_[instTypes_size_].sprtd = true;
  instTypes_[instTypes_size_].blksCycle = false;
  instTypes_size_++;

/* old vector based method
  instType.name = "artificial";
  instType.isCntxtDep = false;
  instType.issuType = 0;
  instType.ltncy = 1;
  instType.pipelined = true;
  instType.sprtd = true;
  instType.blksCycle = false;
  //instTypes_.push_back(instType); replace vectors with arrays
*/

  if (instTypes_size_ == instTypes_alloc_) {  //double size of array if full
    if (instTypes_alloc_ > 0)
      instTypes_alloc_ *= 2;
    else
      instTypes_alloc_ = 2;
    InstTypeInfo *newArray = new InstTypeInfo[instTypes_alloc_];
    //copy old array to new array
    for (int i = 0; i < instTypes_size_; i++) {
      strncpy((char *)newArray[i].name, (const char *)instTypes_[i].name, 50);
      
      newArray[i].isCntxtDep = instTypes_[i].isCntxtDep;
      newArray[i].issuType = instTypes_[i].issuType;
      newArray[i].ltncy = instTypes_[i].ltncy;
      newArray[i].pipelined = instTypes_[i].pipelined;
      newArray[i].sprtd = instTypes_[i].sprtd;
      newArray[i].blksCycle = instTypes_[i].blksCycle;
    }
    //delete old array
    if (instTypes_size_ > 0)
      delete[] instTypes_;
    //set new array as instTypes_
    instTypes_ = newArray;
  }

  //copy element to next open slot
  strncpy((char*)instTypes_[instTypes_size_].name, "artificial", 50);

  instTypes_[instTypes_size_].isCntxtDep = false;
  instTypes_[instTypes_size_].issuType = 0;
  instTypes_[instTypes_size_].ltncy = 1;
  instTypes_[instTypes_size_].pipelined = true;
  instTypes_[instTypes_size_].sprtd = true;
  instTypes_[instTypes_size_].blksCycle = false;
  instTypes_size_++;


  // Clear The registerTypes list to read registers limits from the LLVM machine
  // model
  delete[] registerTypes_;
  registerTypes_size_ = 0;

  if (mdlName_ == "amdgcn") {
    registerTypes_size_ = 2;
    registerTypes_ = new RegTypeInfo[registerTypes_size_];
    //RegTypeInfo SGPR32;
    //SGPR32.name = "SGPR32";
    //SGPR32.count = 80;
    strncpy((char *)registerTypes_[0].name, "SGPR32", 50);
    registerTypes_[0].count = 80;

    //RegTypeInfo VGPR32;
    //VGPR32.name = "VGPR32";
    //VGPR32.count = 24;
    strncpy((char *)registerTypes_[1].name, "VGPR32", 50);
    registerTypes_[1].count = 24;
  } else {
    const auto *TRI = dag.TRI;
    registerTypes_size_ = (int)TRI->getNumRegPressureSets();
    registerTypes_ = new RegTypeInfo[registerTypes_size_];
    for (unsigned pSet = 0; pSet < TRI->getNumRegPressureSets(); ++pSet) {
      RegTypeInfo regType;
      strncpy((char *)regType.name, TRI->getRegPressureSetName(pSet), 50);
      int pressureLimit = regClassInfo->getRegPressureSetLimit(pSet);
      // set registers with 0 limit to 1 to support flags and special cases
      if (pressureLimit > 0)
        regType.count = pressureLimit;
      else
        regType.count = 1;
      strncpy((char *)registerTypes_[(int)pSet].name, (const  char *)regType.name, 50);
      registerTypes_[(int)pSet].count = regType.count;
    }
  }
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
  const std::string instrName = DAG->TII->getName(instr->getOpcode()).data();
  const InstType InstType = MM->GetInstTypeByName(instrName);

  // If the machine model does not have instType with this OpCode name,
  // generate a type for the instruction.
  if (InstType != INVALID_INST_TYPE)
    return InstType;
  else {
    LLVM_DEBUG(dbgs() << "Generating instr type for " << instrName);

    SUnit *SU = DAG->getSUnit(const_cast<MachineInstr *>(instr));
    const MCInstrDesc *instDesc = DAG->getInstrDesc(SU);
    unsigned IDX = instDesc->getSchedClass();

    if (!IID || IID->isEmpty() || IID->isEndMarker(IDX))
      return MM->getDefaultInstType();

    // Create the new instruction type
    InstTypeInfo InstTypeI;
    strncpy((char *)InstTypeI.name, instrName.c_str(), 50);
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
    return MM->GetInstTypeByName(InstTypeI.name);
  }
}
