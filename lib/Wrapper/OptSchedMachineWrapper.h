/*******************************************************************************
Description:  A wrapper that converts an LLVM target to an OptSched
MachineModel. By default machine models are read from ini files however
MachineModelGenerator classes may supplement or override the information
contained in those ini files.
*******************************************************************************/

#ifndef OPTSCHED_MACHINE_MODEL_WRAPPER_H
#define OPTSCHED_MACHINE_MODEL_WRAPPER_H

#include "opt-sched/Scheduler/machine_model.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include <map>

using namespace llvm;

namespace llvm {
namespace opt_sched {

class MachineModelGenerator;

// A wrapper for the OptSched MachineModel
class OptSchedMachineModel : public MachineModel {
public:
  // Use a config file to initialize the machine model.
  OptSchedMachineModel(const char *configFile);
  // Convert information about the target machine into the
  // optimal scheduler machine model
  void convertMachineModel(const llvm::ScheduleDAGInstrs &dag,
                           const llvm::RegisterClassInfo *regClassInfo);
  MachineModelGenerator *getMMGen() { return MMGen.get(); }
  ~OptSchedMachineModel() = default;

private:
  // Should a machine model be generated.
  bool shouldGenerateMM;
  // The machine model generator class.
  std::unique_ptr<MachineModelGenerator> MMGen;
};

// Generate a machine model for a specific chip.
class MachineModelGenerator {
public:
  // Generate instruction scheduling type for all instructions in the current
  // DAG that do not already have assigned instruction types.
  virtual InstType generateInstrType(const llvm::MachineInstr *instr) = 0;
  virtual bool generatesAllData() = 0;
  virtual void generateProcessorData(std::string *mdlName_, int *issueRate_) {}
  virtual ~MachineModelGenerator() = default;
};

// Generate a machine model for the Cortex A7. This will only generate
// instruction types. Things like issue type and issue rate must be specified
// correctly in the machine_model.cfg file. Check
// OptSchedCfg/arch/ARM_cortex_a7_machine_model.cfg for a template.
class CortexA7MMGenerator : public MachineModelGenerator {
public:
  CortexA7MMGenerator(const llvm::ScheduleDAGInstrs *dag, MachineModel *mm);
  // Generate instruction scheduling type for all instructions in the current
  // DAG by using LLVM itineraries.
  InstType generateInstrType(const llvm::MachineInstr *instr);
  bool generatesAllData() { return false; }
  virtual ~CortexA7MMGenerator() = default;

private:
  // Functional Units
  enum FU : unsigned {
    Pipe0 = 1,   // 00000001
    Pipe1 = 2,   // 00000010
    LSPipe = 4,  // 00000100
    NPipe = 8,   // 00001000
    NLSPipe = 16 // 00010000
  };
  const llvm::ScheduleDAGInstrs *DAG;
  MachineModel *MM;
  const llvm::InstrItineraryData *IID;

  // Returns true if a machine instruction should be considered fully pipelined
  // in the machine model.
  bool isMIPipelined(const llvm::MachineInstr *inst, unsigned idx) const;
  // Find the issue type for an instruction.
  IssueType generateIssueType(const llvm::InstrStage *E) const;
};

class CortexA53MMGenerator : public MachineModelGenerator {
public:
  CortexA53MMGenerator(const llvm::ScheduleDAGInstrs *dag, MachineModel *mm)
      : DAG(dag), MM(mm) {}
  InstType generateInstrType(const llvm::MachineInstr *instr);
  bool generatesAllData() { return true; }
  void generateProcessorData(std::string *mdlName_, int *issueRate_);

private:
  std::vector<std::string> ResourceIdToIssueType;
  const llvm::ScheduleDAGInstrs *DAG;
  MachineModel *MM;
};

} // end namespace opt_sched
} // namespace llvm

#endif
