#include "opt-sched/Scheduler/machine_model.h"
// For atoi().
#include <cstdlib>
// for setiosflags(), setprecision().
#include "opt-sched/Scheduler/buffers.h"
#include "opt-sched/Scheduler/logger.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <iomanip>
#include <string>

using namespace llvm::opt_sched;

MachineModel::MachineModel(const string &modelFile) {
  SpecsBuffer buf;
  char buffer[MAX_NAMESIZE];

  buf.Load(modelFile.c_str());

  buf.ReadSpec("MODEL_NAME:", buffer);
  mdlName_ = buffer;

  issueRate_ = buf.ReadIntSpec("ISSUE_RATE:");

  issueTypes_.resize(buf.ReadIntSpec("ISSUE_TYPE_COUNT:"));
  for (size_t j = 0; j < issueTypes_.size(); j++) {
    int pieceCnt;
    char *strngs[INBUF_MAX_PIECES_PERLINE];
    int lngths[INBUF_MAX_PIECES_PERLINE];
    buf.GetNxtVldLine(pieceCnt, strngs, lngths);

    if (pieceCnt != 2)
      llvm::report_fatal_error("Invalid issue type spec", false);

    issueTypes_[j].name = strngs[0];
    issueTypes_[j].slotsCount = atoi(strngs[1]);
  }

  dependenceLatencies_[DEP_DATA] = 1; // Shouldn't be used!
  dependenceLatencies_[DEP_ANTI] =
      (int16_t)buf.ReadIntSpec("DEP_LATENCY_ANTI:");
  dependenceLatencies_[DEP_OUTPUT] =
      (int16_t)buf.ReadIntSpec("DEP_LATENCY_OUTPUT:");
  dependenceLatencies_[DEP_OTHER] =
      (int16_t)buf.ReadIntSpec("DEP_LATENCY_OTHER:");

  registerTypes_.resize(buf.ReadIntSpec("REG_TYPE_COUNT:"));

  for (size_t i = 0; i < registerTypes_.size(); i++) {
    int pieceCnt;
    char *strngs[INBUF_MAX_PIECES_PERLINE];
    int lngths[INBUF_MAX_PIECES_PERLINE];
    buf.GetNxtVldLine(pieceCnt, strngs, lngths);

    if (pieceCnt != 2) {
      llvm::report_fatal_error("Invalid register type spec", false);
    }

    registerTypes_[i].name = strngs[0];
    registerTypes_[i].count = atoi(strngs[1]);
  }

  // Read instruction types.
  instTypes_.resize(buf.ReadIntSpec("INST_TYPE_COUNT:"));

  for (vector<InstTypeInfo>::iterator it = instTypes_.begin();
       it != instTypes_.end(); it++) {
    buf.ReadSpec("INST_TYPE:", buffer);
    it->name = buffer;
    it->isCntxtDep = (it->name.find("_after_") != string::npos);

    buf.ReadSpec("ISSUE_TYPE:", buffer);
    IssueType issuType = GetIssueTypeByName(buffer);

    if (issuType == INVALID_ISSUE_TYPE) {
      llvm::report_fatal_error(std::string("Invalid issue type ") + buffer +
                                   " for inst. type " + it->name,
                               false);
    }

    it->issuType = issuType;
    it->ltncy = (int16_t)buf.ReadIntSpec("LATENCY:");
    it->pipelined = buf.ReadFlagSpec("PIPELINED:", true);
    it->blksCycle = buf.ReadFlagSpec("BLOCKS_CYCLE:", false);
    it->sprtd = buf.ReadFlagSpec("SUPPORTED:", true);
  }
}

InstType MachineModel::GetInstTypeByName(const string &typeName,
                                         const string &prevName) const {
  string composite = prevName.size() ? typeName + "_after_" + prevName : "";
  for (size_t i = 0; i < instTypes_.size(); i++) {
    if (instTypes_[i].isCntxtDep && instTypes_[i].name == composite) {
      return (InstType)i;
    } else if (!instTypes_[i].isCntxtDep && instTypes_[i].name == typeName) {
      return (InstType)i;
    }
  }
  //  Logger::Error("Unrecognized instruction type %s.", typeName.c_str());
  return INVALID_INST_TYPE;
}

int16_t MachineModel::GetRegTypeByName(const char *const regTypeName) const {
  int16_t Type = INVALID_VALUE;
  for (size_t i = 0; i < registerTypes_.size(); i++) {
    if (regTypeName == registerTypes_[i].name) {
      Type = (int16_t)i;
      break;
    }
  }
  assert(Type != INVALID_VALUE &&
         "No register type with that name in machine model");
  return Type;
}

IssueType
MachineModel::GetIssueTypeByName(const char *const issuTypeName) const {
  for (size_t i = 0; i < issueTypes_.size(); i++) {
    if (issuTypeName == issueTypes_[i].name) {
      return (InstType)i;
    }
  }

  return INVALID_ISSUE_TYPE;
}

int MachineModel::GetPhysRegCnt(int16_t regType) const {
  return registerTypes_[regType].count;
}

const string &MachineModel::GetRegTypeName(int16_t regType) const {
  return registerTypes_[regType].name;
}

IssueType MachineModel::GetIssueType(InstType instTypeCode) const {
  return instTypes_[instTypeCode].issuType;
}

bool MachineModel::IsPipelined(InstType instTypeCode) const {
  return instTypes_[instTypeCode].pipelined;
}

bool MachineModel::IsSupported(InstType instTypeCode) const {
  return instTypes_[instTypeCode].sprtd;
}

bool MachineModel::BlocksCycle(InstType instTypeCode) const {
  return instTypes_[instTypeCode].blksCycle;
}

bool MachineModel::IsRealInst(InstType instTypeCode) const {
  IssueType issuType = GetIssueType(instTypeCode);
  return issueTypes_[issuType].name != "NULL";
}

int16_t MachineModel::GetLatency(InstType instTypeCode,
                                 DependenceType depType) const {
  if (depType == DEP_DATA && instTypeCode != INVALID_INST_TYPE) {
    return instTypes_[instTypeCode].ltncy;
  } else {
    return dependenceLatencies_[depType];
  }
}

int MachineModel::GetSlotsPerCycle(IssueType issuType) const {
  return issueTypes_[issuType].slotsCount;
}

int MachineModel::GetSlotsPerCycle(int slotsPerCycle[]) const {
  for (size_t i = 0; i < issueTypes_.size(); i++) {
    slotsPerCycle[i] = issueTypes_[i].slotsCount;
  }
  return issueTypes_.size();
}

const char *MachineModel::GetInstTypeNameByCode(InstType typeCode) const {
  return instTypes_[typeCode].name.c_str();
}

const char *MachineModel::GetIssueTypeNameByCode(IssueType typeCode) const {
  return issueTypes_[typeCode].name.c_str();
}

bool MachineModel::IsBranch(InstType instTypeCode) const {
  return instTypes_[instTypeCode].name == "br";
}

bool MachineModel::IsArtificial(InstType instTypeCode) const {
  return instTypes_[instTypeCode].name == "artificial";
}

bool MachineModel::IsCall(InstType instTypeCode) const {
  return instTypes_[instTypeCode].name == "call";
}

bool MachineModel::IsFloat(InstType instTypeCode) const {
  return instTypes_[instTypeCode].name[0] == 'f';
}

void MachineModel::AddInstType(InstTypeInfo &instTypeInfo) {
  // If this new instruction type is unpipelined notify the model
  if (!instTypeInfo.pipelined)
    includesUnpipelined_ = true;

  instTypes_.push_back(std::move(instTypeInfo));
}

void MachineModel::addIssueType(IssueTypeInfo &IssueTypeInfo) {
  issueTypes_.push_back(std::move(IssueTypeInfo));
}

InstType MachineModel::getDefaultInstType() const {
  return GetInstTypeByName("Default");
}

InstType MachineModel::getDefaultIssueType() const {
  return GetIssueTypeByName("Default");
}

const string &MachineModel::GetModelName() const { return mdlName_; }

int MachineModel::GetInstTypeCnt() const { return instTypes_.size(); }

int MachineModel::GetIssueTypeCnt() const { return issueTypes_.size(); }

int MachineModel::GetIssueRate() const { return issueRate_; }

int16_t MachineModel::GetRegTypeCnt() const {
  return (int16_t)registerTypes_.size();
}
