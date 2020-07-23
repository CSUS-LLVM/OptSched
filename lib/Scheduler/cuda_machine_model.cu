#include "opt-sched/Scheduler/machine_model.h"
// For atoi().
#include <cstdlib>
// for setiosflags(), setprecision().
#include "opt-sched/Scheduler/buffers.h"
#include "opt-sched/Scheduler/logger.h"
#include <cassert>
#include <iomanip>

using namespace llvm::opt_sched;

MachineModel::MachineModel(const string &modelFile) {
  SpecsBuffer buf;
  char buffer[MAX_NAMESIZE];

  buf.Load(modelFile.c_str());

  buf.ReadSpec("MODEL_NAME:", buffer);
  mdlName_ = buffer;

  issueRate_ = buf.ReadIntSpec("ISSUE_RATE:");

  issueTypes_size_ = buf.ReadIntSpec("ISSUE_TYPE_COUNT:");

  issueTypes_ = new IssueTypeInfo[issueTypes_size_];
  for (size_t j = 0; j < (size_t)issueTypes_size_; j++) {
    int pieceCnt;
    char *strngs[INBUF_MAX_PIECES_PERLINE];
    int lngths[INBUF_MAX_PIECES_PERLINE];
    buf.GetNxtVldLine(pieceCnt, strngs, lngths);

    if (pieceCnt != 2)
      Logger::Fatal("Invalid issue type spec");

    strncpy((char *)issueTypes_[j].name, (const char *)strngs[0], 50);

    issueTypes_[j].slotsCount = atoi(strngs[1]);
  }

  dependenceLatencies_[DEP_DATA] = 1; // Shouldn't be used!
  dependenceLatencies_[DEP_ANTI] =
      (int16_t)buf.ReadIntSpec("DEP_LATENCY_ANTI:");
  dependenceLatencies_[DEP_OUTPUT] =
      (int16_t)buf.ReadIntSpec("DEP_LATENCY_OUTPUT:");
  dependenceLatencies_[DEP_OTHER] =
      (int16_t)buf.ReadIntSpec("DEP_LATENCY_OTHER:");

  registerTypes_size_ = buf.ReadIntSpec("REG_TYPE_COUNT:");
  registerTypes_ = new RegTypeInfo[registerTypes_size_];
  for (size_t i = 0; i < (size_t)registerTypes_size_; i++) {
    int pieceCnt;
    char *strngs[INBUF_MAX_PIECES_PERLINE];
    int lngths[INBUF_MAX_PIECES_PERLINE];
    buf.GetNxtVldLine(pieceCnt, strngs, lngths);

    if (pieceCnt != 2) {
      Logger::Fatal("Invalid register type spec");
    }

    strncpy((char *)registerTypes_[i].name, (const char *)strngs[0], 50);

    registerTypes_[i].count = atoi(strngs[1]);
  }

  // Read instruction types.
  instTypes_size_ = instTypes_alloc_ = buf.ReadIntSpec("INST_TYPE_COUNT:");
  instTypes_ = new InstTypeInfo[instTypes_alloc_];

  for (int i = 0; i < instTypes_size_; i++) {
    buf.ReadSpec("INST_TYPE:", buffer);
    strncpy((char *)instTypes_[i].name, (const char *)buffer, 50);

    string tempName(instTypes_[i].name);
    instTypes_[i].isCntxtDep = (tempName.find("_after_") != string::npos);

    buf.ReadSpec("ISSUE_TYPE:", buffer);
    IssueType issuType = GetIssueTypeByName(buffer);

    if (issuType == INVALID_ISSUE_TYPE) {
      Logger::Fatal("Invalid issue type %s for inst. type %s", buffer,
                    instTypes_[i].name);
    }

    instTypes_[i].issuType = issuType;
    instTypes_[i].ltncy = (int16_t)buf.ReadIntSpec("LATENCY:");
    instTypes_[i].pipelined = buf.ReadFlagSpec("PIPELINED:", true);
    instTypes_[i].blksCycle = buf.ReadFlagSpec("BLOCKS_CYCLE:", false);
    instTypes_[i].sprtd = buf.ReadFlagSpec("SUPPORTED:", true);
  }
}

InstType MachineModel::GetInstTypeByName(const string &typeName,
                                         const string &prevName) const {
  string composite = prevName.size() ? typeName + "_after_" + prevName : "";
  for (size_t i = 0; i < (size_t)instTypes_size_; i++) {
    if (instTypes_[i].isCntxtDep && (0 == strncmp(instTypes_[i].name, composite.c_str(), composite.length()))) {
      return (InstType)i;
    } else if (!instTypes_[i].isCntxtDep && (0 == strncmp(instTypes_[i].name, typeName.c_str(), typeName.length()))) {
      return (InstType)i;
    }
  }
  //Logger::Error("Unrecognized instruction type %s.", typeName.c_str());
  return INVALID_INST_TYPE;
}

int16_t MachineModel::GetRegTypeByName(const char *const regTypeName) const {
  int16_t Type = INVALID_VALUE;
  for (size_t i = 0; i < (size_t)registerTypes_size_; i++) {
    if (0 == strncmp(regTypeName, (const char *)registerTypes_[i].name, 50)) {
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
  for (size_t i = 0; i < (size_t)issueTypes_size_; i++) {
    if (0 == strncmp(issuTypeName, (const char *)issueTypes_[i].name, 50)) {
      return (InstType)i;
    }
  }

  return INVALID_ISSUE_TYPE;
}

__host__ __device__
int MachineModel::GetPhysRegCnt(int16_t regType) const {
  return registerTypes_[regType].count;
}

__host__ __device__
const char *MachineModel::GetRegTypeName(int16_t regType) const {
  return registerTypes_[regType].name;
}

__host__ __device__
IssueType MachineModel::GetIssueType(InstType instTypeCode) const {
  return instTypes_[instTypeCode].issuType;
}

__host__ __device__
bool MachineModel::IsPipelined(InstType instTypeCode) const {
  return instTypes_[instTypeCode].pipelined;
}

bool MachineModel::IsSupported(InstType instTypeCode) const {
  return instTypes_[instTypeCode].sprtd;
}

__host__ __device__
bool MachineModel::BlocksCycle(InstType instTypeCode) const {
  return instTypes_[instTypeCode].blksCycle;
}

__host__ __device__
bool MachineModel::IsRealInst(InstType instTypeCode) const {
  IssueType issuType = GetIssueType(instTypeCode);
  bool match = true;
  char nullstr[] = "NULL";

  for (int i = 0; i < 4; i++) {
    if (instTypes_[instTypeCode].name[i] != nullstr[i]);
      match = false;
  }

  return match;
}

int16_t MachineModel::GetLatency(InstType instTypeCode,
                                 DependenceType depType) const {
  if (depType == DEP_DATA && instTypeCode != INVALID_INST_TYPE) {
    return instTypes_[instTypeCode].ltncy;
  } else {
    return dependenceLatencies_[depType];
  }
}

__host__ __device__
int MachineModel::GetSlotsPerCycle(IssueType issuType) const {
  return issueTypes_[issuType].slotsCount;
}

__host__ __device__
int MachineModel::GetSlotsPerCycle(int slotsPerCycle[]) const {
  for (size_t i = 0; i < (size_t)issueTypes_size_; i++) {
    slotsPerCycle[i] = issueTypes_[i].slotsCount;
  }
  return issueTypes_size_;
}

const char *MachineModel::GetInstTypeNameByCode(InstType typeCode) const {
  return instTypes_[typeCode].name;
}

const char *MachineModel::GetIssueTypeNameByCode(IssueType typeCode) const {
  return issueTypes_[typeCode].name;
}

bool MachineModel::IsBranch(InstType instTypeCode) const {
  return (0 == strncmp((const char *)instTypes_[instTypeCode].name, "br", 2));
}

bool MachineModel::IsArtificial(InstType instTypeCode) const {
  return (0 == strncmp((const char *)instTypes_[instTypeCode].name, "artificial", 10));
}

bool MachineModel::IsCall(InstType instTypeCode) const {
  return (0 == strncmp((const char *)instTypes_[instTypeCode].name, "call", 4));
}

bool MachineModel::IsFloat(InstType instTypeCode) const {
  return instTypes_[instTypeCode].name[0] == 'f';
}

void MachineModel::AddInstType(InstTypeInfo &instTypeInfo) {
  // If this new instruction type is unpipelined notify the model
  if (!instTypeInfo.pipelined)
    includesUnpipelined_ = true;

  if (instTypes_size_ == instTypes_alloc_) {  //double size of array if full
    if (instTypes_alloc_ > 0)
      instTypes_alloc_ *= 2;
    else
      instTypes_alloc_ = 2;
    InstTypeInfo *newArray = new InstTypeInfo[instTypes_alloc_];
    //copy old array to new array
    for (int i = 0; i < instTypes_size_; i++) {
      strncpy((char *)newArray[i].name, (const char *)instTypes_[i].name, 50);

      //debug
      printf("copy inst instTypes_[%d].name : %s\n", i, instTypes_[i].name);

      newArray[i].isCntxtDep = instTypes_[i].isCntxtDep;
      newArray[i].issuType = instTypes_[i].issuType;
      newArray[i].ltncy = instTypes_[i].ltncy;
      newArray[i].pipelined = instTypes_[i].pipelined;
      newArray[i].sprtd = instTypes_[i].sprtd;
      newArray[i].blksCycle = instTypes_[i].blksCycle;
    }
    //delete old array
    delete[] instTypes_;
    //set new array as instTypes_
    instTypes_ = newArray;
  }
  //copy element to next open slot
  strncpy((char *)instTypes_[instTypes_size_].name, (const char*)instTypeInfo.name, 50);

  instTypes_[instTypes_size_].issuType = instTypeInfo.issuType;
  instTypes_[instTypes_size_].isCntxtDep = instTypeInfo.isCntxtDep;
  instTypes_[instTypes_size_].ltncy = instTypeInfo.ltncy;
  instTypes_[instTypes_size_].pipelined = instTypeInfo.pipelined;
  instTypes_[instTypes_size_].sprtd = instTypeInfo.sprtd;
  instTypes_[instTypes_size_].blksCycle = instTypeInfo.blksCycle;
  instTypes_size_++;
}

InstType MachineModel::getDefaultInstType() const {
  return GetInstTypeByName("Default");
}

InstType MachineModel::getDefaultIssueType() const {
  return GetIssueTypeByName("Default");
}

const string &MachineModel::GetModelName() const { return mdlName_; }

int MachineModel::GetInstTypeCnt() const { return instTypes_size_; }

__host__ __device__
int MachineModel::GetIssueTypeCnt() const { return issueTypes_size_; }

__host__ __device__
int MachineModel::GetIssueRate() const { return issueRate_; }

__host__ __device__
int16_t MachineModel::GetRegTypeCnt() const {
  return (int16_t)registerTypes_size_;
}

void MachineModel::CopyPointersToDevice(MachineModel *dev_machMdl) {
  //Copy over the vectors into arrays on device
  InstTypeInfo *dev_instTypes = NULL;
  
  //allocate device memory
  if (cudaSuccess != cudaMalloc((void**)&dev_instTypes, 
			        instTypes_size_ * sizeof(InstTypeInfo)))
    printf("Error allocating device mem for dev_instTypes: %s\n",
	   cudaGetErrorString(cudaGetLastError()));
  
  //copy instTypes_ to device
  if (cudaSuccess != cudaMemcpy(dev_instTypes, instTypes_, 
			        instTypes_size_ * sizeof(InstTypeInfo),
				cudaMemcpyHostToDevice))
    printf("Error copying instTypes_ to device: %s\n",
	   cudaGetErrorString(cudaGetLastError()));

  //update pointer dev_machMdl->instTypes_ to device pointer
  if (cudaSuccess != cudaMemcpy(&dev_machMdl->instTypes_, &dev_instTypes, 
			        sizeof(InstTypeInfo *), 
				cudaMemcpyHostToDevice))
    printf("Error updating dev_machMdl->instTypes_: %s\n",
	   cudaGetErrorString(cudaGetLastError()));

  //debug
  printf("Copied instTypes_ to device!\n");

  RegTypeInfo *dev_registerTypes = NULL;

  //allocate device memory
  if (cudaSuccess != cudaMalloc((void**)&dev_registerTypes, registerTypes_size_ * sizeof(RegTypeInfo)))
    printf("Error allocating device mem for dev_registerTypes: %s\n", cudaGetErrorString(cudaGetLastError()));

  //copy registerTypes_ to device
  if (cudaSuccess != cudaMemcpy(dev_registerTypes, registerTypes_, registerTypes_size_ * sizeof(RegTypeInfo), cudaMemcpyHostToDevice))
    printf("Error copying registerTypes_ to device: %s\n", cudaGetErrorString(cudaGetLastError()));

  //update device pointer dev_machMdl->registerTypes_
  if (cudaSuccess != cudaMemcpy(&dev_machMdl->registerTypes_, &dev_registerTypes, sizeof(RegTypeInfo *), cudaMemcpyHostToDevice))
    printf("Error updating registerTypes_ pointer on device: %s\n", cudaGetErrorString(cudaGetLastError()));

  //debug
  printf("Copied registerTypes_ to device!\n");

  IssueTypeInfo *dev_issueTypes = NULL;

  //allocate device memory
  if (cudaSuccess != cudaMalloc((void**)&dev_issueTypes, issueTypes_size_ * sizeof(IssueTypeInfo)))
    printf("Error allocating device mem for dev_issueTypes: %s\n", cudaGetErrorString(cudaGetLastError()));

  //copy issueTypes_ to device
  if (cudaSuccess != cudaMemcpy(dev_issueTypes, issueTypes_, issueTypes_size_ * sizeof(IssueTypeInfo), cudaMemcpyHostToDevice))
    printf("Error copying issueTypes_ to device: %s\n", cudaGetErrorString(cudaGetLastError()));

  //update device pointer dev_machMdl->issueTypes_
  if (cudaSuccess != cudaMemcpy(&(dev_machMdl->issueTypes_), &dev_issueTypes, sizeof(IssueTypeInfo *), cudaMemcpyHostToDevice))
    printf("Error updating issueTypes_ pointer on device: %s\n", cudaGetErrorString(cudaGetLastError()));

  //debug
  printf("Copied issueTypes_ to device!\n");
}
