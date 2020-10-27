#include "opt-sched/Scheduler/register.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm::opt_sched;

__host__ __device__
int16_t Register::GetType() const { return type_; }

__device__ __host__
int Register::GetNum() const { return num_; }

__host__ __device__
int Register::GetWght() const { return wght_; }

void Register::SetType(int16_t type) { type_ = type; }

void Register::SetNum(int num) { num_ = num; }

void Register::SetWght(int wght) { wght_ = wght; }

bool Register::IsPhysical() const { return physicalNumber_ != INVALID_VALUE; }

__host__ __device__
int Register::GetPhysicalNumber() const { return physicalNumber_; }

void Register::SetPhysicalNumber(int physicalNumber) {
  physicalNumber_ = physicalNumber;
}

__host__ __device__
bool Register::IsLive() const {
  assert(crntUseCnt_ <= useCnt_);
  return crntUseCnt_ < useCnt_;
}

bool Register::IsLiveIn() const { return liveIn_; }

__host__ __device__
bool Register::IsLiveOut() const { return liveOut_; }

void Register::SetIsLiveIn(bool liveIn) { liveIn_ = liveIn; }

void Register::SetIsLiveOut(bool liveOut) { liveOut_ = liveOut; }

__host__ __device__
void Register::ResetCrntUseCnt() { crntUseCnt_ = 0; }

void Register::AddUse(const SchedInstruction *inst) {
  uses_.insert(inst->GetNum());
  useCnt_++;
}

void Register::AddDef(const SchedInstruction *inst) {
  defs_.insert(inst->GetNum());
  defCnt_++;
}

__device__ __host__
int Register::GetUseCnt() const { return useCnt_; }

const Register::InstSetType &Register::GetUseList() const { return uses_; }

size_t Register::GetSizeOfUseList() const { return uses_.size(); }

int Register::GetDefCnt() const { return defCnt_; }

const Register::InstSetType &Register::GetDefList() const { return defs_; }

size_t Register::GetSizeOfDefList() const { return defs_.size(); }

__device__
void Register::ResetDefsAndUses() {
  // Only called on device, but will not compile without
  // macro since SmallPtrSet doesnt have a Reset() method
  defs_.Reset();
  defCnt_ = 0;
  uses_.Reset();
  useCnt_ = 0;
}

__device__ __host__
int Register::GetCrntUseCnt() const { return crntUseCnt_; }

__host__ __device__
void Register::AddCrntUse() { crntUseCnt_++; }

void Register::DelCrntUse() { crntUseCnt_--; }

__host__ __device__
void Register::ResetCrntLngth() { crntLngth_ = 0; }

int Register::GetCrntLngth() const { return crntLngth_; }

void Register::IncrmntCrntLngth() { crntLngth_++; }

void Register::DcrmntCrntLngth() { crntLngth_--; }

Register &Register::operator=(const Register &rhs) {
  if (this != &rhs) {
    num_ = rhs.num_;
    type_ = rhs.type_;
  }

  return *this;
}

void Register::SetupConflicts(int regCnt) { conflicts_.Construct(regCnt); }

__host__ __device__
void Register::ResetConflicts() {
  conflicts_.Reset();
  isSpillCnddt_ = false;
}

__host__ __device__
void Register::AddConflict(int regNum, bool isSpillCnddt) {
  assert(regNum != num_);
  assert(regNum >= 0);
  conflicts_.SetBit(regNum, true);
  isSpillCnddt_ = isSpillCnddt_ || isSpillCnddt;
}

int Register::GetConflictCnt() const { return conflicts_.GetOneCnt(); }

bool Register::IsSpillCandidate() const { return isSpillCnddt_; }

bool Register::AddToInterval(const SchedInstruction *inst) {
//#ifdef __CUDA_ARCH__
  return liveIntervalSet_.insert(inst->GetNum());
//#else
  //return liveIntervalSet_.insert(inst->GetNum()).second;
//#endif
}

__host__ __device__
bool Register::IsInInterval(const SchedInstruction *inst) const {
//#ifdef __CUDA_ARCH__
  return liveIntervalSet_.contains(inst->GetNum());
//#else
  //return liveIntervalSet_.count(inst->GetNum()) != 0;
//#endif
}

const Register::InstSetType &Register::GetLiveInterval() const {
  return liveIntervalSet_;
}

bool Register::AddToPossibleInterval(const SchedInstruction *inst) {
//#ifdef __CUDA_ARCH__
  return possibleLiveIntervalSet_.insert(inst->GetNum());
//#else
  //return possibleLiveIntervalSet_.insert(inst->GetNum()).second;
//#endif
}

__host__ __device__
bool Register::IsInPossibleInterval(const SchedInstruction *inst) const {
//#ifdef __CUDA_ARCH__
  return possibleLiveIntervalSet_.contains(inst->GetNum());
//#else
  //return possibleLiveIntervalSet_.count(inst->GetNum()) != 0;
//#endif
}

const Register::InstSetType &Register::GetPossibleLiveInterval() const {
  return possibleLiveIntervalSet_;
}

__device__
void Register::ResetLiveIntervals() {
  // Only called on device, but will not compile without
  // macro since SmallPtrSet doesnt have a Reset() method
//#ifdef __CUDA_ARCH__
  liveIntervalSet_.Reset();
  possibleLiveIntervalSet_.Reset();
//#endif
}

void Register::CopyPointersToDevice(Register *dev_reg) {
  size_t memSize;
  // Copy conflicts->vctr_ to device
  unsigned long *dev_vctr;
  if (conflicts_.GetUnitCnt() > 0) {
    memSize = sizeof(unsigned long) * conflicts_.GetUnitCnt();
    if (cudaSuccess != cudaMalloc(&dev_vctr, memSize))
      Logger::Fatal("Failed to alloc dev mem for conflicts_->vctr");

    if (cudaSuccess != cudaMemcpy(dev_vctr, conflicts_.vctr_, memSize,
                                  cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to copy conflicts_.vctr to device");

    if (cudaSuccess != cudaMemcpy(&dev_reg->conflicts_.vctr_, &dev_vctr,
                                  sizeof(unsigned long *),
                                  cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to update conflicts_->vctr");
  }

  // Copy uses_.elmnt array
  InstCount *dev_elmnt;
  if (uses_.alloc_ > 0) {
    memSize = sizeof(InstCount) * uses_.alloc_;
    if (cudaSuccess != cudaMalloc(&dev_elmnt, memSize))
      Logger::Fatal("Failed to alloc dev mem for dev_uses.elmnt");

    if (cudaSuccess != cudaMemcpy(dev_elmnt, uses_.elmnt, memSize,
			          cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to copy uses_.elmnt to dev");

    if (cudaSuccess != cudaMemcpy(&dev_reg->uses_.elmnt, &dev_elmnt,
			          sizeof(InstCount *), cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to update dev pointers for uses.elmnt");
  }

  // Copy defs_.elmnt array
  if (defs_.alloc_ > 0) {
    memSize = sizeof(InstCount) * defs_.alloc_;
    if (cudaSuccess != cudaMalloc(&dev_elmnt, memSize))
      Logger::Fatal("Failed to alloc dev mem for dev_defs.elmnt");

    if (cudaSuccess != cudaMemcpy(dev_elmnt, defs_.elmnt, memSize,
                                  cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to copy defs_.elmnt to dev");

    if (cudaSuccess != cudaMemcpy(&dev_reg->defs_.elmnt, &dev_elmnt,
                                  sizeof(InstCount *), cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to update dev pointers for defs.elmnt");
  }

  // Copy liveIntervalSet_.elmnt array
  if (liveIntervalSet_.alloc_ > 0) {
    memSize = sizeof(InstCount) * liveIntervalSet_.alloc_;
    if (cudaSuccess != cudaMalloc(&dev_elmnt, memSize))
      Logger::Fatal("Failed to alloc dev mem for dev_liveIntervalSet.elmnt");

    if (cudaSuccess != cudaMemcpy(dev_elmnt, liveIntervalSet_.elmnt, memSize,
                                  cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to copy liveIntervalSet_.elmnt to dev");

    if (cudaSuccess != cudaMemcpy(&dev_reg->liveIntervalSet_.elmnt, &dev_elmnt,
                                  sizeof(InstCount *), cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to update dev pointers for liveIntervalSet.elmnt");
  }

  // Copy possibleLiveIntervalSet_.elmnt array
  if (possibleLiveIntervalSet_.alloc_ > 0) {
    memSize = sizeof(InstCount) * possibleLiveIntervalSet_.alloc_;
    if (cudaSuccess != cudaMalloc(&dev_elmnt, memSize))
      Logger::Fatal("Failed to alloc dev mem for possibleLiveIntervalSet.elmnt");

    if (cudaSuccess != cudaMemcpy(dev_elmnt, possibleLiveIntervalSet_.elmnt, 
			          memSize, cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to copy possibleLiveIntervalSet_.elmnt to dev");

    if (cudaSuccess != cudaMemcpy(&dev_reg->possibleLiveIntervalSet_.elmnt, 
			          &dev_elmnt, sizeof(InstCount *), 
				  cudaMemcpyHostToDevice))
      Logger::Fatal("Failed to updt dev ptr for possibleLiveIntervalSet.elmnt");
  }
}

__host__ __device__
Register::Register(int16_t type, int num, int physicalNumber) {
  type_ = type;
  num_ = num;
  wght_ = 1;
  defCnt_ = 0;
  useCnt_ = 0;
  crntUseCnt_ = 0;
  physicalNumber_ = physicalNumber;
  isSpillCnddt_ = false;
  liveIn_ = false;
  liveOut_ = false;
}

__host__ __device__
RegisterFile::RegisterFile() {
  regType_ = 0;
  Regs = NULL;
  Regs_size_ = Regs_alloc_ = 0;
  physRegCnt_ = 0;
}

__host__ __device__
RegisterFile::~RegisterFile() {
  if (Regs) {
    for (int i = 0; i < Regs_size_; i++)
      delete Regs[i];
    delete[] Regs;
  }
}

__host__ __device__
int RegisterFile::GetRegCnt() const { return getCount(); }

__host__ __device__
int16_t RegisterFile::GetRegType() const { return regType_; }

__host__ __device__
void RegisterFile::SetRegType(int16_t regType) { regType_ = regType; }

__host__ __device__
void RegisterFile::ResetCrntUseCnts() {
  for (int i = 0; i < getCount(); i++) {
    Regs[i]->ResetCrntUseCnt();
  }
}

__host__ __device__
void RegisterFile::ResetCrntLngths() {
  for (int i = 0; i < getCount(); i++) {
    Regs[i]->ResetCrntLngth();
  }
}

Register *RegisterFile::getNext() {
  size_t RegNum = Regs_size_;
  Register *Reg = new Register;
  Reg->SetType(regType_);
  Reg->SetNum(RegNum);
  //Regs.push_back(std::move(Reg));

  if (Regs_alloc_ == Regs_size_) {
    if (Regs_alloc_ > 0)
      Regs_alloc_ *= 2;
    else
      Regs_alloc_ = 2;

    Register **resized = new Register *[Regs_alloc_];
    //copy contents of old array
    for (int i = 0; i < Regs_size_; i++)
      resized[i] = Regs[i];
    
    delete[] Regs;
    Regs = resized;
  }
  Regs[Regs_size_++] = std::move(Reg);

  return Regs[RegNum];
}

__host__ __device__
void RegisterFile::SetRegCnt(int regCnt) {
  if (regCnt == 0)
    return;

  //Regs.resize(regCnt);
  if (Regs_size_ > 0)
    delete[] Regs;
  Regs_size_ = Regs_alloc_ = regCnt;
  Regs = new Register *[Regs_alloc_];

  for (int i = 0; i < getCount(); i++) {
    Register *Reg = new Register;
    Reg->SetType(regType_);
    Reg->SetNum(i);
    Regs[i] = Reg;
  }
}

__host__ __device__
Register *RegisterFile::GetReg(int num) const {
  if (num >= 0 && num < getCount()) {
    return Regs[num];
  } else {
    return NULL;
  }
}

Register *RegisterFile::FindLiveReg(int physNum) const {
  for (int i = 0; i < getCount(); i++) {
    if (Regs[i]->GetPhysicalNumber() == physNum && Regs[i]->IsLive() == true)
      return Regs[i];
  }
  return NULL;
}


int RegisterFile::FindPhysRegCnt() {
  int maxPhysNum = -1;
  for (int i = 0; i < getCount(); i++) {
    if (Regs[i]->GetPhysicalNumber() != INVALID_VALUE &&
        Regs[i]->GetPhysicalNumber() > maxPhysNum)
      maxPhysNum = Regs[i]->GetPhysicalNumber();
  }

  // Assume that physical registers are given sequential numbers
  // starting from 0.
  physRegCnt_ = maxPhysNum + 1;
  return physRegCnt_;
}

__host__ __device__
int RegisterFile::GetPhysRegCnt() const { return physRegCnt_; }

void RegisterFile::SetupConflicts() {
  for (int i = 0; i < getCount(); i++)
    Regs[i]->SetupConflicts(getCount());
}

__host__ __device__
void RegisterFile::ResetConflicts() {
  for (int i = 0; i < getCount(); i++)
    Regs[i]->ResetConflicts();
}

int RegisterFile::GetConflictCnt() {
  int cnflctCnt = 0;
  for (int i = 0; i < getCount(); i++) {
    cnflctCnt += Regs[i]->GetConflictCnt();
  }
  return cnflctCnt;
}

__host__ __device__
void RegisterFile::AddConflictsWithLiveRegs(int regNum, int liveRegCnt) {
  bool isSpillCnddt = (liveRegCnt + 1) > physRegCnt_;
  int conflictCnt = 0;
  for (int i = 0; i < getCount(); i++) {
    if (i != regNum && Regs[i]->IsLive() == true) {
      Regs[i]->AddConflict(regNum, isSpillCnddt);
      Regs[regNum]->AddConflict(i, isSpillCnddt);
      conflictCnt++;
    }
    if (conflictCnt == liveRegCnt)
      break;
  }
}

__device__
void RegisterFile::Reset() {
  ResetConflicts();
  ResetCrntUseCnts();
  ResetCrntLngths();
  
  for (int i = 0; i < getCount(); i++) {
    Regs[i]->ResetDefsAndUses();
    Regs[i]->ResetLiveIntervals();
  }
}

void RegisterFile::CopyPointersToDevice(RegisterFile *dev_regFile) {
  //remove reference to host pointer
  dev_regFile->Regs = NULL;
  //declare and allocate array of pointers
  Register **dev_regs = NULL;
  size_t memSize;
  //allocate device memory
  memSize = getCount() * sizeof(Register *);
  if (cudaSuccess != cudaMalloc((void**)&dev_regs, memSize))
    printf("Error allocating dev mem for dev_regs: %s\n", 
		    cudaGetErrorString(cudaGetLastError()));

  //copy array of host pointers to device
  if (cudaSuccess != cudaMemcpy(dev_regs, Regs, memSize, 
			        cudaMemcpyHostToDevice))
    printf("Error copying Regs to device: %s\n", 
		    cudaGetErrorString(cudaGetLastError()));

  //copy each register to device and update its pointer in dev_regs
  Register *dev_reg = NULL;

  for (int i = 0; i < getCount(); i++) {
    //allocate device memory
    if (cudaSuccess != cudaMalloc((void**)&dev_reg, sizeof(Register)))
      printf("Error allocating dev mem for dev_reg: %s\n", 
		      cudaGetErrorString(cudaGetLastError()));

    //copy register to device
    if (cudaSuccess != cudaMemcpy(dev_reg, Regs[i], sizeof(Register), 
			          cudaMemcpyHostToDevice))
      printf("Error copying Regs[%d] to device: %s\n", i, 
		      cudaGetErrorString(cudaGetLastError()));

    Regs[i]->CopyPointersToDevice(dev_reg);

    //update dev_regs pointer
    if (cudaSuccess != cudaMemcpy(&dev_regs[i], &dev_reg, sizeof(Register *), 
			          cudaMemcpyHostToDevice))
      printf("Error updating dev_regs[%d] on device: %s\n", i, 
		      cudaGetErrorString(cudaGetLastError()));
  }

  //update dev_regFile->Regs pointer
  dev_regFile->Regs = dev_regs;
}
