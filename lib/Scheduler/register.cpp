#include "opt-sched/Scheduler/register.h"

namespace opt_sched {

int16_t Register::GetType() const { return type_; }

int Register::GetNum() const { return num_; }

int Register::GetWght() const { return wght_; }

void Register::SetType(int16_t type) { type_ = type; }

void Register::SetNum(int num) { num_ = num; }

void Register::SetWght(int wght) { wght_ = wght; }

bool Register::IsPhysical() const { return physicalNumber_ != INVALID_VALUE; }

int Register::GetPhysicalNumber() const { return physicalNumber_; }

void Register::SetPhysicalNumber(int physicalNumber) {
  physicalNumber_ = physicalNumber;
}

bool Register::IsLive() const {
  assert(crntUseCnt_ <= useCnt_);
  return crntUseCnt_ < useCnt_;
}

bool Register::IsLiveIn() const { return liveIn_; }

bool Register::IsLiveOut() const { return liveOut_; }

void Register::SetIsLiveIn(bool liveIn) { liveIn_ = liveIn; }

void Register::SetIsLiveOut(bool liveOut) { liveOut_ = liveOut; }

void Register::ResetCrntUseCnt() { crntUseCnt_ = 0; }

void Register::AddUse(const SchedInstruction *inst) {
  uses_.insert(inst);
  useCnt_++;
}

void Register::AddDef(const SchedInstruction *inst) {
  defs_.insert(inst);
  defCnt_++;
}

int Register::GetUseCnt() const { return useCnt_; }

const Register::InstSetType &Register::GetUseList() const { return uses_; }

size_t Register::GetSizeOfUseList() const { return uses_.size(); }

int Register::GetDefCnt() const { return defCnt_; }

const Register::InstSetType &Register::GetDefList() const { return defs_; }

size_t Register::GetSizeOfDefList() const { return defs_.size(); }

int Register::GetCrntUseCnt() const { return crntUseCnt_; }

void Register::AddCrntUse() { crntUseCnt_++; }

void Register::DelCrntUse() { crntUseCnt_--; }

void Register::ResetCrntLngth() { crntLngth_ = 0; }

int Register::GetCrntLngth() const { return crntLngth_; }

void Register::IncrmntCrntLngth() { crntLngth_++; }

void Register::DcrmntCrntLngth() { crntLngth_--; }

const Register &Register::operator=(Register &rhs) {
  if (this != &rhs) {
    num_ = rhs.num_;
    type_ = rhs.type_;
  }

  return *this;
}

void Register::SetupConflicts(int regCnt) { conflicts_.Construct(regCnt); }

void Register::ResetConflicts() {
  conflicts_.Reset();
  isSpillCnddt_ = false;
}

void Register::AddConflict(int regNum, bool isSpillCnddt) {
  assert(regNum != num_);
  assert(regNum >= 0);
  conflicts_.SetBit(regNum, true);
  isSpillCnddt_ = isSpillCnddt_ || isSpillCnddt;
}

int Register::GetConflictCnt() const { return conflicts_.GetOneCnt(); }

bool Register::IsSpillCandidate() const { return isSpillCnddt_; }

bool Register::AddToInterval(const SchedInstruction *inst) {
  return liveIntervalSet_.insert(inst).second;
}

bool Register::IsInInterval(const SchedInstruction *inst) const {
  return liveIntervalSet_.count(inst) != 0;
}

const Register::InstSetType &Register::GetLiveInterval() const {
  return liveIntervalSet_;
}

bool Register::AddToPossibleInterval(const SchedInstruction *inst) {
  return possibleLiveIntervalSet_.insert(inst).second;
}

bool Register::IsInPossibleInterval(const SchedInstruction *inst) const {
  return possibleLiveIntervalSet_.count(inst) != 0;
}

const Register::InstSetType &Register::GetPossibleLiveInterval() const {
  return possibleLiveIntervalSet_;
}

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

RegisterFile::RegisterFile() {
  regCnt_ = 0;
  regs_ = NULL;
  regType_ = 0;
  physRegCnt_ = 0;
}

RegisterFile::~RegisterFile() {
  if (regs_)
    delete[] regs_;
}

int RegisterFile::GetRegCnt() const { return regCnt_; }

int16_t RegisterFile::GetRegType() const { return regType_; }

void RegisterFile::SetRegType(int16_t regType) { regType_ = regType; }

void RegisterFile::ResetCrntUseCnts() {
  for (int i = 0; i < regCnt_; i++) {
    regs_[i].ResetCrntUseCnt();
  }
}

void RegisterFile::ResetCrntLngths() {
  for (int i = 0; i < regCnt_; i++) {
    regs_[i].ResetCrntLngth();
  }
}

void RegisterFile::SetRegCnt(int regCnt) {
  regCnt_ = regCnt;
  if (regCnt_ == 0)
    return;
  if (regs_)
    delete[] regs_;
  regs_ = new Register[regCnt_];

  for (int i = 0; i < regCnt_; i++) {
    regs_[i].SetType(regType_);
    regs_[i].SetNum(i);
  }
}

Register *RegisterFile::GetReg(int num) const {
  if (num >= 0 && num < regCnt_) {
    return regs_ + num;
  } else {
    return NULL;
  }
}

Register *RegisterFile::FindLiveReg(int physNum) const {
  for (int i = 0; i < regCnt_; i++) {
    if (regs_[i].GetPhysicalNumber() == physNum && regs_[i].IsLive() == true)
      return regs_ + i;
  }
  return NULL;
}

int RegisterFile::FindPhysRegCnt() {
  int maxPhysNum = -1;
  for (int i = 0; i < regCnt_; i++) {
    if (regs_[i].GetPhysicalNumber() != INVALID_VALUE &&
        regs_[i].GetPhysicalNumber() > maxPhysNum)
      maxPhysNum = regs_[i].GetPhysicalNumber();
  }

  // Assume that physical registers are given sequential numbers
  // starting from 0.
  physRegCnt_ = maxPhysNum + 1;
  return physRegCnt_;
}

int RegisterFile::GetPhysRegCnt() const { return physRegCnt_; }

void RegisterFile::SetupConflicts() {
  for (int i = 0; i < regCnt_; i++)
    regs_[i].SetupConflicts(regCnt_);
}

void RegisterFile::ResetConflicts() {
  for (int i = 0; i < regCnt_; i++)
    regs_[i].ResetConflicts();
}

int RegisterFile::GetConflictCnt() {
  int cnflctCnt = 0;
  for (int i = 0; i < regCnt_; i++) {
    cnflctCnt += regs_[i].GetConflictCnt();
  }
  return cnflctCnt;
}

void RegisterFile::AddConflictsWithLiveRegs(int regNum, int liveRegCnt) {
  bool isSpillCnddt = (liveRegCnt + 1) > physRegCnt_;
  int conflictCnt = 0;
  for (int i = 0; i < regCnt_; i++) {
    if (i != regNum && regs_[i].IsLive() == true) {
      regs_[i].AddConflict(regNum, isSpillCnddt);
      regs_[regNum].AddConflict(i, isSpillCnddt);
      conflictCnt++;
    }
    if (conflictCnt == liveRegCnt)
      break;
  }
}

} // end namespace opt_sched
