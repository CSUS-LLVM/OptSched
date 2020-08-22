/*******************************************************************************
Description:  Implements generic register and register file classes.
Author:       Ghassan Shobaki
Created:      Jun. 2010
Last Update:  Jun. 2017
*******************************************************************************/

#ifndef OPTSCHED_BASIC_REGISTER_H
#define OPTSCHED_BASIC_REGISTER_H

#include "opt-sched/Scheduler/bit_vector.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/sched_basic_data.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace llvm;

namespace llvm {
namespace opt_sched {

// Represents a a single register of a certain type and tracks the number of
// times this register is defined and used.
class Register {
public:
  Register(int16_t type = 0, int num = 0, int physicalNumber = INVALID_VALUE);

  using InstSetType = SmallPtrSet<const SchedInstruction *, 8>;

  int16_t GetType() const;
  void SetType(int16_t type);

  int GetNum() const;
  void SetNum(int num);

  int GetWght() const;
  void SetWght(int wght);

  bool IsPhysical() const;
  int GetPhysicalNumber() const;
  void SetPhysicalNumber(int physicalNumber);

  void AddUse(const SchedInstruction *inst);
  int GetUseCnt() const;
  const InstSetType &GetUseList() const;
  size_t GetSizeOfUseList() const;
  int GetCrntUseCnt() const;

  void AddDef(const SchedInstruction *inst);
  int GetDefCnt() const;
  const InstSetType &GetDefList() const;
  size_t GetSizeOfDefList() const;

  void AddCrntUse();
  void DelCrntUse();
  void ResetCrntUseCnt();

  void IncrmntCrntLngth();
  void DcrmntCrntLngth();
  void ResetCrntLngth();
  int GetCrntLngth() const;

  bool IsLive() const;
  // Live in registers are defined by the artifical entry node.
  bool IsLiveIn() const;
  void SetIsLiveIn(bool liveIn);
  // Live out registers are used by the artifical exit node.
  bool IsLiveOut() const;
  void SetIsLiveOut(bool liveOut);

  Register &operator=(const Register &rhs);

  void SetupConflicts(int regCnt);
  void ResetConflicts();
  void AddConflict(int regNum, bool isSpillCnddt);
  int GetConflictCnt() const;
  bool IsSpillCandidate() const;

  // Returns true if an insertion actually occurred.
  bool AddToInterval(const SchedInstruction *inst);
  bool IsInInterval(const SchedInstruction *inst) const;
  const InstSetType &GetLiveInterval() const;

  // Returns true if an insertion actually occurred.
  bool AddToPossibleInterval(const SchedInstruction *inst);
  bool IsInPossibleInterval(const SchedInstruction *inst) const;
  const InstSetType &GetPossibleLiveInterval() const;

private:
  int16_t type_;
  int num_;
  int defCnt_;
  int useCnt_;
  int crntUseCnt_;
  int crntLngth_;
  int physicalNumber_;
  BitVector conflicts_;
  bool isSpillCnddt_;
  int wght_;
  bool liveIn_;
  bool liveOut_;

  // (Chris): The OptScheduler's Register class should keep track of all the
  // instructions that defined this register and all the instructions that use
  // this register. This makes it easy to identify any instruction that does
  // not already belong to the live interval of this register. This also
  // requires changes to the way defs and uses are added to this register.
  //
  // A set is used to ensure no duplicates are entered.
  InstSetType uses_;
  InstSetType defs_;

  // (Chris): The live interval set is the set of instructions that are
  // guaranteed to be in this register's live interval. This is computed
  // during the naive and closure static lower bound analysis.
  InstSetType liveIntervalSet_;

  // (Chris): The possible live interval set is the set of instructions that
  // may or may not be added to the live interval of this register. This is
  // computed during the common use lower bound analysis.
  InstSetType possibleLiveIntervalSet_;
};

// Represents a file of registers of a certain type and tracks their usages.
class RegisterFile {
public:
  RegisterFile();
  ~RegisterFile();

  int GetRegCnt() const;
  void SetRegCnt(int regCnt);

  int16_t GetRegType() const;
  void SetRegType(int16_t regType);

  Register *GetReg(int num) const;
  Register *FindLiveReg(int physNum) const;

  void ResetCrntUseCnts();
  void ResetCrntLngths();

  int FindPhysRegCnt();
  int GetPhysRegCnt() const;

  void SetupConflicts();
  void ResetConflicts();
  void AddConflictsWithLiveRegs(int regNum, int liveRegCnt);
  int GetConflictCnt();

  // The number of registers in this register file.
  int getCount() const { return static_cast<int>(Regs.size()); }
  // Increase the size of the register file by one and
  // return the RegNum of the created register.
  Register *getNext();

private:
  int16_t regType_;
  int physRegCnt_;
  mutable SmallVector<std::unique_ptr<Register>, 8> Regs;
};

} // namespace opt_sched
} // namespace llvm

#endif
