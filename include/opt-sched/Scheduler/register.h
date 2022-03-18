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
#include <hip/hip_runtime.h>
#include "opt-sched/Scheduler/device_set.h"

using namespace llvm;

namespace llvm {
namespace opt_sched {

// Forward Declaration to treat circular dependence
class SchedInstruction;

// Represents a a single register of a certain type and tracks the number of
// times this register is defined and used.
class Register {
public:
  __host__
  Register(int16_t type = 0, int num = 0, int physicalNumber = INVALID_VALUE);

  using InstSetType = DevicePtrSet<InstCount>;

  __host__ __device__
  int16_t GetType() const;
  __host__
  void SetType(int16_t type);

  __host__ __device__
  int GetNum() const;
  __host__
  void SetNum(int num);

  __host__ __device__
  int GetWght() const;
  __host__
  void SetWght(int wght);

  bool IsPhysical() const;
  __host__ __device__
  int GetPhysicalNumber() const;
  void SetPhysicalNumber(int physicalNumber);

  __host__
  void AddUse(const SchedInstruction *inst);
  __host__ __device__
  int GetUseCnt() const;
  const InstSetType &GetUseList() const;
  size_t GetSizeOfUseList() const;
  __host__ __device__
  int GetCrntUseCnt() const;

  __host__
  void AddDef(const SchedInstruction *inst);
  int GetDefCnt() const;
  const InstSetType &GetDefList() const;
  size_t GetSizeOfDefList() const;

  // Resets uses_/defs_ and their respecive counts
  __device__
  void ResetDefsAndUses();

  __host__ __device__
  void AddCrntUse();
  void DelCrntUse();
  __host__ __device__
  void ResetCrntUseCnt();

  void IncrmntCrntLngth();
  void DcrmntCrntLngth();
  __host__ __device__
  void ResetCrntLngth();
  int GetCrntLngth() const;

  __host__ __device__
  bool IsLive() const;
  // Live in registers are defined by the artifical entry node.
  bool IsLiveIn() const;
  __host__
  void SetIsLiveIn(bool liveIn);
  // Live out registers are used by the artifical exit node.
  __host__ __device__
  bool IsLiveOut() const;
  __host__
  void SetIsLiveOut(bool liveOut);

  Register &operator=(const Register &rhs);

  void SetupConflicts(int regCnt);
  __host__ __device__
  void ResetConflicts();
  __host__ __device__
  void AddConflict(int regNum, bool isSpillCnddt);
  int GetConflictCnt() const;
  bool IsSpillCandidate() const;

  // Returns true if an insertion actually occurred.
  bool AddToInterval(const SchedInstruction *inst);
  __host__ __device__
  bool IsInInterval(const SchedInstruction *inst) const;
  const InstSetType &GetLiveInterval() const;

  // Returns true if an insertion actually occurred.
  bool AddToPossibleInterval(const SchedInstruction *inst);
  __host__ __device__
  bool IsInPossibleInterval(const SchedInstruction *inst) const;
  const InstSetType &GetPossibleLiveInterval() const;

  // Resets liveIntervalSet_ and possibleLiveIntervalSet_ 
  // for reinitialization in the next region
  __device__
  void ResetLiveIntervals();
  // Allocates a device array that holds values for each parallel thread
  void AllocDevArrayForParallelACO(int numThreads);
  // Copies all array/objects to device and links them to device pointer
  void CopyPointersToDevice(Register *dev_reg);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  void FreeDevicePointers();

private:
  int16_t type_;
  int num_;
  int defCnt_;
  int useCnt_;
  int crntUseCnt_;
  // Device array which holds a separate crntUseCnt_ for each thread
  int *dev_crntUseCnt_;
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
  // computed during the common use lower boudn analysis.
  InstSetType possibleLiveIntervalSet_;
};

// Represents a file of registers of a certain type and tracks their usages.
class RegisterFile {
public:
  __host__
  RegisterFile();
  __host__
  ~RegisterFile();

  __host__ __device__
  int GetRegCnt() const;
  __host__
  void SetRegCnt(int regCnt);

  __host__ __device__
  int16_t GetRegType() const;
  __host__ __device__
  void SetRegType(int16_t regType);

  __host__ __device__
  Register *GetReg(int num) const;
  Register *FindLiveReg(int physNum) const;

  __host__ __device__
  void ResetCrntUseCnts();
  __host__ __device__
  void ResetCrntLngths();

  int FindPhysRegCnt();
  __host__ __device__
  int GetPhysRegCnt() const;

  void SetupConflicts();
  __host__ __device__
  void ResetConflicts();
  __host__ __device__
  void AddConflictsWithLiveRegs(int regNum, int liveRegCnt);
  int GetConflictCnt();

  // The number of registers in this register file.
  __host__ __device__
  int getCount() const { return Regs_size_; }
  // Increase the size of the register file by one and
  // return the RegNum of the created register.
  Register *getNext();

  // Resets the RegisterFile and its Registers to blank state
  // so they can be reinitialized for the next region
  __device__
  void Reset();
  // Copies all array/objects to device and links them to device pointer
  void CopyPointersToDevice(RegisterFile *dev_regFile);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  void FreeDevicePointers();

private:
  int16_t regType_;
  int physRegCnt_;
  mutable Register **Regs;
  int Regs_alloc_;
  int Regs_size_;
};

} // namespace opt_sched
} // namespace llvm

#endif
