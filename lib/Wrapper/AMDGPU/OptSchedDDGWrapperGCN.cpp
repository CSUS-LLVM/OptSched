//===- OptSchedDDGWrapperGCN.cpp - GCN DDG Wrapper ------------------------===//
//
// Conversion from LLVM ScheduleDAG to OptSched DDG for amdgcn target.
//
//===----------------------------------------------------------------------===//

#include "OptSchedDDGWrapperGCN.h"
#include "GCNRegPressure.h"
#include "SIRegisterInfo.h"
#include "opt-sched/Scheduler/register.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "optsched-ddg-wrapper"

using namespace llvm;
using namespace llvm::opt_sched;

OptSchedDDGWrapperGCN::OptSchedDDGWrapperGCN(MachineSchedContext *Context,
                                             ScheduleDAGOptSched *DAG,
                                             OptSchedMachineModel *MM,
                                             LATENCY_PRECISION LatencyPrecision,
                                             const std::string &RegionID)
    : OptSchedDDGWrapperBasic(Context, DAG, MM, LatencyPrecision, RegionID),
      SUnits(DAG->SUnits), LIS(DAG->getLIS()), MRI(DAG->MRI) {}

namespace {

std::unique_ptr<SubRegSet>
createSubRegSet(unsigned Reg, const MachineRegisterInfo &MRI, int16_t Type) {
  return llvm::make_unique<SubRegSet>(
      MRI.getMaxLaneMaskForVReg(Reg).getNumLanes(), Type);
}

// Copied from Target/AMDGPU/GCNRegPressure.cpp
LaneBitmask getDefRegMask(const MachineOperand &MO,
                          const MachineRegisterInfo &MRI) {
  assert(MO.isDef() && MO.isReg() &&
         TargetRegisterInfo::isVirtualRegister(MO.getReg()));

  // We don't rely on read-undef flag because in case of tentative schedule
  // tracking it isn't set correctly yet. This works correctly however since
  // use mask has been tracked before using LIS.
  return MO.getSubReg() == 0
             ? MRI.getMaxLaneMaskForVReg(MO.getReg())
             : MRI.getTargetRegisterInfo()->getSubRegIndexLaneMask(
                   MO.getSubReg());
}

// Copied from Target/AMDGPU/GCNRegPressure.cpp
LaneBitmask getUsedRegMask(const MachineOperand &MO,
                           const MachineRegisterInfo &MRI,
                           const LiveIntervals &LIS) {
  assert(MO.isUse() && MO.isReg() &&
         TargetRegisterInfo::isVirtualRegister(MO.getReg()));

  if (auto SubReg = MO.getSubReg())
    return MRI.getTargetRegisterInfo()->getSubRegIndexLaneMask(SubReg);

  auto MaxMask = MRI.getMaxLaneMaskForVReg(MO.getReg());
  if (MaxMask == LaneBitmask::getLane(0)) // cannot have subregs
    return MaxMask;

  // For a tentative schedule LIS isn't updated yet but livemask should remain
  // the same on any schedule. Subreg defs can be reordered but they all must
  // dominate uses anyway.
  auto SI = LIS.getInstructionIndex(*MO.getParent()).getBaseIndex();
  return getLiveLaneMask(MO.getReg(), SI, LIS, MRI);
}

// Copied from Target/AMDGPU/GCNRegPressure.cpp
SmallVector<RegisterMaskPair, 8>
collectVirtualRegUses(const MachineInstr &MI, const LiveIntervals &LIS,
                      const MachineRegisterInfo &MRI) {
  SmallVector<RegisterMaskPair, 8> Res;
  for (const auto &MO : MI.operands()) {
    if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      continue;
    if (!MO.isUse() || !MO.readsReg())
      continue;

    const auto UsedMask = getUsedRegMask(MO, MRI, LIS);

    auto Reg = MO.getReg();
    auto I =
        std::find_if(Res.begin(), Res.end(), [Reg](const RegisterMaskPair &RM) {
          return RM.RegUnit == Reg;
        });
    if (I != Res.end())
      I->LaneMask |= UsedMask;
    else
      Res.push_back(RegisterMaskPair(Reg, UsedMask));
  }
  return Res;
}

SmallVector<RegisterMaskPair, 8>
collectVirtualRegDefs(const MachineInstr &MI, const LiveIntervals &LIS,
                      const MachineRegisterInfo &MRI) {
  SmallVector<RegisterMaskPair, 8> Res;
  for (const auto &MO : MI.defs()) {
    if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()) ||
        MO.isDead())
      continue;

    const auto DefMask = getDefRegMask(MO, MRI);

    auto Reg = MO.getReg();
    auto I =
        std::find_if(Res.begin(), Res.end(), [Reg](const RegisterMaskPair &RM) {
          return RM.RegUnit == Reg;
        });
    if (I != Res.end())
      I->LaneMask |= DefMask;
    else
      Res.push_back(RegisterMaskPair(Reg, DefMask));
  }
  return Res;
}

SmallVector<RegisterMaskPair, 8>
collectLiveSubRegsAtInstr(const MachineInstr *MI, const LiveIntervals *LIS,
                          const MachineRegisterInfo &MRI, bool After) {
  SlotIndex SI = After ? LIS->getInstructionIndex(*MI).getDeadSlot()
                       : LIS->getInstructionIndex(*MI).getBaseIndex();

  SmallVector<RegisterMaskPair, 8> Res;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = TargetRegisterInfo::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    auto LiveMask = getLiveLaneMask(Reg, SI, *LIS, MRI);
    if (LiveMask.any())
      Res.emplace_back(Reg, LiveMask);
  }
  return Res;
}

} // end anonymous namespace

unsigned OptSchedDDGWrapperGCN::getRegKind(unsigned Reg) const {
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  const auto RC = MRI.getRegClass(Reg);
  auto STI = static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());
  return STI->isSGPRClass(RC) ? SGPR32 : VGPR32;
}

void OptSchedDDGWrapperGCN::convertRegFiles() {
  for (int i = 0; i < MM->GetRegTypeCnt(); i++)
    RegFiles[i].SetRegType(i);

  // Add live-in subregs
  for (const auto &MaskPair :
       collectLiveSubRegsAtInstr(SUnits[0].getInstr(), LIS, MRI, false))
    addSubRegDefs(GetRootInst(), MaskPair.RegUnit, MaskPair.LaneMask, true);

  for (const auto &SU : SUnits) {
    const MachineInstr *MI = SU.getInstr();
    for (const auto &MaskPair : collectVirtualRegUses(*MI, *LIS, MRI))
      addSubRegUses(GetInstByIndx(SU.NodeNum), MaskPair.RegUnit,
                    MaskPair.LaneMask);

    for (const auto &MaskPair : collectVirtualRegDefs(*MI, *LIS, MRI))
      addSubRegDefs(GetInstByIndx(SU.NodeNum), MaskPair.RegUnit,
                    MaskPair.LaneMask);
  }

  // Add live-out subregs
  for (const auto &MaskPair : collectLiveSubRegsAtInstr(
           SUnits[SUnits.size() - 1].getInstr(), LIS, MRI, true))
    addSubRegUses(GetLeafInst(), MaskPair.RegUnit, MaskPair.LaneMask,
                  /*LiveOut=*/true);

  // TODO: Count defined-and-not-used registers as live-out uses to avoid assert
  // errors in OptSched.
  for (int16_t i = 0; i < MM->GetRegTypeCnt(); i++)
    for (int j = 0; j < RegFiles[i].GetRegCnt(); j++) {
      Register *Reg = RegFiles[i].GetReg(j);
      if (Reg->GetUseCnt() == 0)
        addDefAndNotUsed(Reg);
    }

  LLVM_DEBUG(DAG->dumpLLVMRegisters());
  LLVM_DEBUG(dumpOptSchedRegisters());
}

void OptSchedDDGWrapperGCN::addSubRegDefs(SchedInstruction *Instr, unsigned Reg,
                                          const LaneBitmask &LiveMask,
                                          bool LiveIn) {
  if (RegionRegs[Reg] == nullptr)
    RegionRegs[Reg] = createSubRegSet(Reg, MRI, getRegKind(Reg));

  SubRegSet &SubRegs = *RegionRegs[Reg].get();
  RegisterFile &RF = RegFiles[SubRegs.Type];
  unsigned Lane = 0;
  for (auto &ResNo : SubRegs) {
    if ((LiveMask.getLane(Lane) & LiveMask).any()) {
      Register *Reg = RF.getNext();
      ResNo = Reg->GetNum();
      Instr->AddDef(Reg);
      // Weight should always be one since we are only tracking VGPR32 and
      // SGPR32
      Reg->SetWght(1);
      Reg->AddDef(Instr);
      Reg->SetIsLiveIn(LiveIn);
    }
    Lane++;
  }
}

void OptSchedDDGWrapperGCN::addSubRegUses(SchedInstruction *Instr, unsigned Reg,
                                          const LaneBitmask &LiveMask,
                                          bool LiveOut) {
  SubRegSet &SubRegs = *RegionRegs[Reg].get();
  RegisterFile &RF = RegFiles[SubRegs.Type];
  unsigned Lane = 0;
  for (auto &ResNo : SubRegs) {
    if ((LiveMask.getLane(Lane) & LiveMask).any()) {
      Register *Reg = RF.GetReg(ResNo);
      Instr->AddUse(Reg);
      Reg->AddUse(Instr);
      Reg->SetIsLiveOut(LiveOut);
    }
    Lane++;
  }
}
