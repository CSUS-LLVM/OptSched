/*******************************************************************************
Description:  Defines a scheduling region for basic blocks whose scheduler takes
              into account the cost of spilled registers.
Author:       Ghassan Shobaki
Created:      Unknown
Last Update:  Apr. 2011
*******************************************************************************/

#ifndef OPTSCHED_SPILL_BB_SPILL_H
#define OPTSCHED_SPILL_BB_SPILL_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/OptSchedTarget.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <set>
#include <vector>

namespace llvm {
    namespace opt_sched {

        class LengthCostEnumerator;
        class EnumTreeNode;
        class Register;
        class RegisterFile;
        class BitVector;

        class BBWithSpill : public SchedRegion {
        private:
            LengthCostEnumerator *enumrtr_;

            InstCount crntSpillCost_;
            // FIXME: Unused variable
            InstCount optmlSpillCost_;

            // The target machine
            const OptSchedTarget *OST;

            bool enblStallEnum_;
            int SCW_;
            int schedCostFactor_;

            bool SchedForRPOnly_;

            int16_t regTypeCnt_;
            RegisterFile *regFiles_;

            // A bit vector indexed by register number indicating whether that
            // register is live
            WeightedBitVector *liveRegs_;

            // A bit vector indexed by physical register number indicating whether
            // that physical register is live
            WeightedBitVector *livePhysRegs_;

            // Sum of lengths of live ranges. This vector is indexed by register type,
            // and each type will have its sum of live interval lengths computed.
            std::vector<int> sumOfLiveIntervalLengths_;

            InstCount staticSlilLowerBound_ = 0;

            // (Chris): The dynamic lower bound for SLIL is calculated differently from
            // the other cost functions. It is first set when the static lower bound is
            // calculated.
            InstCount dynamicSlilLowerBound_ = 0;

            // FIXME: Unused variable
            int entryInstCnt_;
            // FIXME: Unused variable
            int exitInstCnt_;
            int schduldEntryInstCnt_;
            int schduldExitInstCnt_;
            int schduldInstCnt_;

            InstCount *spillCosts_;
            // Current register pressure for each register type.
            SmallVector<unsigned, 8> regPressures_;
            InstCount *peakRegPressures_;
            InstCount crntStepNum_;
            InstCount peakSpillCost_;
            InstCount totSpillCost_;
            InstCount slilSpillCost_;
            // FIXME: Unused variable
            bool trackLiveRangeLngths_;

            // Virtual Functions:
            // Given a schedule, compute the cost function value
            InstCount CmputNormCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                                     InstCount &execCost, bool trackCnflcts) override;
            InstCount CmputCost_(InstSchedule *sched, COST_COMP_MODE compMode,
                                 InstCount &execCost, bool trackCnflcts) override;
            void CmputSchedUprBound_() override;
            Enumerator *AllocEnumrtr_(Milliseconds timeout) override;
            FUNC_RESULT Enumerate_(Milliseconds startTime, Milliseconds rgnDeadline,
                                   Milliseconds lngthDeadline) override;
            void SetupForSchdulng_() override;
            void FinishHurstc_() override;
            void FinishOptml_() override;
            void CmputAbslutUprBound_() override;
            ConstrainedScheduler *AllocHeuristicScheduler_() override;
            bool EnableEnum_() override;

            // BBWithSpill-specific Functions:
            // FIXME: Not implemented
            InstCount CmputCostLwrBound_(InstCount schedLngth);
            // FIXME: Not implemented
            InstCount CmputCostLwrBound_();
            void InitForCostCmputtn_();
            // FIXME: Not implemented
            InstCount CmputDynmcCost_();

            void UpdateSpillInfoForSchdul_(SchedInstruction *inst, bool trackCnflcts);
            void UpdateSpillInfoForUnSchdul_(SchedInstruction *inst);
            void SetupPhysRegs_();
            void CmputCrntSpillCost_();
            bool ChkSchedule_(InstSchedule *bestSched, InstSchedule *lstSched) override;
            // FIXME: Unused method
            void CmputCnflcts_(InstSchedule *sched);

        public:
            BBWithSpill(const OptSchedTarget *OST_, DataDepGraph *dataDepGraph,
                        long rgnNum, int16_t sigHashSize, LB_ALG lbAlg,
                        SchedPriorities hurstcPrirts, SchedPriorities enumPrirts,
                        bool vrfySched, Pruning PruningStrategy, bool SchedForRPOnly,
                        bool enblStallEnum, int SCW, SPILL_COST_FUNCTION spillCostFunc,
                        SchedulerType HeurSchedType);
            ~BBWithSpill() override;

            int CmputCostLwrBound() override;

            InstCount UpdtOptmlSched(InstSchedule *crntSched,
                                     LengthCostEnumerator *enumrtr) override;
            bool ChkCostFsblty(InstCount trgtLngth, EnumTreeNode *treeNode) override;
            void SchdulInst(SchedInstruction *inst, InstCount cycleNum, InstCount slotNum,
                            bool trackCnflcts) override;
            void UnschdulInst(SchedInstruction *inst, InstCount cycleNum,
                              InstCount slotNum, EnumTreeNode *trgtNode) override;
            void SetSttcLwrBounds(EnumTreeNode *node) override;
            bool ChkInstLglty(SchedInstruction *inst) override;
            void InitForSchdulng() override;

        protected:
            // (Chris)
            inline const std::vector<int> &GetSLIL_() const override {
                return sumOfLiveIntervalLengths_;
            }
        };

    } // namespace opt_sched
} // namespace llvm

#endif
