//===-- OptSchedDDGWrapperBasic.h - Basic DDG Wrapper -----------*- C++ -*-===//
//
// Target independent conversion from LLVM ScheduleDAG to OptSched DDG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_SCHED_DDG_WRAPPER_BASIC_H
#define LLVM_OPT_SCHED_DDG_WRAPPER_BASIC_H

#include "OptSchedMachineWrapper.h"
#include "OptimizingScheduler.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <map>
#include <set>
#include <vector>

using namespace llvm;

namespace llvm {
    namespace opt_sched {

        class LLVMRegTypeFilter;
        class ScheduleDAGOptSched;
        class OptSchedMachineModel;

        class OptSchedDDGWrapperBasic : public DataDepGraph {
        public:
            OptSchedDDGWrapperBasic(llvm::MachineSchedContext *Context,
                                    ScheduleDAGOptSched *DAG, OptSchedMachineModel *MM,
                                    LATENCY_PRECISION LatencyPrecision,
                                    const std::string &RegionID);

            ~OptSchedDDGWrapperBasic() = default;

            // Counts the maximum number of virtual registers of each type used by the
            // graph.
            virtual void countDefs();

            // Counts the number of definitions and usages for each register and updates
            // instructions to point to the registers they define/use.
            virtual void addDefsAndUses();

            /// Dump Optsched register def/use information for the region.
            void dumpOptSchedRegisters() const;

            void convertSUnits() override;
            void convertRegFiles() override;

        protected:
            // A convenience machMdl_ pointer casted to OptSchedMachineModel*.
            OptSchedMachineModel *MM;

            // The LLVM scheduler root class, used to access environment
            // and target info.
            const llvm::MachineSchedContext *Contex;

            // The LLVM Schedule DAG.
            const ScheduleDAGOptSched *DAG;

            // Precision of latency info
            LATENCY_PRECISION LatencyPrecision;

            // An option to treat data dependencies of type ORDER as data dependencies
            bool TreatOrderDepsAsDataDeps;

            // The maximum DAG size to be scheduled using precise latency information
            int MaxSizeForPreciseLatency;

            // The index of the last "assigned" register for each register type.
            std::vector<int> RegIndices;

            // Count each definition of a virtual register with the same resNo
            // as a seperate register in our model. Each resNo is also associated
            // with multiple pressure sets which are treated as seperate registers
            std::map<unsigned, std::vector<Register *>> LastDef;

            // Allow the DAG builder to filter our register types that have low peak
            // pressure.
            bool ShouldFilterRegisterTypes = false;

            // Should we generate a machine model from LLVM itineraries.
            bool ShouldGenerateMM = false;

            // Use to ignore non-critical register types.
            std::unique_ptr<LLVMRegTypeFilter> RTFilter;

            // Check if two nodes are equivalent so that we can order them arbitrarily
            bool nodesAreEquivalent(const llvm::SUnit &SrcNode,
                                    const llvm::SUnit &DstNode);

            // Get the weight of the regsiter class in LLVM
            int getRegisterWeight(const unsigned ResNo) const;

            // Add a live-in register.
            void addLiveInReg(unsigned ResNo);

            // Add a live-out register.
            void addLiveOutReg(unsigned ResNo);

            // Add a Use.
            void addUse(unsigned ResNo, InstCount NodeIndex);

            // Add a Def.
            void addDef(unsigned ResNo, InstCount NodeIndex);

            // Add registers that are defined-and-not-used.
            void addDefAndNotUsed(Register *Reg);

            // Returns the register pressure set types of an instruction result.
            std::vector<int> getRegisterType(unsigned RegUnit) const;

            // Setup artificial root.
            void setupRoot();

            // Setup artificial leaf.
            void setupLeaf();

            // Create an optsched graph node and instruction from an llvm::SUnit.
            void convertSUnit(const llvm::SUnit &SU);

            // Create edges between optsched graph nodes using SUnit successors.
            void convertEdges(const llvm::SUnit &SU);

            // Count number or registers defined by the region boundary.
            void countBoundaryLiveness(std::vector<int> &RegDefCounts,
                                       std::set<unsigned> &Defs,
                                       const llvm::MachineInstr *MI);

            // Find liveness info generated by the region boundary.
            void discoverBoundaryLiveness(const llvm::MachineInstr *MI);

            // Holds a register live range, mapping a producer to a set of consumers.
            struct LiveRange {
                // The node which defines the register tracked by this live range.
                SchedInstruction *producer;
                // The nodes which use the register tracked by this live range.
                std::vector<SchedInstruction *> consumers;
            };
        };

// Exclude certain registers from being visible to the scheduler. Use LLVM's
// register pressure tracker to find the MAX register pressure for each register
// type (pressure set). If the MAX pressure is below a certain threshold don't
// track that register.
        class LLVMRegTypeFilter {
        private:
            const MachineModel *MM;
            const llvm::TargetRegisterInfo *TRI;
            const std::vector<unsigned> &RegionPressure;
            float RegFilterFactor;
            std::map<const int16_t, bool> RegTypeIDFilteredMap;
            std::map<const char *, bool> RegTypeNameFilteredMap;

            // The current implementation of this class filters register by
            // TRI->getRegPressureSetLimit
            void FindPSetsToFilter();

        public:
            LLVMRegTypeFilter(const MachineModel *MM, const llvm::TargetRegisterInfo *TRI,
                              const std::vector<unsigned> &RegionPressure,
                              float RegFilterFactor = .7f);
            ~LLVMRegTypeFilter() = default;

            // The proportion of the register pressure set limit that a register's Max
            // pressure must be higher than in order to not be filtered out. (default .7)
            // The idea is that there is no point in trying to reduce the register
            // pressure
            // of a register type that is in no danger of causing spilling. If the
            // RegFilterFactor is .7, and a random register type has a pressure limit of
            // 10, then we filter out the register types if the MAX pressure for that type
            // is below 7. (10 * .7 = 7)
            void setRegFilterFactor(float RegFilterFactor);

            // Return true if this register type should be filtered out.
            // Indexed by RegTypeID
            bool shouldFilter(int16_t RegTypeID) const;

            // Return true if this register type should be filtered out.
            // Indexed by RegTypeName
            bool shouldFilter(const char *RegTypeName) const;

            // Return true if this register type should be filtered out.
            // Indexed by RegTypeID
            bool operator[](int16_t RegTypeID) const;

            // Return true if this register type should be filtered out.
            // Indexed by RegTypeName
            bool operator[](const char *RegTypeName) const;
        };

    } // end namespace opt_sched
} // end namespace llvm

#endif // LLVM_OPT_SCHED_DDG_WRAPPER_BASIC_H
