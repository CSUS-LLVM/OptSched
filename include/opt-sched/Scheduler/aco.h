/*******************************************************************************
Description:  Implements an Ant colony optimizing scheduler
Author:       Theodore Dubois
Created:      Nov. 2017
Updated By:   Ciprian Elies and Vang Thao
Last Update:  Jan. 2020
*******************************************************************************/

#ifndef OPTSCHED_ACO_H
#define OPTSCHED_ACO_H

#include "opt-sched/Scheduler/gen_sched.h"
#include "opt-sched/Scheduler/simplified_aco_ds.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/device_vector.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <hip/hip_runtime.h>

namespace llvm {
namespace opt_sched {

// setting to 1 locks ACO to iterations_without_improvement iterations
#define RUNTIME_TESTING 0
// Minimum region node count. Doesn't make sence to launch DEV_ACO on small rgns
#define REGION_MIN_SIZE 10
#define MANY_ANT_MIN_SIZE 100
// use edge count to approximate memory usage, using nodeCnt reflect
// memory usage as well. Smaller node count DAGs can use more memory.
#define REGION_MAX_EDGE_CNT 800000
#define NUMBLOCKSMANYANTS 180
#define BLOCKOPTSTALLTHRESHOLD 135

enum class DCF_OPT {
  OFF,
  GLOBAL_ONLY,
  GLOBAL_AND_TIGHTEN,
  GLOBAL_AND_ITERATION
};

struct Choice {
  SchedInstruction *inst;
  double heuristic; // range 1 to 2
  InstCount readyOn; // number of cycles until this instruction becomes ready
  double Score;
};

struct BlockDecision {
  int heurChoice; // range 0-1, which heuristic will be used
  int blockOccupancyNum; // range 0-(difference in occupancy from AMD's schedule)
};

class ACOScheduler : public ConstrainedScheduler {
public:
  ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel,
               InstCount upperBound, SchedPriorities priorities1,
               SchedPriorities priorities2, bool vrfySched, bool IsPostBB, int numBlocks,
               SchedRegion *dev_rgn = NULL, DataDepGraph *dev_DDG = NULL, MachineModel *dev_MM = NULL,
               void *dev_states = NULL, int numDiffOccupancies = 1, int targetOccupancy = 0);
  __host__
  virtual ~ACOScheduler();
  FUNC_RESULT FindSchedule(InstSchedule *schedule, std::vector<InstSchedule *> &SchedsAtDiffOccupancies,
                          SchedRegion *region, ACOScheduler *dev_AcoSchdulr = NULL);
  __host__
  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum);
  // Set the initial schedule for ACO
  // Default is NULL if none are set.
  void setInitialSched(InstSchedule *Sched);
  // Copies the objects pointed to by ACOSched to device
  void CopyPointersToDevice(ACOScheduler *dev_ACOSchedur, bool IsSecondPass);
  // Copies the current pheromone values to device pheromone array
  void CopyPheromonesToDevice(ACOScheduler *dev_AcoSchdulr);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  void FreeDevicePointers(bool IsSecondPass);
  // Allocates device arrays of size numThreads_ of dynamic variables to allow
  // each thread to have its own value
  void AllocDevArraysForParallelACO();
  // Finds a schedule, if passed a device side schedule, use that instead
  // of creating a new one
  __host__ __device__
  InstSchedule *FindOneSchedule(InstCount RPTarget,
                                InstSchedule *dev_schedule = NULL, int kernelNum = -1);
  __host__ __device__
  void UpdatePheromone(InstSchedule *schedule, bool isIterationBest, int kernelNum = -1);
  __host__ __device__
  void ScalePheromoneTable(int kernelNum);
  // Copies pheromone table to passed shared memory array
  __device__ 
  void CopyPheromonesToSharedMem(double *s_pheromone, int kernelNum);
  __host__ __device__
  bool shouldReplaceSchedule(InstSchedule *OldSched, InstSchedule *NewSched,
                             bool IsGlobal, InstCount RPTarget, int occupancyTarget);
  __host__ __device__
  InstCount GetNumAntsTerminated() { return numAntsTerminated_; }
  __host__ __device__
  void SetGlobalBestStalls(int stalls) { globalBestStalls_ = stalls; }
  __host__ __device__
  int GetGlobalBestStalls() { return globalBestStalls_; }
  __host__ __device__
  void SetScRelMax(pheromone_t inScRelMax) { ScRelMax = inScRelMax; }
  
  __host__ __device__
  void setNumDiffOccupancies(int numDiffOccupancies) { numDiffOccupancies_ = numDiffOccupancies; }
  __host__ __device__
  void setTargetOccupancy(int targetOccupancy) { targetOccupancy_ = targetOccupancy; }

  __host__ __device__
  int GetNumBlocks() { return numBlocks_; }
  __host__ __device__
  int GetNumThreads() { return numThreads_; }
  __host__ __device__
  void PrintPheromone(int kernelNum = 0);
    // Holds state for each thread for RNG
  void *dev_states_;
  BlockDecision blockDecisions_[NUMBLOCKSMANYANTS];
  void setupBlockDecisions();
  __host__ __device__
  int GetNumDiffOccupancies() { return numDiffOccupancies_; }
  __host__ __device__
  int GetTargetOccupancy() { return targetOccupancy_; }
  
  int globalBestIndex[5];
private:
  __host__ __device__
  pheromone_t &Pheromone(SchedInstruction *from, SchedInstruction *to, int kernelNum = 0);
  __host__ __device__
  pheromone_t &Pheromone(InstCount from, InstCount to, int kernelNum = 0);
  __host__ __device__
  pheromone_t Score(InstCount FromId, InstCount ToId, HeurType ToHeuristic, bool IsFirstPass, int kernelNum = 0);
  DCF_OPT ParseDCFOpt(const std::string &opt);
  __host__ __device__
  InstCount SelectInstruction(SchedInstruction *lastInst, InstCount totalStalls, 
                              SchedRegion *rgn, bool &unnecessarilyStalling, 
                              bool closeToRPTarget, bool currentlyWaiting, int kernelNum = -1);
  __host__ __device__
  void UpdateACOReadyList(SchedInstruction *Inst, bool IsSecondPass, int heurChoice = 0);

  DeviceVector<pheromone_t> pheromone_;
  // new ds representations
  ACOReadyList *readyLs;
  KeysHelper1 *kHelper1;
  KeysHelper2 *kHelper2;
  pheromone_t MaxPriorityInv;
  pheromone_t MaxPriorityInv2;
  InstCount MaxScoringInst;

  // new ds representations for device
  ACOReadyList *dev_readyLs;
  KeysHelper1 *dev_kHelper1;
  KeysHelper2 *dev_kHelper2;
  InstCount *dev_MaxScoringInst;
  
  // True if pheromone_.elmnts_ alloced on device
  bool dev_pheromone_elmnts_alloced_;
  pheromone_t initialValue_;
  bool use_fixed_bias;
  int count_;
  int heuristicImportance_;
  bool use_tournament;
  bool use_dev_ACO;
  int fixed_bias;
  double bias_ratio;
  double local_decay;
  float decay_factor;
  int noImprovementMax;
  bool print_aco_trace;
  InstSchedule *InitialSchedule;
  bool VrfySched_;
  bool IsPostBB;
  bool IsTwoPassEn;
  bool weightedSecondPass;
  pheromone_t ScRelMax;
  DCF_OPT DCFOption;
  SPILL_COST_FUNCTION DCFCostFn;
  DataDepGraph *dev_DDG_;
  MachineModel *dev_MM_;
  // Used to count how many threads returned last instruction
  int returnLastInstCnt_;
  // Used to count how many ants are terminated early
  int numAntsTerminated_;

  bool justWaited = false;
  int globalBestStalls_ = 0;
  int numBlocks_, numThreads_, numDiffOccupancies_, targetOccupancy_;
  int *dev_RP0OrPositiveCount;
  int RP0OrPositiveCount;

  SchedPriorities priorities1_;
  SchedPriorities priorities2_;

};

} // namespace opt_sched
} // namespace llvm

#endif
