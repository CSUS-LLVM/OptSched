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

// If set to 1 ACO is run on device
#define DEV_ACO 1
// setting to 1 locks ACO to iterations_without_improvement iterations
#define RUNTIME_TESTING 0
// Minimum region node count. Doesn't make sence to launch DEV_ACO on small rgns
#define REGION_MIN_SIZE 10
#define MANY_ANT_MIN_SIZE 100
// use edge count to approximate memory usage, using nodeCnt reflect
// memory usage as well. Smaller node count DAGs can use more memory.
#define REGION_MAX_EDGE_CNT 800000
#define NUMBLOCKSMANYANTS 80
#define NUMTHREADSPERBLOCK 64

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

class ACOScheduler : public ConstrainedScheduler {
public:
  ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel,
               InstCount upperBound, SchedPriorities priorities,
               bool vrfySched, bool IsPostBB,  int numBlocks,
               SchedRegion *dev_rgn = NULL, DataDepGraph *dev_DDG = NULL,
	             MachineModel *dev_MM = NULL, void *dev_states = NULL);
  __host__
  virtual ~ACOScheduler();
  FUNC_RESULT FindSchedule(InstSchedule *schedule, SchedRegion *region, 
		           ACOScheduler *dev_AcoSchdulr = NULL);
  __host__
  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum);
  // Set the initial schedule for ACO
  // Default is NULL if none are set.
  void setInitialSched(InstSchedule *Sched);
  // Copies the objects pointed to by ACOSched to device
  void CopyPointersToDevice(ACOScheduler *dev_ACOSchedur);
  // Copies the current pheromone values to device pheromone array
  void CopyPheromonesToDevice(ACOScheduler *dev_AcoSchdulr);
  // Calls hipFree on all arrays/objects that were allocated with hipMalloc
  void FreeDevicePointers();
  // Allocates device arrays of size numThreads_ of dynamic variables to allow
  // each thread to have its own value
  void AllocDevArraysForParallelACO();
  // Finds a schedule, if passed a device side schedule, use that instead
  // of creating a new one
  __host__ __device__
  InstSchedule *FindOneSchedule(InstCount RPTarget,
                                InstSchedule *dev_schedule = NULL);
  __host__ __device__
  void UpdatePheromone(InstSchedule *schedule, bool isIterationBest);
  __host__ __device__
  void ScalePheromoneTable();
  // Copies pheromone table to passed shared memory array
  __device__ 
  void CopyPheromonesToSharedMem(double *s_pheromone);
  __host__ __device__
  bool shouldReplaceSchedule(InstSchedule *OldSched, InstSchedule *NewSched,
                             bool IsGlobal, InstCount RPTarget);
  __host__ __device__
  InstCount GetNumAntsTerminated() { return numAntsTerminated_; }
  __host__ __device__
  void SetGlobalBestStalls(int stalls) { globalBestStalls_ = stalls; }
  __host__ __device__
  int GetGlobalBestStalls() { return globalBestStalls_; }
  __host__ __device__
  void SetScRelMax(pheromone_t inScRelMax) { ScRelMax = inScRelMax; }
  __host__ __device__
  int GetNumBlocks() { return numBlocks_; }
  __host__ __device__
  int GetNumThreads() { return numThreads_; }
  __host__ __device__
  void PrintPheromone();
    // Holds state for each thread for RNG
  void *dev_states_;
private:
  __host__ __device__
  pheromone_t &Pheromone(SchedInstruction *from, SchedInstruction *to);
  __host__ __device__
  pheromone_t &Pheromone(InstCount from, InstCount to);
  __host__ __device__
  pheromone_t Score(InstCount FromId, InstCount ToId, HeurType ToHeuristic);
  DCF_OPT ParseDCFOpt(const std::string &opt);
  __host__ __device__
  InstCount SelectInstruction(SchedInstruction *lastInst, InstCount totalStalls, 
                              SchedRegion *rgn, bool &unnecessarilyStalling, 
                              bool closeToRPTarget, bool currentlyWaiting);
  __host__ __device__
  void UpdateACOReadyList(SchedInstruction *Inst);

  DeviceVector<pheromone_t> pheromone_;
  // new ds representations
  ACOReadyList *readyLs;
  KeysHelper *kHelper;
  pheromone_t MaxPriorityInv;
  InstCount MaxScoringInst;

  // new ds representations for device
  ACOReadyList *dev_readyLs;
  KeysHelper *dev_kHelper;
  InstCount *dev_MaxScoringInst;
  
  // True if pheromone_.elmnts_ alloced on device
  bool dev_pheromone_elmnts_alloced_;
  pheromone_t initialValue_;
  bool use_fixed_bias;
  int count_;
  int heuristicImportance_;
  bool use_tournament;
  int fixed_bias;
  double bias_ratio;
  double local_decay;
  double decay_factor;
  int noImprovementMax;
  bool print_aco_trace;
  InstSchedule *InitialSchedule;
  bool VrfySched_;
  bool IsPostBB;
  bool IsTwoPassEn;
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
  int numBlocks_, numThreads_;
  int *dev_RP0OrPositiveCount;
  int RP0OrPositiveCount;

};

} // namespace opt_sched
} // namespace llvm

#endif
