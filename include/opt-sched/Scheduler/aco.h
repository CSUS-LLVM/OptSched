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
#include "opt-sched/Scheduler/device_vector.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace llvm {
namespace opt_sched {

typedef double pheremone_t;

// If set to 1 ACO is run on device
#define DEV_ACO 1
#define NUMBLOCKS 20
#define NUMTHREADSPERBLOCK 128
#define NUMTHREADS NUMBLOCKS * NUMTHREADSPERBLOCK

struct Choice {
  SchedInstruction *inst;
  double heuristic; // range 0 to 1
};

class ACOScheduler : public ConstrainedScheduler {
public:
  ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel,
               InstCount upperBound, SchedPriorities priorities,
               bool vrfySched, SchedRegion *dev_rgn = NULL,
	       DataDepGraph *dev_DDG = NULL,
	       DeviceVector<Choice> **dev_ready = NULL,
	       MachineModel *dev_MM = NULL, curandState_t *dev_states = NULL);
  __host__ __device__
  virtual ~ACOScheduler();
  FUNC_RESULT FindSchedule(InstSchedule *schedule, SchedRegion *region, 
		           ACOScheduler *dev_AcoSchdulr = NULL);
  __host__ __device__
  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum);
  // Set the initial schedule for ACO
  // Default is NULL if none are set.
  void setInitialSched(InstSchedule *Sched);
  // Copies the objects pointed to by ACOSched to device
  void CopyPointersToDevice(ACOScheduler *dev_ACOSchedur);
  // Copies the current pheremone values to device pheremone array
  void CopyPheremonesToDevice(ACOScheduler *dev_AcoSchdulr);
  // Calls cudaFree on all arrays/objects that were allocated with cudaMalloc
  void FreeDevicePointers();
  // Allocates device arrays of size NUMTHREADS of dynamic variables to allow
  // each thread to have its own value
  void AllocDevArraysForParallelACO();
  // Finds a schedule, if passed a device side schedule, use that instead
  // of creating a new one
  __host__ __device__
  InstSchedule *FindOneSchedule(InstSchedule *dev_schedule = NULL, 
		                DeviceVector<Choice> *dev_ready = NULL);
private:
  __host__ __device__
  pheremone_t &Pheremone(SchedInstruction *from, SchedInstruction *to);
  __host__ __device__
  pheremone_t &Pheremone(InstCount from, InstCount to);
  __host__ __device__
  double Score(SchedInstruction *from, Choice choice);

  __host__ __device__
  void PrintPheremone();

  __host__ __device__
  SchedInstruction *SelectInstruction(DeviceVector<Choice> &ready,
                                      SchedInstruction *lastInst);
  __host__ __device__
  void UpdatePheremone(InstSchedule *schedule);
  //__host__ __device__
  //InstSchedule *FindOneSchedule(InstSchedule *dev_schedule = NULL);
  DeviceVector<pheremone_t> pheremone_;
  // True if pheremone_.elmnts_ alloced on device
  bool dev_pheremone_elmnts_alloced_;
  pheremone_t initialValue_;
  bool use_fixed_bias;
  int count_;
  int heuristicImportance_;
  bool use_tournament;
  int fixed_bias;
  double bias_ratio;
  double local_decay;
  double decay_factor;
  int ants_per_iteration;
  int noImprovementMax;
  bool print_aco_trace;
  InstSchedule *InitialSchedule;
  bool VrfySched_;
  DataDepGraph *dev_DDG_;
  DeviceVector<Choice> **dev_ready_;
  MachineModel *dev_MM_;
  // Holds state for each thread for RNG
  curandState_t *dev_states_;
  int returnLastInstCnt_;
};

} // namespace opt_sched
} // namespace llvm

#endif
