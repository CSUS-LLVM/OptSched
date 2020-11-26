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
//#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <cuda_runtime.h>

namespace llvm {
namespace opt_sched {

typedef double pheremone_t;

struct Choice {
  SchedInstruction *inst;
  double heuristic; // range 0 to 1
};

class ACOScheduler : public ConstrainedScheduler {
public:
  ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel,
               InstCount upperBound, SchedPriorities priorities,
               bool vrfySched, SchedRegion **dev_rgn, DataDepGraph *dev_DDG,
	       DeviceVector<Choice> **dev_ready, MachineModel *dev_MM);
  __host__ __device__
  virtual ~ACOScheduler();
  FUNC_RESULT FindSchedule(InstSchedule *schedule, SchedRegion *region);
  __host__ __device__
  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum);
  // Set the initial schedule for ACO
  // Default is NULL if none are set.
  void setInitialSched(InstSchedule *Sched);
  // Copies the objects pointed to by ACOSched to device
  void CopyPointersToDevice(ACOScheduler *dev_ACOSchedulr,
		            DataDepGraph *dev_DDG, MachineModel *dev_machMdl,
			    InstSchedule *dev_InitSched);
  // Calls cudaFree on all arrays/objects that were allocated with cudaMalloc
  void FreeDevicePointers();

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
  SchedRegion **dev_rgn_;
  DataDepGraph *dev_DDG_;
  DeviceVector<Choice> **dev_ready_;
  MachineModel *dev_MM_;
};

} // namespace opt_sched
} // namespace llvm

#endif
