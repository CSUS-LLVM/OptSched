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
               bool vrfySched);
  virtual ~ACOScheduler();
  FUNC_RESULT FindSchedule(InstSchedule *schedule, SchedRegion *region);
  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum);
  // Set the initial schedule for ACO 
  // Default is NULL if none are set.
  void SetInitialSched(InstSchedule *sched);
  
private:
  pheremone_t &Pheremone(SchedInstruction *from, SchedInstruction *to);
  pheremone_t &Pheremone(InstCount from, InstCount to);
  double Score(SchedInstruction *from, Choice choice);

  void PrintPheremone();

  SchedInstruction *SelectInstruction(std::vector<Choice> ready,
                                      SchedInstruction *lastInst);
  void UpdatePheremone(InstSchedule *schedule);
  InstSchedule *FindOneSchedule();
  pheremone_t *pheremone_;
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
  bool print_aco_trace;
  std::vector<double> scores(std::vector<Choice> ready, SchedInstruction *last);
  InstSchedule* initialSchedule_;
  bool vrfySched_;
};

} // namespace opt_sched
} // namespace llvm

#endif
