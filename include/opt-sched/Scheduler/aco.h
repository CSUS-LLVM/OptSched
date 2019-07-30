// Ant colony optimizing scheduler

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
  ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel, InstCount upperBound, SchedPriorities priorities);
  ~ACOScheduler() override;
  FUNC_RESULT FindSchedule(InstSchedule *schedule, SchedRegion *region) override;
  inline void UpdtRdyLst_(InstCount cycleNum, int slotNum) override;
private:
  pheremone_t &Pheremone(SchedInstruction *from, SchedInstruction *to);
  pheremone_t &Pheremone(InstCount from, InstCount to);
  double Score(SchedInstruction *from, Choice choice);

  void PrintPheremone();

  SchedInstruction *SelectInstruction(std::vector<Choice> ready, SchedInstruction *lastInst);
  void UpdatePheremone(InstSchedule *schedule);
  InstSchedule *FindOneSchedule();
  pheremone_t *pheremone_;
  pheremone_t initialValue_{};
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
  // FIXME: Unused method
  std::vector<double> scores(const std::vector<Choice>& ready, SchedInstruction *last);
};

} // namespace opt_sched
} // namespace llvm

#endif