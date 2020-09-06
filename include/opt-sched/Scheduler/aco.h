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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <memory>
#include <utility>

namespace llvm {
namespace opt_sched {

typedef double pheromone_t;

struct Choice {
  SchedInstruction *inst;
  double heuristic;  // range 1 to 2
  InstCount readyOn; // number of cycles until this instruction becomes ready
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
  void setInitialSched(InstSchedule *Sched);

private:
  pheromone_t &Pheromone(SchedInstruction *from, SchedInstruction *to);
  pheromone_t &Pheromone(InstCount from, InstCount to);
  double Score(SchedInstruction *from, Choice choice);
  bool shouldReplaceSchedule(InstSchedule* OldSched, InstSchedule* NewSched);

  void PrintPheromone();

  // pheromone Graph Debugging start
  llvm::SmallSet<std::string, 0> DbgRgns;
  llvm::SmallSet<std::pair<InstCount, InstCount>, 0> AntEdges;
  llvm::SmallSet<std::pair<InstCount, InstCount>, 0> CrntAntEdges;
  llvm::SmallSet<std::pair<InstCount, InstCount>, 0> IterAntEdges;
  llvm::SmallSet<std::pair<InstCount, InstCount>, 0> BestAntEdges;
  std::map<std::pair<InstCount, InstCount>, double> LastHeu;
  bool IsDbg = false;
  std::string OutPath;
  std::string graphDisplayAnnotation(int Frm, int To);
  std::string getHeuIfPossible(int Frm, int To);
  void writePheromoneGraph(std::string Stage);
  void writePGraphRecursive(FILE *Out, SchedInstruction *Ins,
                            llvm::SetVector<SchedInstruction *> &Visited);

  // pheromone Graph Debugging end

  Choice SelectInstruction(const llvm::ArrayRef<Choice> &ready,
                           SchedInstruction *lastInst);
  void UpdatePheromone(InstSchedule *schedule);
  std::unique_ptr<InstSchedule> FindOneSchedule();
  llvm::SmallVector<pheromone_t, 0> pheromone_;
  pheromone_t initialValue_;
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
  std::unique_ptr<InstSchedule> InitialSchedule;
  bool VrfySched_;
  pheromone_t ScRelMax;
};

} // namespace opt_sched
} // namespace llvm

#endif
