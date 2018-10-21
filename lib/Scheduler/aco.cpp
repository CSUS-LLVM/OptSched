#include <iostream>
#include <iomanip>
#include <sstream>
#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/random.h"

namespace opt_sched {

//static void PrintInstruction(SchedInstruction *inst);
void PrintSchedule(InstSchedule *schedule);

double RandDouble(double min, double max) {
  double rand = (double) RandomGen::GetRand32() / INT32_MAX;
  return (rand * (max - min)) + min;
}

#define USE_ACS 1
#define BIASED_CHOICES 10
#define LOCAL_DECAY 0.1

#if USE_ACS
#define ANTS_PER_ITERATION 10
#define DECAY_FACTOR 0.1
#else
#define ANTS_PER_ITERATION count_
#define DECAY_FACTOR 0.5
#endif

ACOScheduler::ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel, InstCount upperBound, SchedPriorities priorities) : ConstrainedScheduler(dataDepGraph, machineModel, upperBound) {
  prirts_ = priorities;
  rdyLst_ = new ReadyList(dataDepGraph_, priorities);
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();
  heuristicImportance_ = schedIni.GetInt("ACO_HEURISTIC_IMPORTANCE");

  int pheremone_size = (count_ + 1) * count_;
  pheremone_ = new pheremone_t[pheremone_size];
}

ACOScheduler::~ACOScheduler() {
  delete rdyLst_;
  delete[] pheremone_;
}

// Pheremone table lookup
// -1 means no instruction, so e.g. pheremone(-1, 10) gives pheremone on path
// from empty schedule to schedule only containing instruction 10
pheremone_t &ACOScheduler::Pheremone(SchedInstruction *from, SchedInstruction *to) {
  assert(to != NULL);
  int fromNum = -1; 
  if (from != NULL)
    fromNum = from->GetNum();
  return Pheremone(fromNum, to->GetNum());
}

pheremone_t &ACOScheduler::Pheremone(InstCount from, InstCount to) {
  int row = 0;
  if (from != -1)
    row = from + 1;
  return pheremone_[(row * count_) + to];
}

double ACOScheduler::Score(SchedInstruction *from, Choice choice) {
  return Pheremone(from, choice.inst) * pow(choice.heuristic, heuristicImportance_);
}

std::vector<double> ACOScheduler::scores(std::vector<Choice> ready, SchedInstruction *last) {
  std::vector<double> s;
  for (auto choice : ready)
    s.push_back(Score(last, choice));
  return s;
}

SchedInstruction *ACOScheduler::SelectInstruction(std::vector<Choice> ready, SchedInstruction *lastInst) {
#if USE_ACS
  double choose_best_chance = fmax(0, 1 - (double) BIASED_CHOICES / count_);
  if (RandDouble(0, 1) < choose_best_chance) {
    pheremone_t max = -1;
    Choice maxChoice;
    for (auto choice : ready) {
      if (Score(lastInst, choice) > max) {
        max = Score(lastInst, choice);
        maxChoice = choice;
      }
    }
    return maxChoice.inst;
  }
#endif
  pheremone_t sum = 0;
  for (auto choice : ready)
    sum += Score(lastInst, choice);
  pheremone_t point = RandDouble(0, sum);
  for (auto choice : ready) {
    point -= Score(lastInst, choice);
    if (point <= 0)
      return choice.inst;
  }
  std::cerr << "returning last instruction" << std::endl;
  assert(point < 0.001); // floats should not be this inaccurate
  return ready.back().inst;
}

InstSchedule *ACOScheduler::FindOneSchedule() {
  SchedInstruction *lastInst = NULL;
  InstSchedule *schedule = new InstSchedule(machMdl_, dataDepGraph_, true);
  InstCount maxPriority = rdyLst_->MaxPriority();
  if (maxPriority == 0) maxPriority = 1; // divide by 0 is bad
  Initialize_();

  while (!IsSchedComplete_()) {
    // convert the ready list from a custom priority queue to a std::vector,
    // much nicer for this particular scheduler
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
    std::vector<Choice> ready;
    unsigned long heuristic;
    SchedInstruction *inst = rdyLst_->GetNextPriorityInst(heuristic);
    while (inst != NULL) {
      if (ChkInstLglty_(inst)) {
        Choice c;
        c.inst = inst;
        c.heuristic = (double) heuristic / maxPriority;
        ready.push_back(c);
      }
      inst = rdyLst_->GetNextPriorityInst(heuristic);
    }
    rdyLst_->ResetIterator();

    // print out the ready list for debugging
    /* std::stringstream stream; */
    /* stream << "Ready list: "; */
    /* for (auto choice : ready) { */
      /* stream << choice.inst->GetNum() << ", "; */
    /* } */
    /* Logger::Info(stream.str().c_str()); */

    inst = NULL;
    if (!ready.empty())
      inst = SelectInstruction(ready, lastInst);
#ifdef USE_ACS
    pheremone_t *pheremone = &Pheremone(lastInst, inst);
    *pheremone = (1 - LOCAL_DECAY) * *pheremone + LOCAL_DECAY * initialValue_;
#endif
    lastInst = inst;

    // boilerplate, mostly copied from ListScheduler, try not to touch it
    InstCount instNum;
    if (inst == NULL) {
      instNum = SCHD_STALL;
    } else {
      instNum = inst->GetNum();
      SchdulInst_(inst, crntCycleNum_);
      inst->Schedule(crntCycleNum_, crntSlotNum_);
      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
      DoRsrvSlots_(inst);
      // this is annoying
      SchedInstruction *blah = rdyLst_->GetNextPriorityInst();
      while (blah != NULL && blah != inst) {
        blah = rdyLst_->GetNextPriorityInst();
      }
      if (blah == inst)
        rdyLst_->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);
    }
    /* Logger::Info("Chose instruction %d (for some reason)", instNum); */
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
    rdyLst_->ResetIterator();
  }
  rgn_->UpdateScheduleCost(schedule);
  return schedule;
}

FUNC_RESULT ACOScheduler::FindSchedule(InstSchedule *schedule_out, SchedRegion *region) {
  rgn_ = region;
  
  // initialize pheremone
  // for this, we need the cost of the pure heuristic schedule
  int pheremone_size = (count_ + 1) * count_;
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = 1;
  initialValue_ = 1;
  InstCount heuristicCost = FindOneSchedule()->GetSpillCost() + 1; // prevent divide by zero

#if USE_ACS
  initialValue_ = 2.0 / ((double) count_ * heuristicCost);
#else
  initialValue_ = (double) ANTS_PER_ITERATION / heuristicCost;
#endif
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = initialValue_;
  /* std::cout<<initialValue_<<std::endl; */

  InstSchedule *bestSchedule = NULL;
  Config &schedIni = SchedulerOptions::getInstance();
  int noImprovementMax = schedIni.GetInt("ACO_STOP_ITERATIONS");
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  while (true) {
    InstSchedule *iterationBest = NULL;
    for (int i = 0; i < ANTS_PER_ITERATION; i++) {
      InstSchedule *schedule = FindOneSchedule();
      /* PrintSchedule(schedule); */
      if (iterationBest == NULL || schedule->GetCost() < iterationBest->GetCost()) {
        delete iterationBest;
        iterationBest = schedule;
      } else {
        delete schedule;
      }
    }
#if !USE_ACS
    UpdatePheremone(iterationBest);
#endif
    /* PrintSchedule(iterationBest); */
    /* std::cout << iterationBest->GetSpillCost() << std::endl; */
    // TODO DRY
    if (bestSchedule == NULL || iterationBest->GetCost() < bestSchedule->GetCost()) {
      delete bestSchedule;
      bestSchedule = iterationBest;
      Logger::Info("ACO found schedule with spill cost %d", bestSchedule->GetSpillCost());
      noImprovement = 0;
    } else {
      delete iterationBest;
      noImprovement++;
      /* if (*iterationBest == *bestSchedule) */
      /*   std::cout << "same" << std::endl; */
      if (noImprovement > noImprovementMax)
        break;
    }
#if USE_ACS
    UpdatePheremone(bestSchedule);
#endif
    iterations++;
  }
  schedule_out->Copy(bestSchedule);
  delete bestSchedule;

  Logger::Info("ACO finished after %d iterations", iterations);
  return RES_SUCCESS;
}

void ACOScheduler::UpdatePheremone(InstSchedule *schedule) {
  // I wish InstSchedule allowed you to just iterate over it, but it's got this
  // cycle and slot thing which needs to be accounted for
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);

  SchedInstruction *lastInst = NULL;
  while (instNum != INVALID_VALUE) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(instNum);

    pheremone_t *pheremone = &Pheremone(lastInst, inst);
#if USE_ACS
    // ACS update rule includes decay
    // only the arcs on the current solution are decayed
    *pheremone = (1 - DECAY_FACTOR) * *pheremone + DECAY_FACTOR / (schedule->GetSpillCost() + 1);
#else
    *pheremone = *pheremone + 1/(schedule->GetSpillCost() + 1);
#endif
    lastInst = inst;

    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  schedule->ResetInstIter();

#if !USE_ACS
  // decay pheremone
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      Pheremone(i, j) *= (1 - DECAY_FACTOR);
    }
  }
#endif
  /* PrintPheremone(); */
}

// copied from Enumerator
inline void ACOScheduler::UpdtRdyLst_(InstCount cycleNum, int slotNum) {
  InstCount prevCycleNum = cycleNum - 1;
  LinkedList<SchedInstruction> *lst1 = NULL;
  LinkedList<SchedInstruction> *lst2 = frstRdyLstPerCycle_[cycleNum];

  if (slotNum == 0 && prevCycleNum >= 0) {
    // If at the begining of a new cycle other than the very first cycle, then
    // we also have to include the instructions that might have become ready in
    // the previous cycle due to a zero latency of the instruction scheduled in
    // the very last slot of that cycle [GOS 9.8.02].
    lst1 = frstRdyLstPerCycle_[prevCycleNum];

    if (lst1 != NULL) {
      rdyLst_->AddList(lst1);
      lst1->Reset();
      CleanupCycle_(prevCycleNum);
    }
  }

  if (lst2 != NULL) {
    rdyLst_->AddList(lst2);
    lst2->Reset();
  }
}

void ACOScheduler::PrintPheremone() {
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      std::cerr << std::fixed << std::setprecision(2) << Pheremone(i,j) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

/* Not Used
static void PrintInstruction(SchedInstruction *inst) {
  std::cerr << std::setw(2) << inst->GetNum() << " ";
  std::cerr << std::setw(20) << std::left << inst->GetOpCode();

  std::cerr << " defs ";
  Register **defs;
  uint16_t defsCount = inst->GetDefs(defs);
  for (uint16_t i = 0; i < defsCount; i++) {
    std::cerr << defs[i]->GetNum() << defs[i]->GetType();
    if (i != defsCount - 1)
      std::cerr << ", ";
  }

  std::cerr << " uses ";
  Register **uses;
  uint16_t usesCount = inst->GetUses(uses);
  for (uint16_t i = 0; i < usesCount; i++) {
    std::cerr << uses[i]->GetNum() << uses[i]->GetType();
    if (i != usesCount - 1)
      std::cerr << ", ";
  }
  std::cerr << std::endl;
}
*/

void PrintSchedule(InstSchedule *schedule) {
  std::cerr << schedule->GetSpillCost() << ": ";
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);
  while (instNum != INVALID_VALUE) {
    std::cerr << instNum << " ";
    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  std::cerr << std::endl;
  schedule->ResetInstIter();
}

}
