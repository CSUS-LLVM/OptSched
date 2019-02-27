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

using namespace llvm::opt_sched;

static void PrintInstruction(SchedInstruction *inst);
void PrintSchedule(InstSchedule *schedule);

double RandDouble(double min, double max) {
  double rand = (double) RandomGen::GetRand32() / INT32_MAX;
  return (rand * (max - min)) + min;
}

#define USE_ACS 1
//#define BIASED_CHOICES 10000000
//#define LOCAL_DECAY 0.1

//#if USE_ACS
//#define ANTS_PER_ITERATION 10
//#define DECAY_FACTOR 0.1
//#else
//#define ANTS_PER_ITERATION count_
//#define DECAY_FACTOR 0.5
//#endif

ACOScheduler::ACOScheduler(DataDepGraph *dataDepGraph, MachineModel *machineModel, InstCount upperBound, SchedPriorities priorities) : ConstrainedScheduler(dataDepGraph, machineModel, upperBound) {
  prirts_ = priorities;
  rdyLst_ = new ReadyList(dataDepGraph_, priorities);
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();
  
  use_fixed_bias = schedIni.GetBool("ACO_USE_FIXED_BIAS");
  heuristicImportance_ = schedIni.GetInt("ACO_HEURISTIC_IMPORTANCE");
  use_tournament = schedIni.GetBool("ACO_TOURNAMENT");
  fixed_bias = schedIni.GetInt("ACO_FIXED_BIAS");
  bias_ratio = schedIni.GetFloat("ACO_BIAS_RATIO");
  local_decay = schedIni.GetFloat("ACO_LOCAL_DECAY");
  decay_factor = schedIni.GetFloat("ACO_DECAY_FACTOR");
  ants_per_iteration = schedIni.GetInt("ACO_ANT_PER_ITERATION");
  print_aco_trace = schedIni.GetBool("ACO_TRACE");
  
  /*
  std::cerr << "useOldAlg===="<<useOldAlg<<"\n\n";
  std::cerr << "heuristicImportance_===="<<heuristicImportance_<<"\n\n";
  std::cerr << "tournament===="<<tournament<<"\n\n";
  std::cerr << "bias_ratio===="<<bias_ratio<<"\n\n";
  std::cerr << "local_decay===="<<local_decay<<"\n\n";
  std::cerr << "decay_factor===="<<decay_factor<<"\n\n";
  std::cerr << "ants_per_iteration===="<<ants_per_iteration<<"\n\n";
  */
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
    double choose_best_chance;    
    if (use_fixed_bias)
        choose_best_chance = fmax(0, 1 - (double)fixed_bias / count_);
    else
        choose_best_chance = bias_ratio;
     
    if (RandDouble(0, 1) < choose_best_chance) {
        if (print_aco_trace)
            std::cerr<<"choose_best, use fixed bais: "<<use_fixed_bias<<"\n";
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
    if (use_tournament){
        int POPULATION_SIZE = ready.size();
        int r_pos = (int) (RandDouble(0, 1) *POPULATION_SIZE);
        int s_pos = (int) (RandDouble(0, 1) *POPULATION_SIZE);
    //    int t_pos = (int) (RandDouble(0, 1) *POPULATION_SIZE);
        Choice r = ready[r_pos];
        Choice s = ready[s_pos];
    //    Choice t = ready[t_pos];
        if (print_aco_trace) {
            std::cerr << "tournament Start \n";
            std::cerr << "array_size:"<<POPULATION_SIZE<<"\n";
            std::cerr<<"r:\t"<<r_pos<<"\n";
            std::cerr<<"s:\t"<<s_pos<<"\n";
    //        std::cerr<<"t:\t"<<t_pos<<"\n";

            std::cerr<<"Score r"<<Score(lastInst, r)<<"\n";
            std::cerr<<"Score s"<<Score(lastInst, s)<<"\n";
   //         std::cerr<<"Score t"<<Score(lastInst, t)<<"\n";
        }
        if (Score(lastInst, r) >= Score(lastInst, s)) //&& Score(lastInst, r) >= Score(lastInst, t))
            return r.inst;
   //     else if (Score(lastInst, s) >= Score(lastInst, r) && Score(lastInst, s) >= Score(lastInst, t))
   //         return s.inst;
        else
            return s.inst;
    }      
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
    /*
     std::stringstream stream; 
     stream << "Ready list: "; 
    for (auto choice : ready) { 
      stream << choice.inst->GetNum() << ", "; 
    } 
    Logger::Info(stream.str().c_str()); 
    */

    inst = NULL;
    if (!ready.empty())
      inst = SelectInstruction(ready, lastInst);
	if (inst != NULL) {
#ifdef USE_ACS
 		// local pheremone decay
		pheremone_t *pheremone = &Pheremone(lastInst, inst);
		*pheremone = (1 - local_decay) * *pheremone + local_decay * initialValue_;
#endif
		lastInst = inst;
	}

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
  InstCount heuristicCost = FindOneSchedule()->GetCost() + 1; // prevent divide by zero

#if USE_ACS
  initialValue_ = 2.0 / ((double) count_ * heuristicCost);
#else
  initialValue_ = (double) ants_per_iteration / heuristicCost;
#endif
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = initialValue_;
  std::cerr<<"initialValue_"<<initialValue_<<std::endl;

  InstSchedule *bestSchedule = NULL;
  Config &schedIni = SchedulerOptions::getInstance();
  int noImprovementMax = schedIni.GetInt("ACO_STOP_ITERATIONS");
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  while (true) {
    InstSchedule *iterationBest = NULL;
    for (int i = 0; i < ants_per_iteration; i++) {
      InstSchedule *schedule = FindOneSchedule();
      if(print_aco_trace)
         PrintSchedule(schedule); 
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
    /* std::cout << iterationBest->GetCost() << std::endl; */
    // TODO DRY
    if (bestSchedule == NULL || iterationBest->GetCost() < bestSchedule->GetCost()) {
      delete bestSchedule;
      bestSchedule = iterationBest;
      Logger::Info("ACO found schedule with spill cost %d", bestSchedule->GetCost());
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
  PrintSchedule(bestSchedule);
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
    *pheremone = (1 - decay_factor) * *pheremone + decay_factor / (schedule->GetCost() + 1);
#else
    *pheremone = *pheremone + 1/(schedule->GetCost() + 1);
#endif
    lastInst = inst;

    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  schedule->ResetInstIter();

#if !USE_ACS
  // decay pheremone
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      Pheremone(i, j) *= (1 - decay_factor);
    }
  }
#endif
    if (print_aco_trace)
        PrintPheremone(); 
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
      std::cerr << std::scientific << std::setprecision(8) << Pheremone(i,j) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

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

void PrintSchedule(InstSchedule *schedule) {
  std::cerr << schedule->GetCost() << ": ";
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);
  while (instNum != INVALID_VALUE) {
    std::cerr << instNum << " ";
    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  std::cerr << std::endl;
  schedule->ResetInstIter();
}