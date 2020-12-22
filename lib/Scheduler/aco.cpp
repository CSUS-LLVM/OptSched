#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/dev_defines.h"
#include "llvm/ADT/STLExtras.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace llvm::opt_sched;

#ifndef NDEBUG
static void PrintInstruction(SchedInstruction *inst);
#endif
void PrintSchedule(InstSchedule *schedule);

double RandDouble(double min, double max) {
  double rand = (double)RandomGen::GetRand32() / INT32_MAX;
  return (rand * (max - min)) + min;
}

#define USE_ACS 0
//#define BIASED_CHOICES 10000000
//#define LOCAL_DECAY 0.1

//#if USE_ACS
//#define ANTS_PER_ITERATION 10
//#define DECAY_FACTOR 0.1
//#else
//#define ANTS_PER_ITERATION count_
//#define DECAY_FACTOR 0.5
//#endif

ACOScheduler::ACOScheduler(DataDepGraph *dataDepGraph,
                           MachineModel *machineModel, InstCount upperBound,
                           SchedPriorities priorities, bool vrfySched,
			   SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
			   DeviceVector<Choice> **dev_ready, 
			   MachineModel *dev_MM, curandState_t *dev_states)
    : ConstrainedScheduler(dataDepGraph, machineModel, upperBound) {
  VrfySched_ = vrfySched;
  prirts_ = priorities;
  rdyLst_ = new ReadyList(dataDepGraph_, priorities);
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();
  dev_rgn_ = dev_rgn;
  dev_DDG_ = dev_DDG;
  dev_ready_ = dev_ready;
  dev_MM_ = dev_MM;
  dev_states_ = dev_states;
  dev_pheremone_elmnts_alloced_ = false;

  use_fixed_bias = schedIni.GetBool("ACO_USE_FIXED_BIAS");
  heuristicImportance_ = schedIni.GetInt("ACO_HEURISTIC_IMPORTANCE");
  use_tournament = schedIni.GetBool("ACO_TOURNAMENT");
  fixed_bias = schedIni.GetInt("ACO_FIXED_BIAS");
  bias_ratio = schedIni.GetFloat("ACO_BIAS_RATIO");
  local_decay = schedIni.GetFloat("ACO_LOCAL_DECAY");
  decay_factor = schedIni.GetFloat("ACO_DECAY_FACTOR");
  ants_per_iteration = schedIni.GetInt("ACO_ANT_PER_ITERATION");
  print_aco_trace = schedIni.GetBool("ACO_TRACE");
  noImprovementMax = schedIni.GetInt("ACO_STOP_ITERATIONS");

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
  pheremone_.resize(pheremone_size);
  InitialSchedule = nullptr;
}

ACOScheduler::~ACOScheduler() { delete rdyLst_; }

// Pheremone table lookup
// -1 means no instruction, so e.g. pheremone(-1, 10) gives pheremone on path
// from empty schedule to schedule only containing instruction 10
__host__ __device__
pheremone_t &ACOScheduler::Pheremone(SchedInstruction *from,
                                     SchedInstruction *to) {
  assert(to != NULL);
  int fromNum = -1;
  if (from != NULL)
    fromNum = from->GetNum();
  return Pheremone(fromNum, to->GetNum());
}

__host__ __device__
pheremone_t &ACOScheduler::Pheremone(InstCount from, InstCount to) {
  int row = 0;
  if (from != -1)
    row = from + 1;
  return pheremone_[(row * count_) + to];
}

__host__ __device__
double ACOScheduler::Score(SchedInstruction *from, Choice choice) {
  return Pheremone(from, choice.inst) *
         pow(choice.heuristic, heuristicImportance_);
}

__host__ __device__
SchedInstruction *
ACOScheduler::SelectInstruction(DeviceVector<Choice> &ready,
                                SchedInstruction *lastInst) {
#if USE_ACS
  double choose_best_chance;
  if (use_fixed_bias) {
    //choose_best_chance = fmax(0, 1 - (double)fixed_bias / count_);
    if (0 > 1 - (double)fixed_bias / count_)
      choose_best_chance = 0;
    else
      choose_best_chance = 1 - (double)fixed_bias / count_;
  } else
    choose_best_chance = bias_ratio;

  if (RandDouble(0, 1) < choose_best_chance) {
    if (print_aco_trace)
      printf("choose_best, use fixed bias: %d\n", use_fixed_bias);
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
  if (use_tournament) {
    int POPULATION_SIZE = ready.size();
#ifdef __CUDA_ARCH__
    int r_pos = (int)(curand_uniform(&dev_states_[GLOBALTID]) * POPULATION_SIZE);
    int s_pos = (int)(curand_uniform(&dev_states_[GLOBALTID]) * POPULATION_SIZE);
#else
    int r_pos = (int)(RandDouble(0, 1) * POPULATION_SIZE);
    int s_pos = (int)(RandDouble(0, 1) * POPULATION_SIZE);
    //    int t_pos = (int) (RandDouble(0, 1) *POPULATION_SIZE);
#endif
    Choice r = ready[r_pos];
    Choice s = ready[s_pos];
    //    Choice t = ready[t_pos];
    if (print_aco_trace) {
      printf("tournament Start \n");
      printf("array_size: %d\n", POPULATION_SIZE);
      printf("r:\t%d\n", r_pos);
      printf("s:\t%d\n", s_pos);
    }
    if (Score(lastInst, r) >=
        Score(lastInst, s))
      return r.inst;
    else
      return s.inst;
  }
  pheremone_t sum = 0;
  for (auto choice : ready)
    sum += Score(lastInst, choice);
#ifdef __CUDA_ARCH__
  pheremone_t point = sum * (pheremone_t)(curand_uniform(&dev_states_[GLOBALTID]));
#else
  pheremone_t point = RandDouble(0, sum);
#endif
  for (auto choice : ready) {
    point -= Score(lastInst, choice);
    if (point <= 0)
      return choice.inst;
  }
#ifdef __CUDA_ARCH__
  //atomicAdd(&returnLastInstCnt_, 1);
#else
  printf("returning last instruction\n");
#endif
  assert(point < 0.001); // floats should not be this inaccurate
  return ready.back().inst;
}

__host__ __device__
InstSchedule *ACOScheduler::FindOneSchedule(InstSchedule *dev_schedule, 
		                            DeviceVector<Choice> *dev_ready) {
#ifdef __CUDA_ARCH__ // device version of function
  SchedInstruction *lastInst = NULL;
  InstSchedule *schedule;
  if (!dev_schedule)
    schedule = new InstSchedule(machMdl_, dataDepGraph_, true);
  else
    schedule = dev_schedule;

  InstCount maxPriority = dev_rdyLst_[GLOBALTID]->MaxPriority();
  if (maxPriority == 0)
    maxPriority = 1; // divide by 0 is bad
  Initialize_();
  ((BBWithSpill *)dev_rgn_)->Dev_InitForSchdulng();
  DeviceVector<Choice> *ready;
  if (dev_ready)
    ready = dev_ready;
  else
    ready = new DeviceVector<Choice>(dataDepGraph_->GetInstCnt());
  while (!IsSchedComplete_()) {
    // convert the ready list from a custom priority queue to a std::vector,
    // much nicer for this particular scheduler
    UpdtRdyLst_(dev_crntCycleNum_[GLOBALTID], dev_crntSlotNum_[GLOBALTID]);
    unsigned long heuristic;
    ready->reserve(dev_rdyLst_[GLOBALTID]->GetInstCnt());
    SchedInstruction *inst = dev_rdyLst_[GLOBALTID]->GetNextPriorityInst(heuristic);
    while (inst != NULL) {
      if (ChkInstLglty_(inst)) {
        Choice c;
        c.inst = inst;
        c.heuristic = (double)heuristic / maxPriority;
        ready->push_back(c);
      }
      inst = dev_rdyLst_[GLOBALTID]->GetNextPriorityInst(heuristic);
    }
    dev_rdyLst_[GLOBALTID]->ResetIterator();
    inst = NULL;
    if (!ready->empty())
      inst = SelectInstruction(*ready, lastInst);
/*
      if (returnLastInstCnt_ > 0) {
	if (GLOBALTID == 0) {
	  printf("%d threads returned last instruction\n", returnLastInstCnt_);
          returnLastInstCnt_ = 0;
	}
      }
*/
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
      SchdulInst_(inst, dev_crntCycleNum_[GLOBALTID]);
      inst->Schedule(dev_crntCycleNum_[GLOBALTID],
                     dev_crntSlotNum_[GLOBALTID]);
      ((BBWithSpill *)dev_rgn_)->Dev_SchdulInst(inst,
                                            dev_crntCycleNum_[GLOBALTID],
                                            dev_crntSlotNum_[GLOBALTID],
                                            false);
      DoRsrvSlots_(inst);
      // this is annoying
      SchedInstruction *blah = dev_rdyLst_[GLOBALTID]->GetNextPriorityInst();
      while (blah != NULL && blah != inst) {
        blah = dev_rdyLst_[GLOBALTID]->GetNextPriorityInst();
      }
      if (blah == inst)
        dev_rdyLst_[GLOBALTID]->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);
    }
    /* Logger::Info("Chose instruction %d (for some reason)", instNum); */
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
    dev_rdyLst_[GLOBALTID]->ResetIterator();
    ready->clear();
    // Juggle frstRdyLists on device, prev falls out of scope, frstRdyList[0]
    // becomes prev, 1 becomes 0 etc
    int maxLatency = dataDepGraph_->GetMaxLtncy();
    ArrayList<InstCount> * temp;
    dev_prevFrstRdyLstPerCycle_[GLOBALTID]->Reset();
    temp = dev_prevFrstRdyLstPerCycle_[GLOBALTID];
    dev_prevFrstRdyLstPerCycle_[GLOBALTID] = 
	            dev_frstRdyLstPerCycle_[GLOBALTID][0];
    for (int i = 0; i < maxLatency; i++) {
      dev_frstRdyLstPerCycle_[GLOBALTID][i] = 
	      dev_frstRdyLstPerCycle_[GLOBALTID][i + 1];
    }
    dev_frstRdyLstPerCycle_[GLOBALTID][maxLatency] = temp;
  }
  dev_rgn_->UpdateScheduleCost(schedule);
  return schedule;

#else  // Host version of function
  SchedInstruction *lastInst = NULL;
  InstSchedule *schedule;
  if (!dev_schedule)
    schedule = new InstSchedule(machMdl_, dataDepGraph_, true);
  else
    schedule = dev_schedule;

  InstCount maxPriority = rdyLst_->MaxPriority();
  if (maxPriority == 0)
    maxPriority = 1; // divide by 0 is bad
  Initialize_();
  rgn_->InitForSchdulng();
  DeviceVector<Choice> *ready;
  if (dev_ready)
    ready = dev_ready;
  else
    ready = new DeviceVector<Choice>(dataDepGraph_->GetInstCnt());
  while (!IsSchedComplete_()) {
    // convert the ready list from a custom priority queue to a std::vector,
    // much nicer for this particular scheduler
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);
    unsigned long heuristic;
    ready->reserve(rdyLst_->GetInstCnt());
    SchedInstruction *inst = rdyLst_->GetNextPriorityInst(heuristic);
    while (inst != NULL) {
      if (ChkInstLglty_(inst)) {
        Choice c;
        c.inst = inst;
        c.heuristic = (double)heuristic / maxPriority;
        ready->push_back(c);
      }
      inst = rdyLst_->GetNextPriorityInst(heuristic);
    }
    rdyLst_->ResetIterator();
    inst = NULL;
    if (!ready->empty())
      inst = SelectInstruction(*ready, lastInst);
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
    ready->clear();
  }
  rgn_->UpdateScheduleCost(schedule);
  return schedule;
#endif
}

__global__
void Dev_FindOneSchedule(SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
            ACOScheduler *dev_AcoSchdulr, InstSchedule **dev_schedules,
	    DeviceVector<Choice> **dev_ready) {
  if (GLOBALTID == 0) {
    dev_rgn->SetDepGraph(dev_DDG);
    ((BBWithSpill *)dev_rgn)->SetRegFiles(dev_DDG->getRegFiles());
  }

  dev_AcoSchdulr->FindOneSchedule(dev_schedules[GLOBALTID], 
		                  dev_ready[GLOBALTID]);
}

FUNC_RESULT ACOScheduler::FindSchedule(InstSchedule *schedule_out,
                                       SchedRegion *region,
				       ACOScheduler *dev_AcoSchdulr) {
  rgn_ = region;

  // initialize pheremone
  // for this, we need the cost of the pure heuristic schedule
  int pheremone_size = (count_ + 1) * count_;
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = 1;
  initialValue_ = 1;
  InstSchedule *heuristicSched = FindOneSchedule();
  InstCount heuristicCost =
      heuristicSched->GetCost() + 1; // prevent divide by zero

#if USE_ACS
  initialValue_ = 2.0 / ((double)count_ * heuristicCost);
#else
  initialValue_ = (double)ants_per_iteration / heuristicCost;
#endif
  for (int i = 0; i < pheremone_size; i++)
    pheremone_[i] = initialValue_;
#if __CUDA_ARCH__
  printf("initialValue_: %d\n", initialValue_);
#else
  std::cerr << "initialValue_" << initialValue_ << std::endl;
#endif
  InstSchedule *bestSchedule = InitialSchedule;
  if (bestSchedule) {
    UpdatePheremone(bestSchedule);
  }
  // moved to constructor
  //Config &schedIni = SchedulerOptions::getInstance();
  //int noImprovementMax = schedIni.GetInt("ACO_STOP_ITERATIONS");
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  while (true) {
    InstSchedule *iterationBest = nullptr;
    
    //for (int i = 0; i < ants_per_iteration; i++) {
      
      // **** Start Dev_ACO code **** //
      size_t memSize;
      Logger::Info("Updating Pheremones"); 
      CopyPheremonesToDevice(dev_AcoSchdulr);
      // Copy Schedule to device
      Logger::Info("Creating and copying schedules to device");
      InstSchedule **dev_schedules;
      InstSchedule **host_schedules = new InstSchedule *[NUMTHREADS];
      InstSchedule **schedules = new InstSchedule *[NUMTHREADS];
      
      for (int i = 0; i < NUMTHREADS; i++) {
	schedules[i] = new InstSchedule(machMdl_, dataDepGraph_, true);
        schedules[i]->CopyPointersToDevice(dev_MM_);
	memSize = sizeof(InstSchedule);
	gpuErrchk(cudaMalloc(&host_schedules[i], memSize));
	gpuErrchk(cudaMemcpy(host_schedules[i], schedules[i], memSize,
			     cudaMemcpyHostToDevice));
      }
      memSize = sizeof(InstSchedule *) * NUMTHREADS;
      gpuErrchk(cudaMalloc((void**)&dev_schedules, memSize));
      // Copy schedule to device
      gpuErrchk(cudaMemcpy(dev_schedules, host_schedules, memSize,
                           cudaMemcpyHostToDevice));
      Logger::Info("Launching Dev_FindOneSchedule()");
      Dev_FindOneSchedule<<<NUMBLOCKS,NUMTHREADSPERBLOCK>>>(dev_rgn_, dev_DDG_,
		                     dev_AcoSchdulr, dev_schedules, dev_ready_);
      cudaDeviceSynchronize();
      Logger::Info("Post Kernel Error: %s", cudaGetErrorString(cudaGetLastError()));
      // Copy schedule to Host
      memSize = sizeof(InstSchedule *) * NUMTHREADS;
      gpuErrchk(cudaMemcpy(host_schedules, dev_schedules, memSize, 
			   cudaMemcpyDeviceToHost));
      memSize = sizeof(InstSchedule);
      for (int i = 0; i < NUMTHREADS; i++) {
	gpuErrchk(cudaMemcpy(schedules[i], host_schedules[i], memSize,
				      cudaMemcpyDeviceToHost));
	schedules[i]->CopyPointersToHost(machMdl_);
	cudaFree(host_schedules[i]);
      }
      cudaFree(dev_schedules);
      int bestSched = 0;
      for (int i = 1; i < NUMTHREADS; i++) {
	//if (schedules[i]->GetCost() != -1)
	  //PrintSchedule(schedules[i]);
          //Logger::Info("Schedule[%d] has cost %d", i, schedules[i]->GetCost());
        if (schedules[bestSched]->GetCost() > schedules[i]->GetCost() && schedules[i]->GetCost() != -1)
	  bestSched = i;
      }
      InstSchedule *schedule = schedules[bestSched];
      for (int i = 0; i < NUMTHREADS; i++)
        if (i != bestSched)
          delete schedules[i];
      delete schedules;
      // *** End Dev_ACO code **** //

      //InstSchedule *schedule = FindOneSchedule();
      if (print_aco_trace)
        PrintSchedule(schedule);
      if (iterationBest == nullptr ||
          schedule->GetCost() < iterationBest->GetCost()) {
	if (iterationBest && iterationBest != InitialSchedule)
	  delete iterationBest;
        iterationBest = schedule;
      } else {
        delete schedule;
      }
    //}
#if !USE_ACS
    UpdatePheremone(iterationBest);
#endif
    //PrintSchedule(iterationBest); 
    //std::cout << iterationBest->GetCost() << std::endl;
    // TODO DRY
    if (bestSchedule == nullptr ||
        iterationBest->GetCost() < bestSchedule->GetCost()) {
      if (bestSchedule && bestSchedule != InitialSchedule)
        delete bestSchedule;
      bestSchedule = std::move(iterationBest);
      printf("ACO found schedule with spill cost %d\n",
             bestSchedule->GetCost());
      printf("ACO found schedule "
             "cost:%d, rp cost:%d, sched length: %d, and "
             "iteration:%d\n",
             bestSchedule->GetCost(), bestSchedule->GetSpillCost(),
             bestSchedule->GetCrntLngth(), iterations);

      noImprovement = 0;
    } else {
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
  //if (bestSchedule != InitialSchedule)
    //delete bestSchedule;

  printf("ACO finished after %d iterations\n", iterations);
  return RES_SUCCESS;
}

__host__ __device__
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
    *pheremone = (1 - decay_factor) * *pheremone +
                 decay_factor / (schedule->GetCost() + 1);
#else
    *pheremone = *pheremone + 1 / (schedule->GetCost() + 1);
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
#ifdef __CUDA_ARCH__ // Device version
  InstCount prevCycleNum = cycleNum - 1;
  ArrayList<InstCount> *lst1 = NULL;
  ArrayList<InstCount> *lst2 = dev_frstRdyLstPerCycle_[GLOBALTID][0];

  if (slotNum == 0 && prevCycleNum >= 0) {
    // If at the begining of a new cycle other than the very first cycle, then
    // we also have to include the instructions that might have become ready in
    // the previous cycle due to a zero latency of the instruction scheduled in
    // the very last slot of that cycle [GOS 9.8.02].
    lst1 = dev_prevFrstRdyLstPerCycle_[GLOBALTID];

    if (lst1->size_ > 0) {
      dev_rdyLst_[GLOBALTID]->AddList(lst1);
      lst1->Reset();
      //CleanupCycle_(prevCycleNum);
    }
  }

  if (lst2->size_ > 0) {
    dev_rdyLst_[GLOBALTID]->AddList(lst2);
    lst2->Reset();
  }
#else  // Host version
  InstCount prevCycleNum = cycleNum - 1;
  ArrayList<InstCount> *lst1 = NULL;
  ArrayList<InstCount> *lst2 = frstRdyLstPerCycle_[cycleNum];

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
#endif
}

__host__ __device__
void ACOScheduler::PrintPheremone() {
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      //std::cerr << std::scientific << std::setprecision(8) << Pheremone(i, j)
      //          << " ";
      printf("%.10e ", Pheremone(i, j));
    }
    //std::cerr << std::endl;
    printf("\n");
  }
  //std::cerr << std::endl;
  printf("\n");
}

#ifndef NDEBUG
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
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
#endif

__host__ __device__
void PrintSchedule(InstSchedule *schedule) {
  printf("%d: ", schedule->GetCost());
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);
  while (instNum != INVALID_VALUE) {
    printf("%d ", instNum);
    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  printf("\n");
  schedule->ResetInstIter();
}

void ACOScheduler::setInitialSched(InstSchedule *Sched) {
  if (Sched) {
    InitialSchedule =
        new InstSchedule(machMdl_, dataDepGraph_, VrfySched_);
    InitialSchedule->Copy(Sched);
  }
}

void ACOScheduler::AllocDevArraysForParallelACO() {
  size_t memSize;
  // Alloc dev array for schduldInstCnt_
  memSize = sizeof(InstCount) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_schduldInstCnt_, memSize));
  // Alloc dev array for crntCycleNum_;
  memSize = sizeof(InstCount) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_crntCycleNum_, memSize));
  // Alloc dev array for crntSlotNum_;
  memSize = sizeof(InstCount) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_crntSlotNum_, memSize));
  // Allo dev array for crntRealSlotNum_
  memSize = sizeof(InstCount) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_crntRealSlotNum_, memSize));
  // Alloc dev array for isCrntCycleBlkd_;
  memSize = sizeof(bool) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_isCrntCycleBlkd_, memSize));
  // Alloc dev array for rdyLst_
  memSize = sizeof(ReadyList *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_rdyLst_, memSize));
  // Alloc dev array for avlblSlotsInCrntCycle_
  memSize = sizeof(int16_t *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_avlblSlotsInCrntCycle_, memSize));
  // Alloc dev arrays of avlblSlotsInCrntCycle_ for each thread
  memSize = sizeof(int16_t) * issuTypeCnt_;
  for (int i = 0; i < NUMTHREADS; i++) {
    gpuErrchk(cudaMalloc(&dev_avlblSlotsInCrntCycle_[i], memSize));
  }
  // Alloc dev arrays for frstRdyLstPerCycle_
  memSize = sizeof(ArrayList<InstCount> **) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_frstRdyLstPerCycle_, memSize));
  memSize = sizeof(ArrayList<InstCount> *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_prevFrstRdyLstPerCycle_, memSize));
  memSize = sizeof(ArrayList<InstCount> *) * (dataDepGraph_->GetMaxLtncy() + 1);
  for (int i = 0; i < NUMTHREADS; i++) {
    gpuErrchk(cudaMallocManaged(&dev_frstRdyLstPerCycle_[i], memSize));
  }
  // Alloc dev arrays for rsrvSlots_
  memSize = sizeof(ReserveSlot *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_rsrvSlots_, memSize));
  memSize = sizeof(ReserveSlot) * issuRate_;
  for (int i = 0; i < NUMTHREADS; i++) {
    gpuErrchk(cudaMalloc(&dev_rsrvSlots_[i], memSize));
  }
  memSize = sizeof(int16_t) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_rsrvSlotCnt_, memSize));
}

void ACOScheduler::CopyPheremonesToDevice(ACOScheduler *dev_AcoSchdulr) {
  size_t memSize;
  // Free allocated mem sinve pheremone size can change
  if (dev_AcoSchdulr->dev_pheremone_elmnts_alloced_ == true)
    cudaFree(dev_AcoSchdulr->pheremone_.elmnts_);

  memSize = sizeof(DeviceVector<pheremone_t>);
  gpuErrchk(cudaMemcpy(&dev_AcoSchdulr->pheremone_, &pheremone_, memSize,
            cudaMemcpyHostToDevice));

  memSize = sizeof(pheremone_t) * pheremone_.alloc_;
  gpuErrchk(cudaMalloc(&(dev_AcoSchdulr->pheremone_.elmnts_), memSize));
  gpuErrchk(cudaMemcpy(dev_AcoSchdulr->pheremone_.elmnts_, pheremone_.elmnts_,
		       memSize, cudaMemcpyHostToDevice));
  
  dev_AcoSchdulr->dev_pheremone_elmnts_alloced_ = true;
}

void ACOScheduler::CopyPointersToDevice(ACOScheduler *dev_ACOSchedulr) {
  //dev_ACOSchedulr->InitialSchedule = dev_InitSched;
  size_t memSize;
  dev_ACOSchedulr->machMdl_ = dev_MM_;
  dev_ACOSchedulr->dataDepGraph_ = dev_DDG_;
  // Copy slotsPerTypePerCycle_
  int *dev_slotsPerTypePerCycle;
  memSize = sizeof(int) * issuTypeCnt_;
  gpuErrchk(cudaMalloc(&dev_slotsPerTypePerCycle, memSize));
  gpuErrchk(cudaMemcpy(dev_slotsPerTypePerCycle, slotsPerTypePerCycle_,
		       memSize, cudaMemcpyHostToDevice));
  dev_ACOSchedulr->slotsPerTypePerCycle_ = dev_slotsPerTypePerCycle;
  // Copy instCntPerIssuType_
  InstCount *dev_instCntPerIssuType;
  memSize = sizeof(InstCount) * issuTypeCnt_;
  gpuErrchk(cudaMalloc(&dev_instCntPerIssuType, memSize));
  gpuErrchk(cudaMemcpy(dev_instCntPerIssuType, instCntPerIssuType_, memSize,
		       cudaMemcpyHostToDevice));
  dev_ACOSchedulr->instCntPerIssuType_ = dev_instCntPerIssuType;
  // set root/leaf inst
  dev_ACOSchedulr->rootInst_ = dev_DDG_->GetRootInst();
  dev_ACOSchedulr->leafInst_ = dev_DDG_->GetLeafInst();
  // Copy frstRdyLstPerCycle_[0], an ArrayList, to dev_(prev/crnt)FrstRdyLst_
  Logger::Info("Copying ACO frstRdyLstPerCycle_ Array lists");
  for (int i = 0; i < NUMTHREADS; i++) {
    // Copy each ArrayList to device
    memSize = sizeof(ArrayList<InstCount>);
    gpuErrchk(cudaMallocManaged(&dev_prevFrstRdyLstPerCycle_[i], memSize));
    gpuErrchk(cudaMemcpy(dev_prevFrstRdyLstPerCycle_[i], frstRdyLstPerCycle_[0],
			 memSize, cudaMemcpyHostToDevice));
    memSize = sizeof(InstCount) * dataDepGraph_->GetInstCnt();
    gpuErrchk(cudaMalloc(&dev_prevFrstRdyLstPerCycle_[i]->elmnts_, memSize));
    for (int j = 0; j < dataDepGraph_->GetMaxLtncy() + 1; j++) {
      memSize = sizeof(ArrayList<InstCount>);
      gpuErrchk(cudaMallocManaged(&dev_frstRdyLstPerCycle_[i][j], memSize));
      gpuErrchk(cudaMemcpy(dev_frstRdyLstPerCycle_[i][j], 
			   frstRdyLstPerCycle_[0], memSize,
                           cudaMemcpyHostToDevice));
      memSize = sizeof(InstCount) * dataDepGraph_->GetInstCnt();
      gpuErrchk(cudaMalloc(&dev_frstRdyLstPerCycle_[i][j]->elmnts_, memSize));
    }
  }
  // Copy rdyLst_
  Logger::Info("Copying ACO rdyLsts_");
  ReadyList *dev_rdyLst;
  memSize = sizeof(ReadyList);
  for (int i = 0; i < NUMTHREADS; i++) {
    gpuErrchk(cudaMallocManaged(&dev_rdyLst, memSize));
    gpuErrchk(cudaMemcpy(dev_rdyLst, rdyLst_, memSize, cudaMemcpyHostToDevice));
    rdyLst_->CopyPointersToDevice(dev_rdyLst, dev_DDG_);
    dev_ACOSchedulr->dev_rdyLst_[i] = dev_rdyLst;
  }
}

void ACOScheduler::FreeDevicePointers() {
  cudaFree(dev_schduldInstCnt_);
  cudaFree(dev_crntCycleNum_);
  cudaFree(dev_crntSlotNum_);
  cudaFree(dev_crntRealSlotNum_);
  cudaFree(dev_isCrntCycleBlkd_);
  cudaFree(slotsPerTypePerCycle_);
  cudaFree(instCntPerIssuType_);
  for (int i = 0; i < NUMTHREADS; i++){
    cudaFree(dev_prevFrstRdyLstPerCycle_[i]->elmnts_);
    cudaFree(dev_prevFrstRdyLstPerCycle_[i]);
    for (int j = 0; j < dataDepGraph_->GetMaxLtncy() + 1; j++) {
      cudaFree(dev_frstRdyLstPerCycle_[i][j]->elmnts_);
      cudaFree(dev_frstRdyLstPerCycle_[i][j]);
    }
    cudaFree(dev_frstRdyLstPerCycle_[i]);
    dev_rdyLst_[i]->FreeDevicePointers();
    cudaFree(dev_rdyLst_[i]);
    cudaFree(dev_avlblSlotsInCrntCycle_[i]);
    cudaFree(dev_rsrvSlots_[i]);
  }
  cudaFree(dev_avlblSlotsInCrntCycle_);
  cudaFree(dev_rsrvSlots_);
  cudaFree(dev_rsrvSlotCnt_);
  cudaFree(dev_prevFrstRdyLstPerCycle_);
  cudaFree(dev_frstRdyLstPerCycle_);
  cudaFree(dev_rdyLst_);
  cudaFree(pheremone_.elmnts_);
}
