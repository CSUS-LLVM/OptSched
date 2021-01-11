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

  InstCount maxPriority = dev_rdyLst_->MaxPriority();
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
    ready->reserve(dev_rdyLst_->GetInstCnt());
    SchedInstruction *inst = dev_rdyLst_->GetNextPriorityInst(heuristic);
    while (inst != NULL) {
      if (ChkInstLglty_(inst)) {
        Choice c;
        c.inst = inst;
        c.heuristic = (double)heuristic / maxPriority;
        ready->push_back(c);
      }
      inst = dev_rdyLst_->GetNextPriorityInst(heuristic);
    }
    dev_rdyLst_->ResetIterator();
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
      SchedInstruction *blah = dev_rdyLst_->GetNextPriorityInst();
      while (blah != NULL && blah != inst) {
        blah = dev_rdyLst_->GetNextPriorityInst();
      }
      if (blah == inst)
        dev_rdyLst_->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);
    }
    //debug
    //printf("Thread %d chose instruction %d (for some reason)\n", GLOBALTID, instNum);
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
    dev_rdyLst_->ResetIterator();
    ready->clear();
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
	    DeviceVector<Choice>**dev_ready, InstSchedule *dev_iterationBest) {
  if (GLOBALTID == 0) {
    dev_rgn->SetDepGraph(dev_DDG);
    ((BBWithSpill *)dev_rgn)->SetRegFiles(dev_DDG->getRegFiles());
  }
  dev_schedules[GLOBALTID]->Initialize();
  dev_AcoSchdulr->FindOneSchedule(dev_schedules[GLOBALTID],
		                  dev_ready[GLOBALTID]);
/*
  //debug
  for (int i = 0; i < NUMTHREADS; i++)
    if (GLOBALTID == 0) {
      printf("Printing schedule for thread %d\n", i);
      dev_schedules[GLOBALTID]->Print();
    }
*/
/*
  if (GLOBALTID == 0)
    dev_iterationBest->Copy(dev_schedules[GLOBALTID]);
*/
}

__global__ 
void Dev_SelectBestSched(InstSchedule **dev_schedules,
                         InstSchedule *dev_iterationBest,
                         bool *dev_foundBestSched) {
  // Each thread counts how many schedules are worse or equal to it,
  // those that are better than or equal to NUMTHREADS schedules 
  // set their index in foundBestSched to true. 1 tread iterates through
  // the foundBestSched array and copies the schedule that correlates to the first
  // true it finds
  // reset foundBestSched array
  //dev_foundBestSched[GLOBALTID] = false;
  int cost = dev_schedules[GLOBALTID]->GetCost();
  int compCnt = 0;
  for (int i = 0; i < NUMTHREADS; i++) {
    compCnt+= (dev_schedules[i]->GetCost() >= cost);
  }
/*
  //debug
  printf("Thread %d has compCnt = %d\n", GLOBALTID, compCnt);
  for (int i = 0; i < NUMTHREADS; i++)
    if (GLOBALTID == 0) {
      printf("Printing schedule for thread %d\n", i);
      dev_schedules[GLOBALTID]->Print();
    }
*/
  // Set if thread found an iterationBestSched
  dev_foundBestSched[GLOBALTID] = (compCnt == NUMTHREADS);
  // one thread selects and copied a schedule
  if (GLOBALTID == 0) {
    int i;
    for (i = 0; i < NUMTHREADS; i++) {
      if (dev_foundBestSched[i] == true) {
        //debug
        printf("Thread %d's schedule will be copied back to host\n", i);
        dev_iterationBest->Initialize(); 
        dev_iterationBest->Copy(dev_schedules[i]);
        break;
      }   
    }   
  }
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
  std::cerr << "initialValue_" << initialValue_ << std::endl;
  InstSchedule *bestSchedule = InitialSchedule;
  if (bestSchedule) {
    UpdatePheremone(bestSchedule);
  }

  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  // dev array of dev pointers to schedules
  InstSchedule **dev_schedules;
  // host array of dev pointers to schedules
  InstSchedule **host_schedules = new InstSchedule *[NUMTHREADS];
  // holds schedules to be copied over
  InstSchedule **schedules = new InstSchedule *[NUMTHREADS];
/* Implementation with one array of schedules, has unsolved memory errors
  InstSchedule *dev_schedules;
  InstSchedule *schedules;
*/
  InstSchedule *iterationBest;
  // Holds best schedule for iteration on the device to be copied back to host
  InstSchedule *dev_iterationBest;
  // holds which threads found the best schedule
  bool *dev_foundBestSched;
  while (true) {
    if (DEV_ACO) { // Run ACO on device
      size_t memSize;
      Logger::Info("Updating Pheremones"); 
      CopyPheremonesToDevice(dev_AcoSchdulr);
      // Copy Schedule to device
      if (iterations == 0) {
        Logger::Info("Creating and copying schedules to device");
        //iterationBest = new InstSchedule(machMdl_, dataDepGraph_, true);
        bestSchedule = new InstSchedule(machMdl_, dataDepGraph_, true);
        bestSchedule->Copy(InitialSchedule);
        //printf("InitialSchedule: ");
        //PrintSchedule(bestSchedule);
        for (int i = 0; i < NUMTHREADS; i++) {
          schedules[i] = new InstSchedule(machMdl_, dataDepGraph_, true);
          schedules[i]->AllocateOnDevice(dev_MM_);
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
        // Allocate dev mem for and copy over dev_iterationBest so it is the
        // only schedule that needs to be copied back to host
        iterationBest = new InstSchedule(machMdl_, dataDepGraph_, true);
        iterationBest->AllocateOnDevice(dev_MM_);
        memSize = sizeof(InstSchedule);
        gpuErrchk(cudaMalloc((void**)&dev_iterationBest, memSize));
        gpuErrchk(cudaMemcpy(dev_iterationBest, iterationBest, memSize,
                             cudaMemcpyHostToDevice));
        // allocate an array of bools for dev_foundBestSched
        memSize = sizeof(bool) * NUMTHREADS;
        gpuErrchk(cudaMalloc(&dev_foundBestSched, memSize));
/* Implementation with one array of schedules, has unsolved memory errors
        // Create and array of schedules with dummy constructor
        schedules = new InstSchedule[NUMTHREADS];
        for (int i = 0; i < NUMTHREADS; i++) {
          // Run constructor for each schedule
          schedules[i] = InstSchedule(machMdl_, dataDepGraph_, true);
          // allocate dev arrays for each schedule
          schedules[i].AllocateOnDevice(dev_MM_);
        }
        memSize = sizeof(InstSchedule) * NUMTHREADS;
        // Alloc dev mem for schedules array
        gpuErrchk(cudaMalloc((void**)&dev_schedules, memSize));
        // Copy schedules to device
        gpuErrchk(cudaMemcpy(dev_schedules, schedules, memSize,
                             cudaMemcpyHostToDevice));
*/
      }
      Logger::Info("Launching Dev_FindOneSchedule()");
      Dev_FindOneSchedule<<<NUMBLOCKS,NUMTHREADSPERBLOCK>>>(dev_rgn_, dev_DDG_,
		                     dev_AcoSchdulr, dev_schedules, dev_ready_,
                                     dev_iterationBest);
      cudaDeviceSynchronize();
      Logger::Info("Post Kernel Error: %s", cudaGetErrorString(cudaGetLastError()));
      Logger::Info("Launching Dev_SelectBestSched");
      Dev_SelectBestSched<<<NUMBLOCKS,NUMTHREADSPERBLOCK>>>(dev_schedules, 
                                      dev_iterationBest, dev_foundBestSched);
      cudaDeviceSynchronize();
      Logger::Info("Post Kernel Error: %s", cudaGetErrorString(cudaGetLastError()));
      // Copy dev_iterationBest back to host
      memSize = sizeof(InstSchedule);
      gpuErrchk(cudaMemcpy(iterationBest, dev_iterationBest, memSize,
                           cudaMemcpyDeviceToHost));
      iterationBest->CopyArraysToHost();
/* Implementation with one array of schedules, has unsolved memory errors
      // Copy schedule to Host
      memSize = sizeof(InstSchedule) * NUMTHREADS;
      gpuErrchk(cudaMemcpy(schedules, dev_schedules, memSize, 
			   cudaMemcpyDeviceToHost));
      int bestSched = 0;
      for (int i = 0; i < NUMTHREADS; i++) {
	//PrintSchedule(&schedules[i]);
        // Copy dev array contents to host
        schedules[i].CopyArraysToHost();
        //debug
        printf("Printing schedule for thread %d\n", i);
        PrintSchedule(&schedules[i]);
        if (schedules[i].GetCost() == -1)
          Logger::Fatal("Error with scheduling on thread num %d", i);
        if (schedules[bestSched].GetCost() > schedules[i].GetCost())
	  bestSched = i;
      }
  
      InstSchedule *schedule = &schedules[bestSched];
      if (print_aco_trace)
        PrintSchedule(schedule);
      if (iterationBest->GetCost() == -1 ||
          schedule->GetCost() < iterationBest->GetCost())
        iterationBest->Copy(schedule);
*/
#if !USE_ACS
      UpdatePheremone(iterationBest);
#endif
      //printf("Iteration best: ");
      //PrintSchedule(iterationBest); 

      if (bestSchedule->GetCost() == -1 || iterationBest->GetCost() < bestSchedule->GetCost()) {
        bestSchedule->Copy(iterationBest);
        //printf("ACO found schedule with spill cost %d\n",
               //bestSchedule->GetCost());
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, sched length: %d, and "
               "iteration:%d\n",
               bestSchedule->GetCost(), bestSchedule->GetSpillCost(),
               bestSchedule->GetCrntLngth(), iterations);
        noImprovement = 0;
      } else {
        noImprovement++;
        if (noImprovement > noImprovementMax) {
          //for (int i = 0; i < NUMTHREADS; i++)
            //schedules[i].FreeDeviceArrays();
          break;
        }
      } //End run on GPU

    } else { // Run ACO on cpu
      for (int i = 0; i < NUMTHREADS; i++) {
        InstSchedule *schedule = FindOneSchedule();
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
      }
#if !USE_ACS
      UpdatePheremone(iterationBest);
#endif
      //PrintSchedule(iterationBest); 
      if (bestSchedule == nullptr ||
          iterationBest->GetCost() < bestSchedule->GetCost()) {
        if (bestSchedule && bestSchedule != InitialSchedule)
          delete bestSchedule;
        bestSchedule = std::move(iterationBest);
        //printf("ACO found schedule with spill cost %d\n",
               //bestSchedule->GetCost());
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, sched length: %d, and "
               "iteration:%d\n",
               bestSchedule->GetCost(), bestSchedule->GetSpillCost(),
               bestSchedule->GetCrntLngth(), iterations);
        noImprovement = 0;
      } else {
        noImprovement++;
        if (noImprovement > noImprovementMax)
          break;
      }
    } // End run on CPU
#if USE_ACS
    UpdatePheremone(bestSchedule);
#endif
    iterations++;
  }
  printf("Best schedule: ");
  PrintSchedule(bestSchedule);
  schedule_out->Copy(bestSchedule);
  if (DEV_ACO) {
    iterationBest->FreeDeviceArrays();
    delete iterationBest;
    cudaFree(dev_iterationBest);
    delete bestSchedule;
    // cudaFree dev arrays
    for (int i = 0; i < NUMTHREADS; i++) {
      schedules[i]->FreeDeviceArrays();
      cudaFree(host_schedules[i]);
      delete schedules[i];
    }
    cudaFree(dev_schedules);
    delete[] host_schedules;
    delete[] schedules;
    cudaFree(dev_foundBestSched);
/*  Implementation with one array of schedules, has unsolved memory errors
    for (int i = 0; i < NUMTHREADS; i++)
      schedules[i].FreeArrays();
    delete[] schedules;
    cudaFree(dev_schedules);
*/
  } else {
    if (bestSchedule != InitialSchedule)
      delete bestSchedule;
  }
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
  int lstSize = dev_instsWithPrdcsrsSchduld_[GLOBALTID]->size_;
  PriorityArrayList<InstCount, InstCount> *lst = 
	                         dev_instsWithPrdcsrsSchduld_[GLOBALTID];
  SchedInstruction *inst;
  // PriorityArrayList holds keys in decreasing order, so insts with earliest
  // rdyCycle are last on the list
  while (lstSize > 0 && lst->keys_[lstSize - 1] <= cycleNum) {
    inst = dataDepGraph_->GetInstByIndx(lst->elmnts_[lstSize - 1]);
    dev_rdyLst_->AddInst(inst);
    lst->RmvLastElmnt();
    lstSize--;
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
  rdyLst_->AllocDevArraysForParallelACO(NUMTHREADS);
  // Alloc dev array for avlblSlotsInCrntCycle_
  memSize = sizeof(int16_t *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_avlblSlotsInCrntCycle_, memSize));
  // Alloc dev arrays of avlblSlotsInCrntCycle_ for each thread
  memSize = sizeof(int16_t) * issuTypeCnt_;
  for (int i = 0; i < NUMTHREADS; i++) {
    gpuErrchk(cudaMalloc(&dev_avlblSlotsInCrntCycle_[i], memSize));
  }
  // Alloc dev arrays for dev_instsWithPrdcsrsSchduld_
  memSize = sizeof(PriorityArrayList<InstCount, InstCount> *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_instsWithPrdcsrsSchduld_, memSize));
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
  // Create temp PriorityArrayList to copy to dev_instsWithPrdcsrsSchduld_
  Logger::Info("Copying ACO dev_instsWithPrdscsrsSchduld_");
  PriorityArrayList<InstCount, InstCount> *temp;
  temp = new 
	PriorityArrayList<InstCount, InstCount>(dataDepGraph_->GetInstCnt());
  // Allocate and Copy PriorityArrayList for each thread
  for (int i = 0; i < NUMTHREADS; i++) {
    memSize = sizeof(PriorityArrayList<InstCount, InstCount>);
    gpuErrchk(cudaMallocManaged(&dev_instsWithPrdcsrsSchduld_[i], memSize));
    gpuErrchk(cudaMemcpy(dev_instsWithPrdcsrsSchduld_[i], temp, memSize,
			cudaMemcpyHostToDevice));
    // Allocate dev mem for elmnts_ and keys_
    memSize = sizeof(InstCount) * dataDepGraph_->GetInstCnt();
    gpuErrchk(cudaMalloc(&dev_instsWithPrdcsrsSchduld_[i]->elmnts_, memSize));
    gpuErrchk(cudaMalloc(&dev_instsWithPrdcsrsSchduld_[i]->keys_, memSize));
  }
  delete temp;
  // Copy rdyLst_
  Logger::Info("Copying ACO rdyLst_");
  memSize = sizeof(ReadyList);
  gpuErrchk(cudaMallocManaged(&dev_ACOSchedulr->dev_rdyLst_, memSize));
  gpuErrchk(cudaMemcpy(dev_ACOSchedulr->dev_rdyLst_, rdyLst_, memSize,
		       cudaMemcpyHostToDevice));
  rdyLst_->CopyPointersToDevice(dev_ACOSchedulr->dev_rdyLst_, dev_DDG_,
		                NUMTHREADS);

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
    cudaFree(dev_instsWithPrdcsrsSchduld_[i]->elmnts_);
    cudaFree(dev_instsWithPrdcsrsSchduld_[i]->keys_);
    cudaFree(dev_instsWithPrdcsrsSchduld_[i]);
    cudaFree(dev_avlblSlotsInCrntCycle_[i]);
    cudaFree(dev_rsrvSlots_[i]);
  }
  dev_rdyLst_->FreeDevicePointers(NUMTHREADS);
  cudaFree(dev_avlblSlotsInCrntCycle_);
  cudaFree(dev_rsrvSlots_);
  cudaFree(dev_rsrvSlotCnt_);
  cudaFree(dev_instsWithPrdcsrsSchduld_);
  cudaFree(dev_rdyLst_);
  cudaFree(pheremone_.elmnts_);
}
