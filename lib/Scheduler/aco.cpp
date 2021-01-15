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
  double rand;
#ifdef __CUDA_ARCH__
  rand = curand_uniform(&dev_states_[GLOBALTID]);
#else
  rand = RandDouble(0, 1); 
#endif
//#if USE_ACS
  double choose_best_chance;
  if (use_fixed_bias) {
/*
    if (0 > 1 - (double)fixed_bias / count_)
      choose_best_chance = 0;
    else
      choose_best_chance = 1 - (double)fixed_bias / count_;
*/
    choose_best_chance = (1 - (double)fixed_bias / count_) * (0 < 1 - (double)fixed_bias / count_);
  } else
    choose_best_chance = bias_ratio;

  if (rand < choose_best_chance) {
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
//#endif
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
  pheremone_t point = sum * rand;
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

__device__ int dev_noImprovement = 0; // how many iterations with no improvement
__device__ int dev_doneBlockCnt = 0;
__device__ int mutex = 0;

__global__
void Dev_ACO(SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
            ACOScheduler *dev_AcoSchdulr, InstSchedule **dev_schedules,
            DeviceVector<Choice>**dev_ready, InstSchedule *dev_bestSched,
            int noImprovementMax) {
  // holds cost and index of bestSched per block
  __shared__ int bestCost, bestIndex;
  dev_rgn->SetDepGraph(dev_DDG);
  ((BBWithSpill *)dev_rgn)->SetRegFiles(dev_DDG->getRegFiles());
  int dev_iterations = 0;
  dev_noImprovement = 0;
  
  // Start ACO
  // subtracting numthreads -1 from noImprovementMax to prevent fast
  // blocks from starting an extra iteration.
  while (dev_noImprovement < noImprovementMax - (NUMBLOCKS - 1)) {
    dev_doneBlockCnt = 0;
    // Reset schedules to post constructor state
    dev_schedules[GLOBALTID]->Initialize();
    dev_AcoSchdulr->FindOneSchedule(dev_schedules[GLOBALTID],
                                  dev_ready[GLOBALTID]);
    // Make sure all threads in block have constructed schedules
    __syncthreads();
    // 1 thread per block finds blockIterationBest, updates pheremones, and 
    // compares blockIterationBest to dev_bestSched
    if (threadIdx.x == 0) { // only thread 0 of each block enters this branch
      // iterate over all schedules created by this thread block
      bestCost = dev_schedules[GLOBALTID]->GetCost();
      bestIndex = GLOBALTID;
      for (int i = GLOBALTID + 1; i < GLOBALTID + NUMTHREADSPERBLOCK; i++) {
        if (dev_schedules[i]->GetCost() == -1)
          printf("thread %d of block %d found thread %d made invalid sched\n",
                 GLOBALTID, blockIdx.x, i);
        if (dev_schedules[i]->GetCost() < bestCost) {
          bestCost = dev_schedules[i]->GetCost();
          bestIndex = i;
          // debug
          //printf("Block %d has set bestIndex to %d with bestCost %d\n", blockIdx.x, i, bestCost);
        }
      }
    }
    // make sure thread 0 has picked best sched for the block before
    // updating pheremones
    __syncthreads();
    // debug
    //printf("Thread %d has bestIndex = %d and bestCost = %d\n", GLOBALTID, bestIndex, bestCost);
    dev_AcoSchdulr->UpdatePheremone(dev_schedules[bestIndex]);
    // 1 thread per block compares block iteration best to overall bestsched
    if (threadIdx.x == 0) {
      // Compare to initialSched/current best
      if (bestCost < dev_bestSched->GetCost()) {
        // mutex lock updating best sched so multiple blocks dont write
        // at the same time
        while(atomicCAS(&mutex, 0, 1) != 0);
        // check if this blocks iteration best schedule is still better
        // since another block could have lowered cost of best sched
        if (bestCost < dev_bestSched->GetCost()) {
          dev_bestSched->Copy(dev_schedules[bestIndex]);
          printf("ACO found schedule "
                 "cost:%d, rp cost:%d, sched length: %d, and "
                 "iteration:%d\n",
               dev_bestSched->GetCost(), dev_bestSched->GetSpillCost(),
               dev_bestSched->GetCrntLngth(), dev_iterations);
          dev_noImprovement = 0;
        }
        // unlock mutex 
        atomicExch(&mutex, 0);
      } else {
        atomicAdd(&dev_noImprovement, 1);
        if (dev_noImprovement > noImprovementMax)
          break;
      }
      // wait for other blocks to finish before starting next iteration
      atomicAdd(&dev_doneBlockCnt, 1);
      while (dev_doneBlockCnt < NUMBLOCKS);
      // if dev_noImprovement is not a multiple of NUMBLOCKS after blocks finish
      // current iteration, that means a block has found a new bestSched
      // and reset dev_noImprovement, and other blocks have increased
      // it after not finding a new bestSched. Set to 0 if this is the case.
      if (dev_noImprovement % NUMBLOCKS != 0)
          dev_noImprovement = 0;
    }
    // make sure no threads reset schedule before above operations complete
    __syncthreads();
    dev_iterations++;
  }
  if (threadIdx.x == 0)
    printf("Block %d finished Dev_ACO after %d iterations\n", blockIdx.x, dev_iterations); 
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
  InstSchedule *iterationBest;
  
  if (DEV_ACO) { // Run ACO on device
    size_t memSize;
    // Update pheremones on device
    CopyPheremonesToDevice(dev_AcoSchdulr);
    Logger::Info("Creating and copying schedules to device");
    // dev array of dev pointers to schedules
    InstSchedule **dev_schedules;
    // holds device copy of best sched, to be copied back to host after kernel
    InstSchedule *dev_bestSched;
    // host array of dev pointers to schedules, 1 per thread plus 1 extra per
    // block to hold block iteration best
    InstSchedule **host_schedules = new InstSchedule *[NUMTHREADS];
    // holds schedules to be copied over
    InstSchedule **schedules = new InstSchedule *[NUMTHREADS];
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
    bestSchedule->AllocateOnDevice(dev_MM_);
    bestSchedule->CopyArraysToDevice();
    memSize = sizeof(InstSchedule);
    gpuErrchk(cudaMalloc((void**)&dev_bestSched, memSize));
    gpuErrchk(cudaMemcpy(dev_bestSched, bestSchedule, memSize,
                         cudaMemcpyHostToDevice));
    Logger::Info("Launching Dev_ACO");
    // Launch with noImprovementMax * NUMBLOCKS so each threadblock can
    // increment dev_noImprovement once per iteration
    Dev_ACO<<<NUMBLOCKS,NUMTHREADSPERBLOCK>>>(dev_rgn_, dev_DDG_,
                      dev_AcoSchdulr, dev_schedules, dev_ready_, dev_bestSched,
                      noImprovementMax * NUMBLOCKS);
    cudaDeviceSynchronize();
    Logger::Info("Post Kernel Error: %s", cudaGetErrorString(cudaGetLastError()));
    // Copy dev_bestSched back to host
    memSize = sizeof(InstSchedule);
    gpuErrchk(cudaMemcpy(bestSchedule, dev_bestSched, memSize,
                         cudaMemcpyDeviceToHost));
    bestSchedule->CopyArraysToHost();
    // Free allocated memory that is no longer needed
    bestSchedule->FreeDeviceArrays();
    cudaFree(dev_bestSched);
    for (int i = 0; i < NUMTHREADS; i++) {
      schedules[i]->FreeDeviceArrays();
      cudaFree(host_schedules[i]);
      delete schedules[i];
    }
    delete[] host_schedules;
    delete[] schedules;

    } else { // Run ACO on cpu
      while (true) {
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
#if USE_ACS
      UpdatePheremone(bestSchedule);
#endif
      iterations++;
    }
  } // End run on CPU

  printf("Best schedule: ");
  PrintSchedule(bestSchedule);
  schedule_out->Copy(bestSchedule);
  if (bestSchedule != InitialSchedule)
    delete bestSchedule;
  if (!DEV_ACO)
    printf("ACO finished after %d iterations\n", iterations);
  return RES_SUCCESS;
}

__host__ __device__
void ACOScheduler::UpdatePheremone(InstSchedule *schedule) {
#ifdef __CUDA_ARCH__ // device version of function
  // parallel on a block level, use threadIdx.x instead of GLOBALTID
  int instNum = threadIdx.x;
  // Each thread updates pheremone table for 1 instruction
  // For the case NUMTHREADSPERBLOCK < count_, increase instNum by 
  // NUMTHREADSPERBLOCK at the end of the loop.
  InstCount lastInstNum = -1;
  pheremone_t *pheremone;
  while (instNum < count_) {
    // debug
    //printf("Updating pheremone for instNum = %d\n", instNum);
    // Get the instruction that comes before inst in the schedule
    // if instNum == count_ - 2 it has the root inst and lastInstNum = -1
    lastInstNum = schedule->GetPrevInstNum(instNum);
    // Get corresponding pheremone and update it
    pheremone = &Pheremone(lastInstNum, instNum);
    *pheremone = *pheremone + 1 / (schedule->GetCost() + 1);
    // decay pheremone for all trails leaving instNum
    for (int j = 0; j < count_; j++) {
      Pheremone(instNum, j) *= (1 - decay_factor);
    }
    // Increase instNum by NUMTHREADSPERBLOCK in case there are less threads
    // per block than instruction in the region
    instNum += NUMTHREADSPERBLOCK;
  }
  if (print_aco_trace)
    PrintPheremone();

#else // host version of function
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
#endif
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
