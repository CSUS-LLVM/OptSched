#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "llvm/ADT/STLExtras.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace llvm::opt_sched;

#ifndef NDEBUG
static void PrintInstruction(SchedInstruction *inst);
#endif
void PrintSchedule(InstSchedule *schedule);

__host__ __device__
double RandDouble(double min, double max) {
#ifdef __CUDA_ARCH__
  double rand = (double)RandomGen::Dev_GetRand32() / INT32_MAX;
#else
  double rand = (double)RandomGen::GetRand32() / INT32_MAX;
#endif
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
			   SchedRegion **dev_rgn, DataDepGraph **dev_DDG,
			   DeviceVector<Choice> **dev_ready, 
			   MachineModel *dev_MM)
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
    int r_pos = (int)(RandDouble(0, 1) * POPULATION_SIZE);
    int s_pos = (int)(RandDouble(0, 1) * POPULATION_SIZE);
    //    int t_pos = (int) (RandDouble(0, 1) *POPULATION_SIZE);
    Choice r = ready[r_pos];
    Choice s = ready[s_pos];
    //    Choice t = ready[t_pos];
    if (print_aco_trace) {
      printf("tournament Start \n");
      printf("array_size: %d\n", POPULATION_SIZE);
      printf("r:\t%d\n", r_pos);
      printf("s:\t%d\n", s_pos);
      //        std::cerr<<"t:\t"<<t_pos<<"\n";

      //std::cerr << "Score r" << Score(lastInst, r) << "\n";
      //std::cerr << "Score s" << Score(lastInst, s) << "\n";
      //         std::cerr<<"Score t"<<Score(lastInst, t)<<"\n";
    }
    if (Score(lastInst, r) >=
        Score(lastInst, s)) //&& Score(lastInst, r) >= Score(lastInst, t))
      return r.inst;
    //     else if (Score(lastInst, s) >= Score(lastInst, r) && Score(lastInst,
    //     s) >= Score(lastInst, t))
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
  printf("returning last instruction\n");
  assert(point < 0.001); // floats should not be this inaccurate
  return ready.back().inst;
}

__host__ __device__
InstSchedule *ACOScheduler::FindOneSchedule(InstSchedule *dev_schedule, 
		                            DeviceVector<Choice> *dev_ready) {
#ifdef __CUDA_ARCH__
  rgn_ = dev_rgn_[threadIdx.x];
#endif
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
#ifdef __CUDA_ARCH__
  ((BBWithSpill *)rgn_)->Dev_InitForSchdulng();
#else
  rgn_->InitForSchdulng();
#endif

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
#ifdef __CUDA_ARCH__
      ((BBWithSpill *)rgn_)->Dev_SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
#else
      rgn_->SchdulInst(inst, crntCycleNum_, crntSlotNum_, false);
#endif
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
  //ready.deallocate();
  return schedule;
}

__global__
void Dev_FindOneSchedule(SchedRegion **dev_rgn, DataDepGraph **dev_DDG,
            ACOScheduler **dev_AcoSchdulr, InstSchedule **dev_schedules,
	    DeviceVector<Choice> **dev_ready) {
  int x = threadIdx.x + blockIdx.x * 5;
  dev_rgn[x]->SetDepGraph(dev_DDG[x]);
  ((BBWithSpill *)dev_rgn[x])->SetRegFiles(dev_DDG[x]->getRegFiles());

  dev_AcoSchdulr[x]->FindOneSchedule(dev_schedules[x], dev_ready[x]);
}

FUNC_RESULT ACOScheduler::FindSchedule(InstSchedule *schedule_out,
                                       SchedRegion *region) {
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
    
    for (int i = 0; i < ants_per_iteration; i++) {
      
      // **** Start Dev_ACO code **** //
      size_t memSize;
      ACOScheduler **host_AcoSchdulr = new ACOScheduler *[100];
      // Copy ACOScheduler to device
      memSize = sizeof(ACOScheduler);
      for (int i = 0; i < 100; i++) {
        if (cudaSuccess != cudaMallocManaged(&host_AcoSchdulr[i], memSize))
          Logger::Fatal("Failed to alloc dev mem for ACOSchdulr");

        if (cudaSuccess != cudaMemcpy(host_AcoSchdulr[i], this, memSize,
                                      cudaMemcpyHostToDevice))
          Logger::Fatal("Failed to copy AcoSchdulr to device");

        CopyPointersToDevice(host_AcoSchdulr[i], dev_DDG_[i], dev_MM_, NULL);
      }
      ACOScheduler **dev_AcoSchdulr;
      memSize = sizeof(ACOScheduler *) * 100;
      if (cudaSuccess != cudaMalloc(&dev_AcoSchdulr, memSize))
        Logger::Fatal("Failed to alloc dev mem for array of dev_AcoSchdulrs");
      if (cudaSuccess != cudaMemcpy(dev_AcoSchdulr, host_AcoSchdulr, memSize,
			            cudaMemcpyHostToDevice))
        Logger::Fatal("Failed to copy array of dev_AcoSchudlrs");
      // Copy Schedule to device
      InstSchedule **dev_schedules;
      InstSchedule **host_schedules = new InstSchedule *[100];
      InstSchedule **schedules = new InstSchedule *[100];
      
      for (int i = 0; i< 100; i++) {
	schedules[i] = new InstSchedule(machMdl_, dataDepGraph_, true);
        schedules[i]->CopyPointersToDevice(dev_MM_);
	memSize = sizeof(InstSchedule);
	if (cudaSuccess != cudaMalloc(&host_schedules[i], memSize))
	  Logger::Fatal("Failed to alloc dev mem for schedule %d: %s", i, 
			  cudaGetErrorString(cudaGetLastError()));
	if (cudaSuccess != cudaMemcpy(host_schedules[i], schedules[i], memSize,
				      cudaMemcpyHostToDevice))
          Logger::Fatal("Failed to copy schedule %d", i);
      }
      memSize = sizeof(InstSchedule *) * 100;
      if (cudaSuccess != cudaMalloc((void**)&dev_schedules, memSize))
        Logger::Fatal("Error allocating dev mem for dev_schedules: %s",
                      cudaGetErrorString(cudaGetLastError()));

      // Copy schedule to device
      if (cudaSuccess != cudaMemcpy(dev_schedules, host_schedules, memSize,
                                    cudaMemcpyHostToDevice))
        Logger::Fatal("Error copying schedule to device: %s",
                      cudaGetErrorString(cudaGetLastError()));

      Logger::Info("Launching Dev_FindOneSchedule()");
      Dev_FindOneSchedule<<<1,30>>>(dev_rgn_, dev_DDG_, dev_AcoSchdulr, 
		                   dev_schedules, dev_ready_);
      cudaDeviceSynchronize();
      Logger::Info("Post Kernel Error: %s", cudaGetErrorString(cudaGetLastError()));

      // Copy schedule to Host
      memSize = sizeof(InstSchedule *) * 100;
      if (cudaSuccess != cudaMemcpy(host_schedules, dev_schedules, memSize,
                                cudaMemcpyDeviceToHost))
        Logger::Fatal("Error copying dev_schedules to host: %s",
                      cudaGetErrorString(cudaGetLastError()));
      memSize = sizeof(InstSchedule);
      for (int i = 0; i < 100; i++) {
	if (cudaSuccess != cudaMemcpy(schedules[i], host_schedules[i], memSize,
				      cudaMemcpyDeviceToHost))
	  Logger::Fatal("Failed to copy schedule %d to host", i);
	schedules[i]->CopyPointersToHost(machMdl_);
	cudaFree(host_schedules[i]);
        host_AcoSchdulr[i]->FreeDevicePointers();
        cudaFree(host_AcoSchdulr[i]);
      }
      cudaFree(dev_schedules);
      cudaFree(dev_AcoSchdulr);
      delete[] host_AcoSchdulr;
      int bestSched = 0;
      for (int i = 1; i < 100; i++)
        if (schedules[bestSched]->GetCost() > schedules[i]->GetCost() && schedules[i]->GetCost() != -1)
	  bestSched = i;
      InstSchedule *schedule = schedules[bestSched];
      for (int i = 0; i < 100; i++)
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
    }
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

void ACOScheduler::CopyPointersToDevice(ACOScheduler *dev_ACOSchedulr,
                                        DataDepGraph *dev_DDG, 
					MachineModel *dev_machMdl,
					InstSchedule *dev_InitSched) {
  dev_ACOSchedulr->InitialSchedule = dev_InitSched;
  size_t memSize;
  // Copy pheremone_->elmnts_ to device
  pheremone_t *dev_elmnts;
  memSize = sizeof(pheremone_t) * pheremone_.alloc_;
  if (cudaSuccess != cudaMalloc(&dev_elmnts, memSize))
    Logger::Fatal("Failed to alloc dev mem for pheremone_->elmnts_");
  
  if (cudaSuccess != cudaMemcpy(dev_elmnts, pheremone_.elmnts_, memSize,
			        cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy pheremone_->elmnts to device");

  dev_ACOSchedulr->pheremone_.elmnts_ = dev_elmnts;

  dev_ACOSchedulr->machMdl_ = dev_machMdl;
  dev_ACOSchedulr->dataDepGraph_ = dev_DDG;
  // Copy slotsPerTypePerCycle_
  int *dev_slotsPerTypePerCycle;
  memSize = sizeof(int) * issuTypeCnt_;
  if (cudaSuccess != cudaMalloc(&dev_slotsPerTypePerCycle, memSize))
    Logger::Fatal("Failed to alloc dev mem for slotsPerTypePerCycle");

  if (cudaSuccess != cudaMemcpy(dev_slotsPerTypePerCycle, slotsPerTypePerCycle_,
			        memSize, cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy slotspertypepercycle to device");

  dev_ACOSchedulr->slotsPerTypePerCycle_ = dev_slotsPerTypePerCycle;

  // Copy instCntPerIssuType_
  InstCount *dev_instCntPerIssuType;
  memSize = sizeof(InstCount) * issuTypeCnt_;
  if (cudaSuccess != cudaMalloc(&dev_instCntPerIssuType, memSize))
    Logger::Fatal("Failed to alloc dev mem for instCntPerIssuType_");
  
  if (cudaSuccess != cudaMemcpy(dev_instCntPerIssuType, instCntPerIssuType_,
			        memSize, cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy instCntPerIssuType_ to device");

  dev_ACOSchedulr->instCntPerIssuType_ = dev_instCntPerIssuType;

  // set root/leaf inst
  dev_ACOSchedulr->rootInst_ = dev_DDG->GetRootInst();
  dev_ACOSchedulr->leafInst_ = dev_DDG->GetLeafInst();

  // Copy frstRdyLstPerCycle_, an array of NULL values
  ArrayList<InstCount> **dev_frstRdyLstPerCycle;
  memSize = sizeof(ArrayList<InstCount> *) * schedUprBound_;
  if (cudaSuccess != cudaMalloc(&dev_frstRdyLstPerCycle, memSize))
    Logger::Fatal("Failed to alloc dev mem for dev_frstRdyLstPerCycle");

  if (cudaSuccess != cudaMemcpy(dev_frstRdyLstPerCycle, frstRdyLstPerCycle_,
			        memSize, cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy frstRdyLstPerCycle_ to device");

  dev_ACOSchedulr->frstRdyLstPerCycle_ = dev_frstRdyLstPerCycle;

  // Copy avlblSlotsInCrntCycle_
  int16_t *dev_avlblSlotsInCrntCycle;
  memSize = sizeof(int16_t) * issuTypeCnt_;
  if (cudaSuccess != cudaMalloc(&dev_avlblSlotsInCrntCycle, memSize))
    Logger::Fatal("Failed to alloc dev mem for avlblSlotsInCrntCycle_");

  if (cudaSuccess != cudaMemcpy(dev_avlblSlotsInCrntCycle, 
			        avlblSlotsInCrntCycle_, memSize,
				cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy avlblSlotsInCrntCycle_ to device");

  dev_ACOSchedulr->avlblSlotsInCrntCycle_ = dev_avlblSlotsInCrntCycle;

  // Copy rdyLst_
  ReadyList *dev_rdyLst;
  memSize = sizeof(ReadyList);
  if (cudaSuccess != cudaMallocManaged(&dev_rdyLst, memSize))
    Logger::Fatal("Failed to alloc dev mem for rdyLst_");

  if (cudaSuccess != cudaMemcpy(dev_rdyLst, rdyLst_, memSize,
			        cudaMemcpyHostToDevice))
    Logger::Fatal("Failed to copy rdyLst_ to device");

  rdyLst_->CopyPointersToDevice(dev_rdyLst, dev_DDG);

  dev_ACOSchedulr->rdyLst_ = dev_rdyLst;
}

void ACOScheduler::FreeDevicePointers() {
  cudaFree(pheremone_.elmnts_);
  cudaFree(slotsPerTypePerCycle_);
  cudaFree(instCntPerIssuType_);
  cudaFree(frstRdyLstPerCycle_);
  cudaFree(avlblSlotsInCrntCycle_);
  rdyLst_->FreeDevicePointers();
  cudaFree(rdyLst_);
}
