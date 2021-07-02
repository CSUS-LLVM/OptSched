#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/dev_defines.h"
#include <thrust/functional.h>
#include <cooperative_groups.h>
#include "llvm/ADT/STLExtras.h"
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace llvm::opt_sched;
namespace cg = cooperative_groups;

#ifndef NDEBUG
static void PrintInstruction(SchedInstruction *inst);
#endif
void PrintSchedule(InstSchedule *schedule);

double RandDouble(double min, double max) {
  double rand = (double)RandomGen::GetRand32() / INT32_MAX;
  return (rand * (max - min)) + min;
}

#define USE_ACS 0
#define TWO_STEP 1
#define MIN_DEPOSITION 1
#define MAX_DEPOSITION 6
#define MAX_DEPOSITION_MINUS_MIN (MAX_DEPOSITION - MIN_DEPOSITION)
#define ACO_SCHED_STALLS 1
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
                           bool IsPostBB, SchedRegion *dev_rgn,
                           DataDepGraph *dev_DDG, 
                           DeviceVector<Choice> *dev_ready,
			   MachineModel *dev_MM, curandState_t *dev_states)
    : ConstrainedScheduler(dataDepGraph, machineModel, upperBound) {
  VrfySched_ = vrfySched;
  this->IsPostBB = IsPostBB;
  prirts_ = priorities;
  rdyLst_ = new ReadyList(dataDepGraph_, priorities);
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();
  dev_rgn_ = dev_rgn;
  dev_DDG_ = dev_DDG;
  dev_ready_ = dev_ready;
  dev_MM_ = dev_MM;
  dev_states_ = dev_states;
  dev_pheromone_elmnts_alloced_ = false;
  numAntsTerminated_ = 0;

  use_fixed_bias = schedIni.GetBool("ACO_USE_FIXED_BIAS");
  use_tournament = schedIni.GetBool("ACO_TOURNAMENT");
  bias_ratio = schedIni.GetFloat("ACO_BIAS_RATIO");
  local_decay = schedIni.GetFloat("ACO_LOCAL_DECAY");
  decay_factor = schedIni.GetFloat("ACO_DECAY_FACTOR");
  ants_per_iteration = schedIni.GetInt("ACO_ANT_PER_ITERATION");
  print_aco_trace = schedIni.GetBool("ACO_TRACE");
  IsTwoPassEn = schedIni.GetBool("USE_TWO_PASS");

  /*
  std::cerr << "useOldAlg===="<<useOldAlg<<"\n\n";
  std::cerr << "heuristicImportance_===="<<heuristicImportance_<<"\n\n";
  std::cerr << "tournament===="<<tournament<<"\n\n";
  std::cerr << "bias_ratio===="<<bias_ratio<<"\n\n";
  std::cerr << "local_decay===="<<local_decay<<"\n\n";
  std::cerr << "decay_factor===="<<decay_factor<<"\n\n";
  std::cerr << "ants_per_iteration===="<<ants_per_iteration<<"\n\n";
  */

  int pheromone_size = (count_ + 1) * count_;
  pheromone_.resize(pheromone_size);
  InitialSchedule = nullptr;
}

ACOScheduler::~ACOScheduler() { delete rdyLst_; }

// Pheromone table lookup
// -1 means no instruction, so e.g. pheromone(-1, 10) gives pheromone on path
// from empty schedule to schedule only containing instruction 10
__host__ __device__
pheromone_t &ACOScheduler::Pheromone(SchedInstruction *from,
                                     SchedInstruction *to) {
  assert(to != NULL);
  int fromNum = -1;
  if (from != NULL)
    fromNum = from->GetNum();
  return Pheromone(fromNum, to->GetNum());
}

__host__ __device__
pheromone_t &ACOScheduler::Pheromone(InstCount from, InstCount to) {
  int row = 0;
  if (from != -1)
    row = from + 1;
  return pheromone_[(row * count_) + to];
}

__host__ __device__
double ACOScheduler::Score(SchedInstruction *from, Choice choice) {
  return Pheromone(from, choice.inst) *
         pow(choice.heuristic, heuristicImportance_);
}

__host__ __device__
bool ACOScheduler::shouldReplaceSchedule(InstSchedule *OldSched,
                                         InstSchedule *NewSched,
                                         bool IsGlobal) {
  // return true if the old schedule is null (eg:there is no old schedule)
  // return false if the new schedule is is NULL
  if (!NewSched) {
    return false;
  } else if (!OldSched) {
    return true;
  }

  // return false if new schedule is invalid
  // return true if old schedule is invalid
  if (NewSched->GetCost() == INVALID_VALUE)
    return false;
  if (OldSched->GetCost() == INVALID_VALUE)
    return true;

  // if it is the 1st pass return the cost comparison
  // if it is the 2nd pass return true if the RP cost and ILP cost is less
#ifdef __CUDA_ARCH__
  bool isSecondPass = dev_rgn_->IsSecondPass();
#else
  bool isSecondPass = rgn_->IsSecondPass(); 
#endif
  if (!IsTwoPassEn || !isSecondPass) {
    InstCount NewCost = (!IsTwoPassEn) ? NewSched->GetCost() : NewSched->GetNormSpillCost();
    InstCount OldCost = (!IsTwoPassEn) ? OldSched->GetCost() : OldSched->GetNormSpillCost();

    if (NewCost < OldCost)
      return true;
    else
      return false;
  }
  else {
    InstCount NewCost = NewSched->GetExecCost();
    InstCount OldCost = OldSched->GetExecCost();
    InstCount NewSpillCost = NewSched->GetNormSpillCost();
    InstCount OldSpillCost = OldSched->GetNormSpillCost();
    // Lower Spill Cost always wins
    if (NewSpillCost < OldSpillCost)
      return true;
    else if (NewSpillCost == OldSpillCost && NewCost < OldCost)
      return true;
    else
      return false;
  }
}

__host__ __device__
Choice ACOScheduler::SelectInstruction(DeviceVector<Choice> &ready,
                                       SchedInstruction *lastInst,
                                       double ScoreSum) {


  //genereate the random numbers that we will need for deciding if
  //we are going to use the fixed bias or if we are going to use
  //fitness porportional selection.  Generate the number used for
  //the fitness porportional selection point
  double rand;
  pheromone_t point;
#ifdef __CUDA_ARCH__
  rand = curand_uniform(&dev_states_[GLOBALTID]);
  point = ScoreSum * curand_uniform(&dev_states_[GLOBALTID]);
#else
  rand = RandDouble(0, 1);
  point = RandDouble(0, ScoreSum);
#endif

  //here we compute the chance that we will use fp selection or auto pick the best
  double choose_best_chance;
  if (use_fixed_bias) { //this is a non-diverging if stmt
    choose_best_chance = (1 - (double)fixed_bias / count_) * (0 < 1 - (double)fixed_bias / count_);
  } else
    choose_best_chance = bias_ratio;


  //here we determine the max scoring instruction and the fp choice
  //this code is a bit dense, but what we are doing is picking the
  //indices of the max and fp choice instructions
  //The only branch in this code is the branch for deciding to stay in the loop vs exit the loop
  //this will diverge if two ants ready lists are of different sizes
  size_t maxIndx=0, fpIndx=0;
  pheromone_t max = -1, scoreProgress = 0;
  bool foundFPIndx = false;
  for (size_t i = 0; i < ready.size(); ++i) {
    const Choice &choice = ready[i];

    //code for picking the max
    bool CurrIsMax = choice.Score > max;
    max = CurrIsMax ? choice.Score : max;
    maxIndx = CurrIsMax ? i : maxIndx;

    //code for picking the fitness proportional choice
    scoreProgress += choice.Score;
    bool PastPoint = scoreProgress >= point;
    bool SetFP = PastPoint & !foundFPIndx;
    fpIndx = SetFP ? i : fpIndx;
    foundFPIndx |= PastPoint;
  }

  //finally we pick whether we will return the fp choice or max score inst w/o using a branch
  bool UseMax = rand < choose_best_chance;
  size_t indx = UseMax ? maxIndx : fpIndx;
  return ready[indx];

}

__host__ __device__
InstSchedule *ACOScheduler::FindOneSchedule(InstCount RPTarget, 
                                            InstSchedule *dev_schedule,
		                            DeviceVector<Choice> *dev_ready) {
#ifdef __CUDA_ARCH__ // device version of function
  SchedInstruction *inst = NULL;
  SchedInstruction *lastInst = NULL;
  InstSchedule *schedule = dev_schedule;
  InstCount maxPriority = dev_rdyLst_->MaxPriority();
  bool IsSecondPass = dev_rgn_->IsSecondPass();
  if (maxPriority == 0)
    maxPriority = 1; // divide by 0 is bad
  Initialize_();
  ((BBWithSpill *)dev_rgn_)->Dev_InitForSchdulng();

  SchedInstruction *waitFor = NULL;
  InstCount waitUntil = 0;
  double maxPriorityInv = 1 / maxPriority;
  DeviceVector<Choice> *ready = dev_ready;
  while (!IsSchedComplete_()) {
    UpdtRdyLst_(dev_crntCycleNum_[GLOBALTID], dev_crntSlotNum_[GLOBALTID]);

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    if (!waitFor) {
      // if we have not already committed to schedule an instruction
      // next then pick one. First add ready instructions.  Including
      //"illegal" e.g. blocked instructions

      // convert the ready list from a custom priority queue to a std::vector,
      // much nicer for this particular scheduler
      double ScoreSum=0;
      unsigned long heuristic;
      ready->reserve(dev_rdyLst_->GetInstCnt());
      SchedInstruction *rInst = dev_rdyLst_->GetNextPriorityInst(heuristic);
      while (rInst != NULL) {
        if (ACO_SCHED_STALLS || ChkInstLglty_(inst)) {
          Choice c;
          c.inst = rInst;
          c.heuristic = (double)heuristic * maxPriorityInv + 1;
          c.readyOn = 0;
          c.Score = Score(lastInst,c);
          ScoreSum += c.Score;
          ready->push_back(c);
        }
        rInst = dev_rdyLst_->GetNextPriorityInst(heuristic);
      }
      dev_rdyLst_->ResetIterator();

#if ACO_SCHED_STALLS
      // add all instructions that are waiting due to latency to the choices
      // list
      PriorityArrayList<InstCount, InstCount> *lst = 
	                         dev_instsWithPrdcsrsSchduld_[GLOBALTID];
      SchedInstruction *fIns;
      for (InstCount fInstNum = lst->GetLastElmnt(); fInstNum != END;
           fInstNum = lst->GetPrevElmnt()) {
        fIns = dataDepGraph_->GetInstByIndx(fInstNum);
        bool changed;
        unsigned long heuristic = dev_rdyLst_->CmputKey_(fIns, false, changed);
        Choice c;
        c.inst = fIns;
        c.heuristic = (double)heuristic * maxPriorityInv + 1;
        c.readyOn = lst->GetCrntKey();
        c.Score = Score(lastInst,c);
        ScoreSum += c.Score;
        ready->push_back(c);
      }
      lst->ResetIterator();
#endif
      
      if (!ready->empty()) {
        Choice Sel = SelectInstruction(*ready, lastInst, ScoreSum);
        waitUntil = Sel.readyOn;
        inst = Sel.inst;
        if (waitUntil > dev_crntCycleNum_[GLOBALTID] || !ChkInstLglty_(inst)) {
          waitFor = inst;
          inst = NULL;
        }
      }
      if (inst != NULL) {
#if USE_ACS
        // local pheromone decay
        pheromone_t *pheromone = &Pheromone(lastInst, inst);
        *pheromone = 
          (1 - local_decay) * *pheromone + local_decay * initialValue_;
#endif
        lastInst = inst;
      }
    }

    // 2)Schedule a stall if we are still waiting, Schedule the instruction we
    // are waiting for if possible, decrement waiting time
    if (waitFor && waitUntil <= dev_crntCycleNum_[GLOBALTID]) {
      if (ChkInstLglty_(waitFor)) {
        inst = waitFor;
        waitFor = NULL;
      }
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
      // In the second pass, calculate cost incrementally and terminate
      // ants that violate the RPTarget early
      ((BBWithSpill *)dev_rgn_)->Dev_SchdulInst(inst,
                                            dev_crntCycleNum_[GLOBALTID],
                                            dev_crntSlotNum_[GLOBALTID],
                                            false);
      // If an ant violates the RP cost constraint, terminate further
      // schedule construction
      if (((BBWithSpill *)dev_rgn_)->GetCrntSpillCost() > RPTarget) {
        // set schedule cost to INVALID_VALUE so it is not considered for
        // iteration best or global best
        schedule->SetCost(INVALID_VALUE);
        // keep track of ants terminated
        atomicAdd(&numAntsTerminated_, 1);
        dev_rdyLst_->ResetIterator();
        dev_rdyLst_->Reset();
        ready->clear();
        dev_instsWithPrdcsrsSchduld_[GLOBALTID]->Reset();
        // end schedule construction
        return NULL;
      } 
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
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
    dev_rdyLst_->ResetIterator();
    ready->clear();
  }
  dev_rgn_->UpdateScheduleCost(schedule);
  return schedule;

#else  // **** Host version of function ****
  SchedInstruction *lastInst = NULL;
  InstSchedule *schedule;
  schedule = new InstSchedule(machMdl_, dataDepGraph_, true);
  InstCount maxPriority = rdyLst_->MaxPriority();
  bool IsSecondPass = rgn_->IsSecondPass();
  if (maxPriority == 0)
    maxPriority = 1; // divide by 0 is bad
  Initialize_();
  rgn_->InitForSchdulng();

  SchedInstruction *waitFor = NULL;
  InstCount waitUntil = 0;
  double maxPriorityInv = 1 / maxPriority;
  DeviceVector<Choice> *ready = 
      new DeviceVector<Choice>(dataDepGraph_->GetInstCnt());
  SchedInstruction *inst = NULL;
  while (!IsSchedComplete_()) {
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    inst = NULL;
    if (!waitFor) {
      // if we have not already committed to schedule an instruction
      // next then pick one. First add ready instructions.  Including
      //"illegal" e.g. blocked instructions

      // convert the ready list from a custom priority queue to a std::vector,
      // much nicer for this particular scheduler
      double ScoreSum=0;
      unsigned long heuristic;
      ready->reserve(rdyLst_->GetInstCnt());
      SchedInstruction *rInst = rdyLst_->GetNextPriorityInst(heuristic);
      while (rInst != NULL) {
        if (ACO_SCHED_STALLS || ChkInstLglty_(rInst)) {
          Choice c;
          c.inst = rInst;
          c.heuristic = (double)heuristic * maxPriorityInv + 1;
          c.readyOn = 0;
          c.Score = Score(lastInst,c);
          ScoreSum += c.Score;
          ready->push_back(c);
        }
        rInst = rdyLst_->GetNextPriorityInst(heuristic);
      }
      rdyLst_->ResetIterator();

#if ACO_SCHED_STALLS
      // add all instructions that are waiting due to latency to the choices
      // list
      for (InstCount fCycle = 1; fCycle < dataDepGraph_->GetMaxLtncy() &&
                                 crntCycleNum_ + fCycle < schedUprBound_;
           ++fCycle) {
        ArrayList<InstCount> *futureReady =
            frstRdyLstPerCycle_[crntCycleNum_ + fCycle];
        if (!futureReady)
          continue;

        SchedInstruction *fIns;
        for (InstCount fInstNum = futureReady->GetFrstElmnt(); fInstNum != END;
             fInstNum = futureReady->GetNxtElmnt()) {
          fIns = dataDepGraph_->GetInstByIndx(fInstNum);
          bool changed;
          unsigned long heuristic = rdyLst_->CmputKey_(fIns, false, changed);
          Choice c;
          c.inst = fIns;
          c.heuristic = (double)heuristic * maxPriorityInv + 1;
          c.readyOn = crntCycleNum_ + fCycle;
          c.Score = Score(lastInst,c);
          ScoreSum += c.Score;
          ready->push_back(c);
        }
        futureReady->ResetIterator();
      }
#endif

      if (!ready->empty()) {
        Choice Sel = SelectInstruction(*ready, lastInst, ScoreSum);
        waitUntil = Sel.readyOn;
        inst = Sel.inst;
        if (waitUntil > crntCycleNum_ || !ChkInstLglty_(inst)) {
          waitFor = inst;
          inst = NULL;
        }
      }
      if (inst != NULL) {
  #if USE_ACS
        // local pheromone decay
        pheromone_t *pheromone = &Pheromone(lastInst, inst);
        *pheromone = (1 - local_decay) * *pheromone + local_decay * initialValue_;
  #endif
        lastInst = inst;
      }
    }

    // 2)Schedule a stall if we are still waiting, Schedule the instruction we
    // are waiting for if possible, decrement waiting time
    if (waitFor && waitUntil <= crntCycleNum_) {
      if (ChkInstLglty_(waitFor)) {
        inst = waitFor;
        waitFor = NULL;
      }
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
      // If an ant violates the RP cost constraint, terminate further
      // schedule construction
      if (((BBWithSpill*)rgn_)->GetCrntSpillCost() > RPTarget) {
        // end schedule construction
        // keep track of ants terminated
        numAntsTerminated_++;
        delete rdyLst_;
        delete ready;
        rdyLst_ = new ReadyList(dataDepGraph_, prirts_); 
        delete schedule;
        return NULL;
      }
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
  delete ready;
  rgn_->UpdateScheduleCost(schedule);
  return schedule;
#endif
}

// Reduce to only index of best schedule per 2 blocks in output array
__inline__ __device__
void reduceToBestSchedPerBlock(InstSchedule **dev_schedules, int *blockBestIndex, ACOScheduler *dev_AcoSchdulr) {
  __shared__ int sdata[NUMTHREADSPERBLOCK];
  uint gtid = GLOBALTID;
  uint tid = threadIdx.x;
  int blockSize = NUMTHREADSPERBLOCK;
  
  // load candidate schedules into smem
  if (dev_AcoSchdulr->shouldReplaceSchedule(dev_schedules[gtid * 2], dev_schedules[gtid * 2 + 1], false))
    sdata[tid] = gtid * 2 + 1;
  else
    sdata[tid] = gtid * 2;
  __syncthreads();

  // do reduction on indexes in shared mem
  for (uint s = 1; s < blockDim.x; s*=2) {
    if (tid%(2*s) == 0) {
      if (dev_AcoSchdulr->shouldReplaceSchedule(
          dev_schedules[sdata[tid]], 
          dev_schedules[sdata[tid + s]], false))
        sdata[tid] = sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    blockBestIndex[blockIdx.x] = sdata[0];

}

// 1 block only to allow proper synchronization
// reduce to only one best index. At the end of this function globalBestIndex
// should be in blockBestIndex[0]
__inline__ __device__
void reduceToBestSched(InstSchedule **dev_schedules, int *blockBestIndex, ACOScheduler *dev_AcoSchdulr) {
  __shared__ int sBestIndex[NUMBLOCKS/4];
  uint tid = threadIdx.x;
  int index, sBestIndex1, sBestIndex2;
  
  // Load best indices into shared mem, reduce by half while doing so
  // If there are more than 64 schedules in blockBestIndex, some threads
  // will have to load in more than one value
  while (tid < NUMBLOCKS/4) {
    if (dev_AcoSchdulr->shouldReplaceSchedule(dev_schedules[blockBestIndex[tid * 2]], 
                                              dev_schedules[blockBestIndex[tid * 2 + 1]], false))
      sBestIndex[tid] = blockBestIndex[tid * 2 + 1];
    else
      sBestIndex[tid] = blockBestIndex[tid * 2];

    tid += blockDim.x;
  }
  __syncthreads();

  // reduce in smem
  for (uint s = 1; s < NUMBLOCKS/4; s *= 2) {
    tid = threadIdx.x;
    // if there are more than 32 schedules in smem, a thread
    // may reduce more than once per loop
    while (tid < NUMBLOCKS/4) {
      index = 2 * s * tid;

      if (index + s < NUMBLOCKS/4) {
        sBestIndex1 = sBestIndex[index];
        sBestIndex2 = sBestIndex[index + s];
        if (dev_AcoSchdulr->shouldReplaceSchedule(
            dev_schedules[sBestIndex1],
            dev_schedules[sBestIndex2], false))
          sBestIndex[index] = sBestIndex2;
      }
      tid += blockDim.x;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    blockBestIndex[0] = sBestIndex[0];
}

// default pheromone update scheme, one iteration best schedule is used to
// update pheromones every iteration
#define ONE_PER_ITER 0
// Update pheromones with best schedule found by each block every iteration
#define ONE_PER_BLOCK 1
// Update pheromones with all schedules
#define ALL 2
// select which pheromone update scheme to use
#define PHER_UPDATE_SCHEME ONE_PER_ITER

__device__ int globalBestIndex, dev_noImprovement;

__global__
void Dev_ACO(SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
            ACOScheduler *dev_AcoSchdulr, InstSchedule **dev_schedules,
            DeviceVector<Choice> *dev_ready, InstSchedule *dev_bestSched,
            int noImprovementMax, int *blockBestIndex) {
  // holds cost and index of bestSched per block
  __shared__ int bestIndex, dev_iterations;
  __shared__ bool needsSLIL;
  needsSLIL = ((BBWithSpill *)dev_rgn)->needsSLIL();
  bool IsSecondPass = dev_rgn->IsSecondPass();
  dev_rgn->SetDepGraph(dev_DDG);
  ((BBWithSpill *)dev_rgn)->SetRegFiles(dev_DDG->getRegFiles());
  dev_noImprovement = 0;
  dev_iterations = 0;
  // Used to synchronize all launched threads
  auto threadGroup = cg::this_grid();
  // Get RPTarget
  InstCount RPTarget;

  // If in second pass and not using SLIL, set RPTarget
  if (!needsSLIL)
    RPTarget = dev_bestSched->GetSpillCost();
  else
    RPTarget = INT_MAX;

  // Start ACO
  while (dev_noImprovement < noImprovementMax) {
    // Reset schedules to post constructor state
    dev_schedules[GLOBALTID]->Initialize();
    dev_AcoSchdulr->FindOneSchedule(RPTarget,
                                    dev_schedules[GLOBALTID],
                                    &dev_ready[GLOBALTID]);
    // Sync threads after schedule creation
    threadGroup.sync();
    globalBestIndex = INVALID_VALUE;
    // reduce dev_schedules to 1 best schedule per block
    if (GLOBALTID < NUMTHREADS/2)
      reduceToBestSchedPerBlock(dev_schedules, blockBestIndex, dev_AcoSchdulr);

    threadGroup.sync();

    // one block to reduce blockBest schedules to one best schedule
    if (blockIdx.x == 0)
      reduceToBestSched(dev_schedules, blockBestIndex, dev_AcoSchdulr);

    threadGroup.sync();    

    if (GLOBALTID == 0 && 
        dev_schedules[blockBestIndex[0]]->GetCost() != INVALID_VALUE)
      globalBestIndex = blockBestIndex[0];

    // perform pheremone update based on selected scheme
#if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
    // Another hard sync point after iteration best selection
    threadGroup.sync();
    if (globalBestIndex != INVALID_VALUE) 
      dev_AcoSchdulr->UpdatePheromone(dev_schedules[globalBestIndex]);
#elif (PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
    // each block finds its blockIterationBest
    if (threadIdx.x == 0) {
      bestCost = dev_schedules[GLOBALTID]->GetCost();
      bestIndex = GLOBALTID; 
      for (int i = GLOBALTID + 1; i < GLOBALTID + NUMTHREADSPERBLOCK; i++) {
        if (dev_schedules[i]->GetCost() < bestCost) {
          bestCost = dev_schedules[i]->GetCost();
          bestIndex = i; 
        }
      }   
    }
    // wait for thread 0 of each block to find blockIterationBest
    threadGroup.sync();
    dev_AcoSchdulr->UpdatePheromone(dev_schedules[bestIndex]);
#elif (PHER_UPDATE_SCHEME == ALL)
    // each block loops over all schedules created by its threads and
    // updates pheromones in block level parallel
    for (int i = blockIdx.x * NUMTHREADSPERBLOCK; 
         i < ((blockIdx.x + 1) * NUMTHREADSPERBLOCK); i++) {
      dev_AcoSchdulr->UpdatePheromone(dev_schedules[i]);
    }
#endif
    // 1 thread compares iteration best to overall bestsched
    if (GLOBALTID == 0) {
      // Compare to initialSched/current best
      if (globalBestIndex != INVALID_VALUE &&
          dev_AcoSchdulr->shouldReplaceSchedule(dev_bestSched, 
                                                dev_schedules[globalBestIndex], 
                                                true)) {
        dev_bestSched->Copy(dev_schedules[globalBestIndex]);
        // update RPTarget if we are in second pass and not using SLIL
        if (!needsSLIL)
          RPTarget = dev_bestSched->GetSpillCost();
        printf("New best sched found by thread %d\n", globalBestIndex);
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, exec cost: %d, and "
               "iteration:%d"
               " (sched length: %d, abs rp cost: %d, rplb: %d)\n",
             dev_bestSched->GetCost(), dev_bestSched->GetNormSpillCost(),
             dev_bestSched->GetExecCost(), dev_iterations,
             dev_bestSched->GetCrntLngth(), dev_bestSched->GetSpillCost(),
             dev_rgn->GetRPCostLwrBound());
#if !RUNTIME_TESTING
          dev_noImprovement = 0;
#else
          // for testing compile times disable resetting dev_noImprovement to
          // allow the same number of iterations every time
          atomicAdd(&dev_noImprovement, 1);
#endif     
      } else {
        atomicAdd(&dev_noImprovement, 1);
        if (dev_noImprovement > noImprovementMax)
          break;
      }
    }
    // wait for other blocks to finish before starting next iteration
    threadGroup.sync();
    // make sure no threads reset schedule before above operations complete
    if (threadIdx.x == 0)
      dev_iterations++;
  }
  if (GLOBALTID == 0) {
    printf("ACO finished after %d iterations\n", dev_iterations);
    printf("%d ants terminated early\n", dev_AcoSchdulr->GetNumAntsTerminated());
  }
}

FUNC_RESULT ACOScheduler::FindSchedule(InstSchedule *schedule_out,
                                       SchedRegion *region,
				       ACOScheduler *dev_AcoSchdulr) {
  rgn_ = region;

  // get settings
  Config &schedIni = SchedulerOptions::getInstance();
  bool IsFirst = !rgn_->IsSecondPass();
  heuristicImportance_ = schedIni.GetInt(
      IsFirst ? "ACO_HEURISTIC_IMPORTANCE" : "ACO2P_HEURISTIC_IMPORTANCE");
  if (dev_AcoSchdulr)
    dev_AcoSchdulr->heuristicImportance_ = heuristicImportance_;
  fixed_bias = schedIni.GetInt(IsFirst ? "ACO_FIXED_BIAS" : "ACO2P_FIXED_BIAS");
  if (dev_AcoSchdulr)
    dev_AcoSchdulr->fixed_bias = fixed_bias;
  noImprovementMax = schedIni.GetInt(IsFirst ? "ACO_STOP_ITERATIONS"
                                             : "ACO2P_STOP_ITERATIONS");
  if (dev_AcoSchdulr)
    dev_AcoSchdulr->noImprovementMax = noImprovementMax;

  // compute the relative maximum score inverse
  ScRelMax = rgn_->GetHeuristicCost();

  // initialize pheromone
  // for this, we need the cost of the pure heuristic schedule
  int pheromone_size = (count_ + 1) * count_;
  for (int i = 0; i < pheromone_size; i++)
    pheromone_[i] = 1;
  initialValue_ = 1;
  InstCount MaxRPTarget = std::numeric_limits<InstCount>::max();
  InstSchedule *heuristicSched = FindOneSchedule(MaxRPTarget);
  InstCount heuristicCost =
      heuristicSched->GetCost() + 1; // prevent divide by zero
  InstCount InitialCost = InitialSchedule ? InitialSchedule->GetCost() : 0;

#if USE_ACS
  initialValue_ = 2.0 / ((double)count_ * heuristicCost);
#else
  initialValue_ = (double)NUMTHREADS / heuristicCost;
#endif
  for (int i = 0; i < pheromone_size; i++)
    pheromone_[i] = initialValue_;
  std::cerr << "initialValue_" << initialValue_ << std::endl;
  InstSchedule *bestSchedule = InitialSchedule;
  if (bestSchedule) {
    UpdatePheromone(bestSchedule);
  }
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  InstSchedule *iterationBest = nullptr;

  if (DEV_ACO) { // Run ACO on device
    size_t memSize;
    // Update pheromones on device
    CopyPheromonesToDevice(dev_AcoSchdulr);
    Logger::Info("Creating and copying schedules to device"); 
    // An array to temporarily hold schedules to be copied over
    memSize = sizeof(InstSchedule) * NUMTHREADS;
    InstSchedule *temp_schedules = (InstSchedule *)malloc(memSize);
    // An array of pointers to schedules which are copied over
    InstSchedule **host_schedules = new InstSchedule *[NUMTHREADS];
    // Allocate one large array that will be split up between the dev arrays
    // of all InstSchedules. Massively decrease calls to cudaMalloc/Free
    InstCount *dev_temp;
    size_t sizePerSched = bestSchedule->GetSizeOfDevArrays();
    memSize = sizePerSched * NUMTHREADS * sizeof(InstCount);
    gpuErrchk(cudaMalloc(&dev_temp, memSize));
    memSize = sizeof(InstSchedule);
    for (int i = 0; i < NUMTHREADS; i++) {
      // Create new schedule
      host_schedules[i] = new InstSchedule(machMdl_, dataDepGraph_, true);
      // Pass a dev array to the schedule to be divided up between the required
      // dev arrays for InstSchedule
      host_schedules[i]->SetDevArrayPointers(dev_MM_, 
                                             &dev_temp[i*sizePerSched]);
      // Copy to temp_schedules array to later copy to device with 1 cudaMemcpy
      memcpy(&temp_schedules[i], host_schedules[i], memSize);
    }
    // Allocate and Copy array of schedules to device
    // A device array of schedules
    InstSchedule *dev_schedules_arr;
    memSize = sizeof(InstSchedule) * NUMTHREADS;
    gpuErrchk(cudaMalloc(&dev_schedules_arr, memSize));
    // Copy schedules to device
    gpuErrchk(cudaMemcpy(dev_schedules_arr, temp_schedules, memSize,
                         cudaMemcpyHostToDevice));
    free(temp_schedules);
    // Create a dev array of pointers to dev_schedules_arr
    // Passing and array of schedules and dereferencing the array
    // to get pointers slows down the kernel significantly
    InstSchedule **dev_schedules;
    memSize = sizeof(InstSchedule *) * NUMTHREADS;
    gpuErrchk(cudaMallocManaged(&dev_schedules, memSize));
    for (int i = 0; i < NUMTHREADS; i++)
      dev_schedules[i] = &dev_schedules_arr[i];
    gpuErrchk(cudaMemPrefetchAsync(dev_schedules, memSize, 0));
    // Copy over best schedule
    // holds device copy of best sched, to be copied back to host after kernel
    InstSchedule *dev_bestSched;
    bestSchedule = new InstSchedule(machMdl_, dataDepGraph_, true);
    bestSchedule->Copy(InitialSchedule);
    bestSchedule->AllocateOnDevice(dev_MM_);
    bestSchedule->CopyArraysToDevice();
    memSize = sizeof(InstSchedule);
    gpuErrchk(cudaMalloc((void**)&dev_bestSched, memSize));
    gpuErrchk(cudaMemcpy(dev_bestSched, bestSchedule, memSize,
                         cudaMemcpyHostToDevice));
    // Create a global mem array for device to use in parallel reduction
    int *dev_blockBestIndex;
    memSize = (NUMBLOCKS/2) * sizeof(int);
    gpuErrchk(cudaMalloc(&dev_blockBestIndex, memSize));
    // Make sure managed memory is copied to device before kernel start
    memSize = sizeof(ACOScheduler);
    gpuErrchk(cudaMemPrefetchAsync(dev_AcoSchdulr, memSize, 0));
    Logger::Info("Launching Dev_ACO with %d blocks of %d threads", NUMBLOCKS,
                                                           NUMTHREADSPERBLOCK);
    // Using Cooperative Grid Groups requires launching with
    // cudaLaunchCooperativeKernel which requires kernel args to be an array
    // of void pointers to host memory locations of the arguments
    dim3 gridDim(NUMBLOCKS);
    dim3 blockDim(NUMTHREADSPERBLOCK);
    void *dArgs[8];
    dArgs[0] = (void*)&dev_rgn_;
    dArgs[1] = (void*)&dev_DDG_;
    dArgs[2] = (void*)&dev_AcoSchdulr;
    dArgs[3] = (void*)&dev_schedules;
    dArgs[4] = (void*)&dev_ready_;
    dArgs[5] = (void*)&dev_bestSched;
    dArgs[6] = (void*)&noImprovementMax;
    dArgs[7] = (void*)&dev_blockBestIndex;
    gpuErrchk(cudaLaunchCooperativeKernel((void*)Dev_ACO, gridDim, blockDim, 
                                          dArgs));
    cudaDeviceSynchronize();
    Logger::Info("Post Kernel Error: %s", 
                 cudaGetErrorString(cudaGetLastError()));
    // Copy dev_bestSched back to host
    memSize = sizeof(InstSchedule);
    gpuErrchk(cudaMemcpy(bestSchedule, dev_bestSched, memSize,
                         cudaMemcpyDeviceToHost));
    bestSchedule->CopyArraysToHost();
    // Free allocated memory that is no longer needed
    bestSchedule->FreeDeviceArrays();
    cudaFree(dev_bestSched);
    for (int i = 0; i < NUMTHREADS; i++) {
      delete host_schedules[i];
    }
    delete[] host_schedules;
    // delete the large array shared by all schedules
    cudaFree(dev_temp);
    cudaFree(dev_schedules);

  } else { // Run ACO on cpu
    Logger::Info("Running host ACO with %d ants per iteration", NUMTHREADS);
    InstCount RPTarget;
    if (!((BBWithSpill *)rgn_)->needsSLIL())
      RPTarget = bestSchedule->GetSpillCost();
    else
      RPTarget = MaxRPTarget;
    while (noImprovement < noImprovementMax) {
      iterationBest = nullptr;
      for (int i = 0; i < NUMTHREADS; i++) {
        InstSchedule *schedule = FindOneSchedule(RPTarget);
        if (print_aco_trace)
          PrintSchedule(schedule);
        if (shouldReplaceSchedule(iterationBest, schedule, false)) {
          if (iterationBest)
            delete iterationBest;          
          iterationBest = schedule;
        } else {
            if (schedule)
              delete schedule;
        }
      }
#if !USE_ACS
      if (iterationBest)
        UpdatePheromone(iterationBest);
#endif
      if (shouldReplaceSchedule(bestSchedule, iterationBest, true)) {
        if (bestSchedule && bestSchedule != InitialSchedule)
          delete bestSchedule;
        bestSchedule = std::move(iterationBest);
        if (!((BBWithSpill *)rgn_)->needsSLIL())
          RPTarget = bestSchedule->GetSpillCost();
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, sched length: %d, and "
               "iteration:%d\n",
               bestSchedule->GetCost(), bestSchedule->GetSpillCost(),
               bestSchedule->GetCrntLngth(), iterations);
#if !RUNTIME_TESTING
          noImprovement = 0;
#else
          // Disable resetting noImp to lock iterations to 10
          noImprovement++;
#endif
      } else {
        delete iterationBest;
        noImprovement++;
      }
#if USE_ACS
      UpdatePheromone(bestSchedule);
#endif
      iterations++;
    }
    Logger::Info("%d ants terminated early", numAntsTerminated_);
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
void ACOScheduler::UpdatePheromone(InstSchedule *schedule) {
#ifdef __CUDA_ARCH__ // device version of function
#if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
  // parallel on global level
  int instNum = GLOBALTID;
#elif (PHER_UPDATE_SCHEME == ALL || PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
  // parallel on block level
  int instNum = threadIdx.x;
#endif
  // Each thread updates pheromone table for 1 instruction
  // For the case NUMTHREADS < count_, increase instNum by 
  // NUMTHREADS at the end of the loop.
  InstCount lastInstNum = -1;
  thrust::maximum<double> dmax;
  thrust::minimum<double> dmin;
  pheromone_t portion = schedule->GetCost() / (ScRelMax * 1.5);
  pheromone_t deposition;
  if (portion < 1)
    deposition = (1 - portion) * MAX_DEPOSITION_MINUS_MIN + MIN_DEPOSITION;
  else
    deposition = MIN_DEPOSITION;

  pheromone_t *pheromone;
  while (instNum < count_) {
    // Get the instruction that comes before inst in the schedule
    // if instNum == count_ - 2 it has the root inst and lastInstNum = -1
    lastInstNum = schedule->GetPrevInstNum(instNum);
    // Get corresponding pheromone and update it
    pheromone = &Pheromone(lastInstNum, instNum);
    *pheromone = *pheromone + deposition;
    // decay pheromone for all trails leading to instNum
    for (int j = 0; j < count_; j++) {
      pheromone = &Pheromone(j, instNum);
      *pheromone *= (1 - decay_factor);
      *pheromone = dmax(1, dmin(8, *pheromone));
    }
#if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
    // parallel on global level
    // Increase instNum by NUMTHREADS until over count_
    instNum += NUMTHREADS;
#elif (PHER_UPDATE_SCHEME == ALL || PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
    // parallel on block level
    instNum += NUMTHREADSPERBLOCK;
#endif
  }
  if (print_aco_trace)
    PrintPheromone();

#else // host version of function
  // I wish InstSchedule allowed you to just iterate over it, but it's got this
  // cycle and slot thing which needs to be accounted for
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);

  SchedInstruction *lastInst = NULL;
  pheromone_t portion = schedule->GetCost() / (ScRelMax * 1.5);
  pheromone_t deposition =
      fmax((1 - portion) * MAX_DEPOSITION_MINUS_MIN, 0) + MIN_DEPOSITION;
  pheromone_t *pheromone;
  while (instNum != INVALID_VALUE) {  
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(instNum);

    pheromone = &Pheromone(lastInst, inst);
#if USE_ACS
    // ACS update rule includes decay
    // only the arcs on the current solution are decayed
    *pheromone = (1 - decay_factor) * *pheromone +
                 decay_factor / (schedule->GetCost() + 1);
#else
    *pheromone = *pheromone + deposition;
#endif
    lastInst = inst;

    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  schedule->ResetInstIter();

#if !USE_ACS
  // decay pheromone
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      pheromone = &Pheromone(j, instNum);
      *pheromone *= (1 - decay_factor);
      *pheromone = fmax(1, fmin(8, *pheromone));
    }
  }
#endif
  if (print_aco_trace)
    PrintPheromone();
#endif
}

__device__
void ACOScheduler::CopyPheromonesToSharedMem(double *s_pheromone) {
  InstCount toInstNum = threadIdx.x;
  while (toInstNum < count_) {
    for (int fromInstNum = -1; fromInstNum < count_; fromInstNum++)
      s_pheromone[((fromInstNum + 1) * count_) + toInstNum] = 
                                        Pheromone(fromInstNum, toInstNum);
    toInstNum += NUMTHREADSPERBLOCK;
  }
}

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
void ACOScheduler::PrintPheromone() {
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      //std::cerr << std::scientific << std::setprecision(8) << Pheromone(i, j)
      //          << " ";
      printf("%.10e ", Pheromone(i, j));
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

void ACOScheduler::CopyPheromonesToDevice(ACOScheduler *dev_AcoSchdulr) {
  size_t memSize;
  // Free allocated mem sinve pheromone size can change
  if (dev_AcoSchdulr->dev_pheromone_elmnts_alloced_ == true)
    cudaFree(dev_AcoSchdulr->pheromone_.elmnts_);

  memSize = sizeof(DeviceVector<pheromone_t>);
  gpuErrchk(cudaMemcpy(&dev_AcoSchdulr->pheromone_, &pheromone_, memSize,
            cudaMemcpyHostToDevice));

  memSize = sizeof(pheromone_t) * pheromone_.alloc_;
  gpuErrchk(cudaMalloc(&(dev_AcoSchdulr->pheromone_.elmnts_), memSize));
  gpuErrchk(cudaMemcpy(dev_AcoSchdulr->pheromone_.elmnts_, pheromone_.elmnts_,
		       memSize, cudaMemcpyHostToDevice));
  
  dev_AcoSchdulr->dev_pheromone_elmnts_alloced_ = true;
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
  // Create an array of PriorityArrayLists, allocate dev mem for it 
  // and elmnts_ for each one, and copy it to device
  PriorityArrayList<InstCount, InstCount> *temp = 
    new PriorityArrayList<InstCount, InstCount>[NUMTHREADS];
  // Allocate elmnts_ and keys_ for all PArrayLists
  InstCount *dev_elmnts, *dev_keys;
  memSize = sizeof(InstCount) * count_ * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_elmnts, memSize));
  gpuErrchk(cudaMalloc(&dev_keys, memSize));
  // set correct maxSize, elmnts_, and keys_ for each PArrayList
  for (int i = 0; i < NUMTHREADS; i++) {
    temp[i].maxSize_ = count_;
    temp[i].elmnts_ = &dev_elmnts[i * count_];
    temp[i].keys_ = &dev_keys[i * count_];
  }
  // Alloc dev mem and copy array of PArrayLists to device
  PriorityArrayList<InstCount, InstCount> *dev_array;
  memSize = sizeof(PriorityArrayList<InstCount, InstCount>) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_array, memSize));
  gpuErrchk(cudaMemcpy(dev_array, temp, memSize, cudaMemcpyHostToDevice));
  // set dev_instsWithPrdcsrsScheduld_ pointers to each PAL in array
  for (int i = 0; i < NUMTHREADS; i++)
    dev_instsWithPrdcsrsSchduld_[i] = &dev_array[i];
  // make sure host also has a copy of array for later deletion
  memSize = sizeof(PriorityArrayList<InstCount, InstCount>) * NUMTHREADS;
  gpuErrchk(cudaMemPrefetchAsync(dev_array, memSize, cudaCpuDeviceId));
  // remove references to dev arrays in host copy and delete host copy
  for (int i = 0; i < NUMTHREADS; i++) {
    temp[i].elmnts_ = NULL;
    temp[i].keys_ = NULL;
  }
  delete[] temp;

  // Copy rdyLst_
  memSize = sizeof(ReadyList);
  gpuErrchk(cudaMallocManaged(&dev_ACOSchedulr->dev_rdyLst_, memSize));
  gpuErrchk(cudaMemcpy(dev_ACOSchedulr->dev_rdyLst_, rdyLst_, memSize,
		       cudaMemcpyHostToDevice));
  rdyLst_->CopyPointersToDevice(dev_ACOSchedulr->dev_rdyLst_, dev_DDG_,
		                NUMTHREADS);
  // make sure cudaMallocManaged memory is copied to device before kernel start
  memSize = sizeof(int16_t *) * NUMTHREADS;
  gpuErrchk(cudaMemPrefetchAsync(dev_avlblSlotsInCrntCycle_, memSize, 0));
  memSize = sizeof(PriorityArrayList<InstCount, InstCount> *) * NUMTHREADS;
  gpuErrchk(cudaMemPrefetchAsync(dev_instsWithPrdcsrsSchduld_, memSize, 0));
  memSize = sizeof(ReserveSlot *) * NUMTHREADS;
  gpuErrchk(cudaMemPrefetchAsync(dev_rsrvSlots_, memSize, 0));
  memSize = sizeof(ReadyList);
  gpuErrchk(cudaMemPrefetchAsync(&dev_ACOSchedulr->dev_rdyLst_, memSize, 0));
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
    //cudaFree(dev_instsWithPrdcsrsSchduld_[i]->elmnts_);
    //cudaFree(dev_instsWithPrdcsrsSchduld_[i]->keys_);
    //cudaFree(dev_instsWithPrdcsrsSchduld_[i]);
    cudaFree(dev_avlblSlotsInCrntCycle_[i]);
    cudaFree(dev_rsrvSlots_[i]);
  }
  cudaFree(dev_instsWithPrdcsrsSchduld_[0]->elmnts_);
  cudaFree(dev_instsWithPrdcsrsSchduld_[0]->keys_);
  cudaFree(dev_instsWithPrdcsrsSchduld_[0]);
  dev_rdyLst_->FreeDevicePointers(NUMTHREADS);
  cudaFree(dev_avlblSlotsInCrntCycle_);
  cudaFree(dev_rsrvSlots_);
  cudaFree(dev_rsrvSlotCnt_);
  cudaFree(dev_instsWithPrdcsrsSchduld_);
  cudaFree(dev_rdyLst_);
  cudaFree(pheromone_.elmnts_);
}
