#include "hip/hip_runtime.h"
#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/dev_defines.h"
// #include <thrust/functional.h>
#include <hip/hip_cooperative_groups.h>
#include "llvm/ADT/STLExtras.h"
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <sstream>
#include <hiprand/hiprand_kernel.h>

using namespace llvm::opt_sched;
namespace cg = cooperative_groups;

#ifndef NDEBUG
static void PrintInstruction(SchedInstruction *inst);
#endif
__host__ __device__
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
//#define CHECK_DIFFERENT_SCHEDULES 1
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
                           bool IsPostBB, int numBlocks,
                           SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
			                     MachineModel *dev_MM, void *dev_states)
    : ConstrainedScheduler(dataDepGraph, machineModel, upperBound, true) {
  VrfySched_ = vrfySched;
  this->IsPostBB = IsPostBB;
  prirts_ = priorities;
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();
  dev_rgn_ = dev_rgn;
  dev_DDG_ = dev_DDG;
  dev_MM_ = dev_MM;
  dev_states_ = dev_states;
  dev_pheromone_elmnts_alloced_ = false;
  numAntsTerminated_ = 0;
  numBlocks_ = numBlocks;
  numThreads_ = numBlocks_ * NUMTHREADSPERBLOCK;

  use_dev_ACO = schedIni.GetBool("DEV_ACO");
  if(!use_dev_ACO || count_ < REGION_MIN_SIZE)
    numThreads_ = schedIni.GetInt("HOST_ANTS");
  else {
    dev_rgn_->SetNumThreads(numThreads_);
    dev_DDG_->SetNumThreads(numThreads_);
  }

  use_fixed_bias = schedIni.GetBool("ACO_USE_FIXED_BIAS");
  use_tournament = schedIni.GetBool("ACO_TOURNAMENT");
  bias_ratio = schedIni.GetFloat("ACO_BIAS_RATIO");
  local_decay = schedIni.GetFloat("ACO_LOCAL_DECAY");
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

  //construct the ACOReadyList member and a key helper
  readyLs = new ACOReadyList(dataDepGraph->GetMaxIndependentInstructions());
  kHelper = new KeysHelper(priorities);
  kHelper->initForRegion(dataDepGraph);

  InitialSchedule = nullptr;
}

ACOScheduler::~ACOScheduler() {
  if (readyLs)
    delete readyLs;
  if (kHelper)
    delete kHelper;
}

__device__
hiprandState_t *getDevRandStates(ACOScheduler *ACOSchedulr) {
  return (hiprandState_t *) (ACOSchedulr->dev_states_);
}

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
pheromone_t ACOScheduler::Score(InstCount FromId, InstCount ToId, HeurType ToHeuristic) {
  // tuneable heuristic importance is temporarily disabled
  // double Hf = pow(ToHeuristic, heuristicImportance_);
  pheromone_t HeurScore = ToHeuristic * MaxPriorityInv + 1;
  pheromone_t Hf = heuristicImportance_ ? HeurScore : 1.0;
  return Pheromone(FromId, ToId) * Hf;
}

__host__ __device__
bool ACOScheduler::shouldReplaceSchedule(InstSchedule *OldSched,
                                         InstSchedule *NewSched,
                                         bool IsGlobal, InstCount RPTarget) {
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
#ifdef __HIP_DEVICE_COMPILE__
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
    #ifdef __HIP_DEVICE_COMPILE__
      bool needsSLIL = ((BBWithSpill *)dev_rgn_)->needsSLIL();
      bool needsTarget = ((BBWithSpill *)dev_rgn_)->needsTarget();
    #else
      bool needsSLIL = ((BBWithSpill *)rgn_)->needsSLIL();
      bool needsTarget = ((BBWithSpill *)rgn_)->needsTarget();
    #endif
    InstCount NewCost = NewSched->GetExecCost();
    InstCount OldCost = OldSched->GetExecCost();
    InstCount NewSpillCost = NewSched->GetNormSpillCost();
    InstCount OldSpillCost = OldSched->GetNormSpillCost();

    if (needsTarget) {
      // If both schedules are under occupancy target, then pick shorter one
      if (NewSpillCost <= RPTarget && OldSpillCost <= RPTarget) {
        if (NewCost < OldCost)
          return true;
        else
          return false;
      }
    }
    // if using SLIL and old schedule is 0 PERP, new schedule wins if it
    // is 0 PERP and shorter
    else if (needsSLIL && OldSched->getIsZeroPerp()) {
      if (NewSched->getIsZeroPerp() && NewCost < OldCost)
        return true;
      else if (NewSched->getIsZeroPerp() && NewCost == OldCost && NewSpillCost < OldSpillCost)
        return true;
      else
        return false;
    }
    // if old schedule is not 0 PERP and new schedule is 0 PERP,
    // it wins regardless of schedule length
    else if (needsSLIL && NewSched->getIsZeroPerp()) {
      return true;
    }
    // Otherwise, Lower Spill Cost always wins
    else if (NewSpillCost < OldSpillCost)
      return true;
    else if (NewSpillCost == OldSpillCost && NewCost < OldCost)
      return true;
    else
      return false;
  }
}

__host__ __device__
InstCount ACOScheduler::SelectInstruction(SchedInstruction *lastInst, InstCount totalStalls,
                                          SchedRegion *rgn, bool &unnecessarilyStalling,
                                          bool closeToRPTarget, bool currentlyWaiting) {
#ifdef __HIP_DEVICE_COMPILE__
  #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("Crash Beginning of SelectInstruction()\n");
    }
  #endif
  // if we are waiting and have no fully-ready instruction that is 
  // net 0 or benefit to RP, then return -1 to schedule a stall
  if (currentlyWaiting && dev_RP0OrPositiveCount[GLOBALTID] == 0)
    return -1;

  // calculate MaxScoringInst, and ScoreSum
  pheromone_t MaxScore = -1;
  InstCount MaxScoreIndx = 0;
  dev_readyLs->ScoreSum = 0;
  int lastInstId = lastInst->GetNum();
  // this bool is to check if stalling could be avoided
  bool couldAvoidStalling = false;
  // this bool is to check if we should currently avoid unnecessary stalls
  // because RP is low or we have too many stalls in the schedule
  bool RPIsHigh = false;
  bool tooManyStalls = totalStalls >= globalBestStalls_ * 9 / 10;
  dev_readyLs->dev_ScoreSum[GLOBALTID] = 0;

  for (InstCount I = 0; I < dev_readyLs->getReadyListSize(); ++I) {
    RPIsHigh = false;
    InstCount CandidateId = *dev_readyLs->getInstIdAtIndex(I);
    SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
    HeurType candidateLUC = candidateInst->GetLastUseCnt();
    int16_t candidateDefs = candidateInst->GetDefCnt();

    // compute the score
    HeurType Heur = *dev_readyLs->getInstHeuristicAtIndex(I);
    pheromone_t IScore = Score(lastInstId, *dev_readyLs->getInstIdAtIndex(I), Heur);
    if (dev_RP0OrPositiveCount[GLOBALTID] != 0 && candidateDefs > candidateLUC)
      IScore = IScore * 9/10;

    #ifdef DEBUG_INSTR_SELECTION
    if (GLOBALTID==0)
      printf("Before Inst: %d, score: %f\n", *dev_readyLs->getInstIdAtIndex(I), IScore);
    #endif

    if (currentlyWaiting) {
      // if currently waiting on an instruction, do not consider semi-ready instructions 
      if (*dev_readyLs->getInstReadyOnAtIndex(I) > dev_crntCycleNum_[GLOBALTID])
        continue;

      // as well as instructions with a net negative impact on RP
      if (candidateDefs > candidateLUC)
        continue;
    }

    // add a score penalty for instructions that are not ready yet
    // unnecessary stalls should not be considered if current RP is low, or if we already have too many stalls
    if (*dev_readyLs->getInstReadyOnAtIndex(I) > dev_crntCycleNum_[GLOBALTID]) {
      if (dev_RP0OrPositiveCount[GLOBALTID] != 0) {
        #ifdef DEBUG_INSTR_SELECTION
        if (GLOBALTID==0)
          printf("Zeroing out Inst: %d, score: %f, only rp negative: %s, close to RP Target: %s\n", *dev_readyLs->getInstIdAtIndex(I), IScore, dev_RP0OrPositiveCount[GLOBALTID] ? "true" : "false", closeToRPTarget ? "true" : "false");
        #endif
        IScore = 0.0000001;
      }
      else {
        int cyclesNeededToWait = *dev_readyLs->getInstReadyOnAtIndex(I) - dev_crntCycleNum_[GLOBALTID];
        if (cyclesNeededToWait < globalBestStalls_)
          IScore = IScore * (globalBestStalls_ - cyclesNeededToWait * 2) / globalBestStalls_;
        else 
          IScore = IScore / globalBestStalls_;

        // check if any reg types used by the instructions are above the physical register limit
        SchedInstruction *tempInst = dataDepGraph_->GetInstByIndx(*dev_readyLs->getInstIdAtIndex(I));
        // TODO(bruce): convert to dev uses
        RegIndxTuple *uses;
        Register *use;
        uint16_t usesCount = tempInst->GetUseCnt();
        int useStart = tempInst->ddgUseIndex;
        for (uint16_t i = 0; i < usesCount; i++) {
          use = dataDepGraph_->getRegByTuple(dataDepGraph_->getUseByIndex(useStart + i));
          int16_t regType = use->GetType();
          if ( ((BBWithSpill *)rgn)->IsRPHigh(regType) ) {
            RPIsHigh = true;
            break;
          }
        }

        // reduce likelihood of selecting an instruction we have to wait for IF
        // RP is low or we have too many stalls already
        if (!(closeToRPTarget && RPIsHigh) || tooManyStalls) {
          if (globalBestStalls_ > totalStalls)
            IScore = IScore * (globalBestStalls_ - totalStalls * 2) / globalBestStalls_;
          else
            IScore = IScore / globalBestStalls_;
        }
      }
    }
    else {
      couldAvoidStalling = true;
    }

    if (IScore < 0.0000001)
      IScore = 0.0000001;
    #ifdef DEBUG_INSTR_SELECTION
    if (GLOBALTID==0)
      printf("After Inst: %d, score: %f, cycles to wait: %d\n", *dev_readyLs->getInstIdAtIndex(I), IScore, *dev_readyLs->getInstReadyOnAtIndex(I) - dev_crntCycleNum_[GLOBALTID]);
    #endif
    *dev_readyLs->getInstScoreAtIndex(I) = IScore;
    dev_readyLs->dev_ScoreSum[GLOBALTID] += IScore;
    
    if(IScore > MaxScore) {
      MaxScoreIndx = I;
      MaxScore = IScore;
    }
  }
#else
  // if we are waiting and have no fully-ready instruction that is 
  // net 0 or benefit to RP, then return -1 to schedule a stall
  if (currentlyWaiting && RP0OrPositiveCount == 0)
    return -1;

  // calculate MaxScoringInst, and ScoreSum
  pheromone_t MaxScore = -1;
  InstCount MaxScoreIndx = 0;
  readyLs->ScoreSum = 0;
  int lastInstId = lastInst->GetNum();
  // this bool is to check if stalling could be avoided
  bool couldAvoidStalling = false;
  // this bool is to check if we should currently avoid unnecessary stalls
  // because RP is low or we have too many stalls in the schedule
  bool RPIsHigh = false;
  bool tooManyStalls = totalStalls >= globalBestStalls_ * 5 / 10;
  readyLs->ScoreSum = 0;

  for (InstCount I = 0; I < readyLs->getReadyListSize(); ++I) {
    RPIsHigh = false;
    InstCount CandidateId = *readyLs->getInstIdAtIndex(I);
    SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
    HeurType candidateLUC = candidateInst->GetLastUseCnt();
    int16_t candidateDefs = candidateInst->GetDefCnt();

    // compute the score
    HeurType Heur = *readyLs->getInstHeuristicAtIndex(I);
    pheromone_t IScore = Score(lastInstId, *readyLs->getInstIdAtIndex(I), Heur);
    if (RP0OrPositiveCount != 0 && candidateDefs > candidateLUC)
      IScore = IScore * 9/10;

    if (currentlyWaiting) {
      // if currently waiting on an instruction, do not consider semi-ready instructions 
      if (*readyLs->getInstReadyOnAtIndex(I) > crntCycleNum_)
        continue;

      // as well as instructions with a net negative impact on RP
      if (candidateDefs > candidateLUC)
        continue;
    }
    
    // add a score penalty for instructions that are not ready yet
    // unnecessary stalls should not be considered if current RP is low, or if we already have too many stalls
    if (*readyLs->getInstReadyOnAtIndex(I) > crntCycleNum_) {
      if (RP0OrPositiveCount != 0) {
        IScore = 0.0000001;
      }
      else {
        int cyclesNeededToWait = *readyLs->getInstReadyOnAtIndex(I) - crntCycleNum_;
        if (cyclesNeededToWait < globalBestStalls_)
          IScore = IScore * (globalBestStalls_ - cyclesNeededToWait * 2) / globalBestStalls_;
        else 
          IScore = IScore / globalBestStalls_;

        // check if any reg types used by the instructions are above the physical limit
        SchedInstruction *tempInst = dataDepGraph_->GetInstByIndx(*readyLs->getInstIdAtIndex(I));
        RegIndxTuple *uses;
        Register *use;
        uint16_t usesCount = tempInst->GetUses(uses);
        for (uint16_t i = 0; i < usesCount; i++) {
          use = dataDepGraph_->getRegByTuple(&uses[i]);
          int16_t regType = use->GetType();
          if ( ((BBWithSpill *)rgn)->IsRPHigh(regType) ) {
            RPIsHigh = true;
            break;
          }
        }

        // reduce likelihood of selecting an instruction we have to wait for IF
        // RP is low or we have too many stalls already
        if (!(closeToRPTarget && RPIsHigh) || tooManyStalls) {
          if (globalBestStalls_ > totalStalls)
            IScore = IScore * (globalBestStalls_ - totalStalls * 2) / globalBestStalls_;
          else
            IScore = IScore / globalBestStalls_;
        }
      }
    }
    else {
      couldAvoidStalling = true;
    }

    if (IScore < 0.0000001)
      IScore = 0.0000001;
    *readyLs->getInstScoreAtIndex(I) = IScore;
    readyLs->ScoreSum += IScore;
    
    if(IScore > MaxScore) {
      MaxScoreIndx = I;
      MaxScore = IScore;
    }
  }
#endif

  //generate the random numbers that we will need for deciding if
  //we are going to use the fixed bias or if we are going to use
  //fitness proportional selection.  Generate the number used for
  //the fitness proportional selection point
  double rand;
  pheromone_t point;
#ifdef __HIP_DEVICE_COMPILE__
  auto dev_states = getDevRandStates(this);
  rand = hiprand_uniform(&dev_states[GLOBALTID]);
  point = dev_readyLs->dev_ScoreSum[GLOBALTID] * hiprand_uniform(&dev_states[GLOBALTID]);
#else
  rand = RandDouble(0, 1);
  point = RandDouble(0, readyLs->ScoreSum);
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

  // select the instruction index for fp choice
  size_t fpIndx=0;
  #ifdef __HIP_DEVICE_COMPILE__
    __shared__ bool dev_useMax;
    // select useMax for each block
    if (hipThreadIdx_x == 0)
      dev_useMax = (rand < choose_best_chance) || currentlyWaiting;
    __syncthreads();
    #ifdef DEBUG_EXPLORATION_EXPLOITATION_TOGETHER
      if (GLOBALTID < 5 )
        printf("GLOBALTID %d rand: %f, choose_best_chance: %f, useMax: %s\n", GLOBALTID, rand, choose_best_chance, dev_useMax ? "true" : "false");
    #endif
    if (!dev_useMax) {
      for (size_t i = 0; i < dev_readyLs->getReadyListSize(); ++i) {
        point -= *dev_readyLs->getInstScoreAtIndex(i);
        if (point <= 0) {
          fpIndx = i;
          break;
        }
      }
    }
  #else
    for (size_t i = 0; i < readyLs->getReadyListSize(); ++i) {
      point -= *readyLs->getInstScoreAtIndex(i);
      if (point <= 0) {
        fpIndx = i;
        break;
      }
    }
  #endif
  //finally we pick whether we will return the fp choice or max score inst w/o using a branch
  #ifdef __HIP_DEVICE_COMPILE__
    size_t indx = dev_useMax ? MaxScoreIndx : fpIndx;
  #else
    bool UseMax = (rand < choose_best_chance) || currentlyWaiting;
    size_t indx = UseMax ? MaxScoreIndx : fpIndx;
  #endif
  #ifdef __HIP_DEVICE_COMPILE__
    #ifdef DEBUG_INSTR_SELECTION
    if (GLOBALTID==0) {
      printf("Selecting: %d\n", *dev_readyLs->getInstIdAtIndex(indx));
    }
    #endif
    if (couldAvoidStalling && *dev_readyLs->getInstReadyOnAtIndex(indx) > dev_crntCycleNum_[GLOBALTID])
      unnecessarilyStalling = true;
    else
      unnecessarilyStalling = false;
  #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("End of SelectInstruction()\n");
    }
  #endif
  #else
    if (couldAvoidStalling && *readyLs->getInstReadyOnAtIndex(indx) > crntCycleNum_)
      unnecessarilyStalling = true;
    else
      unnecessarilyStalling = false;
  #endif
  return indx;
}

__host__ __device__
InstSchedule *ACOScheduler::FindOneSchedule(InstCount RPTarget, 
                                            InstSchedule *dev_schedule) {
#ifdef __HIP_DEVICE_COMPILE__ // device version of function
  SchedInstruction *inst = NULL;
  SchedInstruction *lastInst = NULL;
  ACOReadyListEntry LastInstInfo;
  InstSchedule *schedule = dev_schedule;
  schedule->SetNumThreads(numThreads_);
  bool IsSecondPass = dev_rgn_->IsSecondPass();
  dev_readyLs->clearReadyList();
  ScRelMax = dev_rgn_->GetHeuristicCost();
  bool unnecessarilyStalling = false;

  // The MaxPriority that we are getting from the ready list represents
  // the maximum possible heuristic/key value that we can have
  HeurType MaxPriority = dev_kHelper->getMaxValue();
  if (MaxPriority == 0)
    MaxPriority = 1; // divide by 0 is bad
  Initialize_();

  SchedInstruction *waitFor = NULL;
  InstCount waitUntil = 0;
  MaxPriorityInv = 1 / (pheromone_t)MaxPriority;

  // initialize the aco ready list so that the start instruction is ready
  // The luc component is 0 since the root inst uses no instructions
  InstCount RootId = rootInst_->GetNum();
  HeurType RootHeuristic = dev_kHelper->computeKey(rootInst_, true, dev_DDG_->RegFiles, dev_DDG_);
  pheromone_t RootScore = Score(-1, RootId, RootHeuristic);
  ACOReadyListEntry InitialRoot{RootId, 0, RootHeuristic, RootScore};
  dev_readyLs->addInstructionToReadyList(InitialRoot);
  dev_readyLs->dev_ScoreSum[GLOBALTID] = RootScore;
  dev_MaxScoringInst[GLOBALTID] = 0;
  lastInst = dataDepGraph_->GetInstByIndx(RootId);
  bool closeToRPTarget = false;
  dev_RP0OrPositiveCount[GLOBALTID] = 0;
  #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("Crash before while loop inside FindOneSchedule()\n");
    }
  #endif
  while (!IsSchedComplete_()) {
    // incrementally calculate if there are any instructions with a neutral
    // or positive effect on RP
    for (InstCount I = 0; I < dev_readyLs->getReadyListSize(); ++I) {
      if (*dev_readyLs->getInstReadyOnAtIndex(I) == dev_crntCycleNum_[GLOBALTID]) {
        InstCount CandidateId = *dev_readyLs->getInstIdAtIndex(I);
        SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType candidateLUC = candidateInst->GetLastUseCnt();
        int16_t candidateDefs = candidateInst->GetDefCnt();
        if (candidateDefs <= candidateLUC) {
          dev_RP0OrPositiveCount[GLOBALTID] = dev_RP0OrPositiveCount[GLOBALTID] + 1;
        }
      }
    }

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    if (!waitFor && waitUntil <= dev_crntCycleNum_[GLOBALTID]) {
      assert(dev_readyLs->getReadyListSize() > 0 || waitFor != NULL);

      if ( !((BBWithSpill *)dev_rgn_)->needsTarget() ) {
        InstCount closeToRPCheck = RPTarget - 2 < RPTarget * 9 / 10 ? RPTarget - 2 : RPTarget * 9 / 10;
        closeToRPTarget = ((BBWithSpill *)dev_rgn_)->GetCrntSpillCost() >= closeToRPCheck;
      }
      else
        closeToRPTarget = ((BBWithSpill *)dev_rgn_)->closeToRPConstraint();
      
      #ifdef DEBUG_CLOSE_TO_OCCUPANCY
      if (GLOBALTID == 0) {
        printf("cyclenum: %d Close to RP Target: %s\n", dev_crntCycleNum_[GLOBALTID], closeToRPTarget ? "true" : "false");
      }
      #endif
      // select the instruction and get info on it
      InstCount SelIndx = SelectInstruction(lastInst, schedule->getTotalStalls(), dev_rgn_, unnecessarilyStalling, closeToRPTarget, waitFor ? true: false);
      #ifdef DEBUG_ACO_CRASH_LOCATIONS
        if (hipThreadIdx_x == 0) {
          printf("After SelectInstruction()\n");
        }
      #endif
      if (SelIndx != -1) {
        LastInstInfo = dev_readyLs->removeInstructionAtIndex(SelIndx);
        
        InstCount InstId = LastInstInfo.InstId;
        inst = dataDepGraph_->GetInstByIndx(InstId);
        #ifdef DEBUG_ACO_CRASH_LOCATIONS
          if (hipThreadIdx_x == 0) {
            printf("After Test Print()\n");
          }
        #endif
        // potentially wait on the current instruction
        if (LastInstInfo.ReadyOn > crntCycleNum_ || !ChkInstLglty_(inst)) {
          waitUntil = LastInstInfo.ReadyOn;
          // should not wait for an instruction while already
          // waiting for another instruction
          assert(waitFor == NULL);
          waitFor = inst;
          inst = NULL;
        }

        if (inst != NULL) {
#if USE_ACS
          // local pheromone decay
          pheromone_t *pheromone = &Pheromone(lastInst, inst);
          *pheromone = 
            (1 - local_decay) * *pheromone + local_decay * initialValue_;
#endif
          // save the last instruction scheduled
          lastInst = inst;
        }
      }
    }

    // 2)Schedule a stall if we are still waiting, Schedule the instruction we
    // are waiting for if possible, decrement waiting time
    if (waitFor && waitUntil <= dev_crntCycleNum_[GLOBALTID]) {
      if (ChkInstLglty_(waitFor)) {
        inst = waitFor;
        waitFor = NULL;
        lastInst = inst;
      }
    }

    // boilerplate, mostly copied from ListScheduler, try not to touch it
    InstCount instNum;
    if (!inst) {
      instNum = SCHD_STALL;
      #ifdef DEBUG_INSTR_SELECTION
      if (GLOBALTID==0)
        printf("Scheduling stall\n");
      #endif
      schedule->incrementTotalStalls();
      if (unnecessarilyStalling)
        schedule->incrementUnnecessaryStalls();
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
        dev_readyLs->clearReadyList();
        // end schedule construction
        return NULL;
      } 
      DoRsrvSlots_(inst);
      // this is annoying
      UpdtSlotAvlblty_(inst);
      #ifdef DEBUG_ACO_CRASH_LOCATIONS
        if (hipThreadIdx_x == 0) {
          printf("Before UpdateACOReadyList()\n");
        }
      #endif
      // new readylist update
      UpdateACOReadyList(inst);
      #ifdef DEBUG_ACO_CRASH_LOCATIONS
        if (hipThreadIdx_x == 0) {
          printf("After UpdateACOReadyList()\n");
        }
      #endif
    }
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
  }
  #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("After while loop inside FindOneSchedule()\n");
    }
  #endif
  dev_rgn_->UpdateScheduleCost(schedule);
  schedule->setIsZeroPerp( ((BBWithSpill *)dev_rgn_)->ReturnPeakSpillCost() == 0 );
  return schedule;

#else  // **** Host version of function ****
  SchedInstruction *lastInst = NULL;
  ACOReadyListEntry LastInstInfo;
  InstSchedule *schedule;
  schedule = new InstSchedule(machMdl_, dataDepGraph_, true);
  bool IsSecondPass = rgn_->IsSecondPass();
  bool unnecessarilyStalling = false;
  // The MaxPriority that we are getting from the ready list represents the maximum possible heuristic/key value that we can have
  // I want to move all the heuristic computation stuff to another class for code tidiness reasons.
  HeurType MaxPriority = kHelper->getMaxValue();
  if (MaxPriority == 0)
    MaxPriority = 1; // divide by 0 is bad
  Initialize_();

  SchedInstruction *waitFor = NULL;
  InstCount waitUntil = 0;
  MaxPriorityInv = 1 / (pheromone_t)MaxPriority;

  // initialize the aco ready list so that the start instruction is ready
  // The luc component is 0 since the root inst uses no instructions
  InstCount RootId = rootInst_->GetNum();
  HeurType RootHeuristic = kHelper->computeKey(rootInst_, true, dataDepGraph_->RegFiles);
  pheromone_t RootScore = Score(-1, RootId, RootHeuristic);
  ACOReadyListEntry InitialRoot{RootId, 0, RootHeuristic, RootScore};
  readyLs->addInstructionToReadyList(InitialRoot);
  readyLs->ScoreSum = RootScore;
  MaxScoringInst = 0;
  lastInst = dataDepGraph_->GetInstByIndx(RootId);
  bool closeToRPTarget = false;
  RP0OrPositiveCount = 0;

  SchedInstruction *inst = NULL;
  while (!IsSchedComplete_()) {
    // incrementally calculate if there are any instructions with a neutral
    // or positive effect on RP
    for (InstCount I = 0; I < readyLs->getReadyListSize(); ++I) {
      if (*readyLs->getInstReadyOnAtIndex(I) == crntCycleNum_) {
        InstCount CandidateId = *readyLs->getInstIdAtIndex(I);
        SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType candidateLUC = candidateInst->GetLastUseCnt();
        int16_t candidateDefs = candidateInst->GetDefCnt();
        if (candidateDefs <= candidateLUC) {
          RP0OrPositiveCount = RP0OrPositiveCount + 1;
        }
      }
    }

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    inst = NULL;
    if (!(waitFor && waitUntil <= crntCycleNum_)) {
      // If an instruction is ready select it
      assert(readyLs->getReadyListSize() > 0  || waitFor != NULL); // we should always have something in the rl

      InstCount closeToRPCheck = RPTarget - 2 < RPTarget * 9 / 10 ? RPTarget - 2 : RPTarget * 9 / 10;
      closeToRPTarget = ((BBWithSpill *)rgn_)->GetCrntSpillCost() >= closeToRPCheck;
      // select the instruction and get info on it
      InstCount SelIndx = SelectInstruction(lastInst, schedule->getTotalStalls(), rgn_, unnecessarilyStalling, closeToRPTarget, waitFor ? true: false);

      if (SelIndx != -1) {
        LastInstInfo = readyLs->removeInstructionAtIndex(SelIndx);
        
        InstCount InstId = LastInstInfo.InstId;
        inst = dataDepGraph_->GetInstByIndx(InstId);
        // potentially wait on the current instruction
        if (LastInstInfo.ReadyOn > crntCycleNum_ || !ChkInstLglty_(inst)) {
          waitUntil = LastInstInfo.ReadyOn;
          // should not wait for an instruction while already
          // waiting for another instruction
          assert(waitFor == NULL);
          waitFor = inst;
          inst = NULL;
        }

        if (inst != NULL) {
  #if USE_ACS
          // local pheromone decay
          pheromone_t *pheromone = &Pheromone(lastInst, inst);
          *pheromone = (1 - local_decay) * *pheromone + local_decay * initialValue_;
  #endif
          // save the last instruction scheduled
          lastInst = inst;
        }
      }
    }

    // 2)Schedule a stall if we are still waiting, Schedule the instruction we
    // are waiting for if possible, decrement waiting time
    if (waitFor && waitUntil <= crntCycleNum_) {
      if (ChkInstLglty_(waitFor)) {
        inst = waitFor;
        waitFor = NULL;
        lastInst = inst;
      }
    }

    // boilerplate, mostly copied from ListScheduler, try not to touch it
    InstCount instNum;
    if (!inst) {
      instNum = SCHD_STALL;
      schedule->incrementTotalStalls();
      if (unnecessarilyStalling)
        schedule->incrementUnnecessaryStalls();
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
        readyLs->clearReadyList();
        delete schedule;
        return NULL;
      }
      DoRsrvSlots_(inst);
      // this is annoying
      UpdtSlotAvlblty_(inst);

      // new readylist update
      UpdateACOReadyList(inst);
    }
    /* Logger::Info("Chose instruction %d (for some reason)", instNum); */
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
  }
  rgn_->UpdateScheduleCost(schedule);
  schedule->setIsZeroPerp( ((BBWithSpill *)rgn_)->ReturnPeakSpillCost() == 0 );
  return schedule;
#endif
}

// Reduce to only index of best schedule per 2 blocks in output array
__inline__ __device__
void reduceToBestSchedPerBlock(InstSchedule **dev_schedules, int *blockBestIndex, ACOScheduler *dev_AcoSchdulr, InstCount RPTarget) {
  __shared__ int sdata[NUMTHREADSPERBLOCK];
  uint gtid = GLOBALTID;
  uint tid = hipThreadIdx_x;
  int blockSize = NUMTHREADSPERBLOCK;
  
  // load candidate schedules into smem
  if (dev_AcoSchdulr->shouldReplaceSchedule(dev_schedules[gtid * 2], dev_schedules[gtid * 2 + 1], false, RPTarget))
    sdata[tid] = gtid * 2 + 1;
  else
    sdata[tid] = gtid * 2;
  __syncthreads();

  // do reduction on indexes in shared mem
  for (uint s = 1; s < hipBlockDim_x; s*=2) {
    if (tid%(2*s) == 0) {
      if (dev_AcoSchdulr->shouldReplaceSchedule(
          dev_schedules[sdata[tid]], 
          dev_schedules[sdata[tid + s]], false, RPTarget))
        sdata[tid] = sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    blockBestIndex[hipBlockIdx_x] = sdata[0];

}

// 1 block only to allow proper synchronization
// reduce to only one best index. At the end of this function globalBestIndex
// should be in blockBestIndex[0]
__inline__ __device__
void reduceToBestSched(InstSchedule **dev_schedules, int *blockBestIndex,
                       ACOScheduler *dev_AcoSchdulr, int numBlocks, InstCount RPTarget) {
  __shared__ int sBestIndex[NUMBLOCKSMANYANTS/4];
  uint tid = hipThreadIdx_x;
  int index, sBestIndex1, sBestIndex2;
  
  // Load best indices into shared mem, reduce by half while doing so
  // If there are more than 64 schedules in blockBestIndex, some threads
  // will have to load in more than one value
  while (tid < numBlocks/4) {
    if (dev_AcoSchdulr->shouldReplaceSchedule(dev_schedules[blockBestIndex[tid * 2]], 
                                              dev_schedules[blockBestIndex[tid * 2 + 1]], false, RPTarget))
      sBestIndex[tid] = blockBestIndex[tid * 2 + 1];
    else
      sBestIndex[tid] = blockBestIndex[tid * 2];

    tid += hipBlockDim_x;
  }
  __syncthreads();

  // reduce in smem
  for (uint s = 1; s < numBlocks/4; s *= 2) {
    tid = hipThreadIdx_x;
    // if there are more than 32 schedules in smem, a thread
    // may reduce more than once per loop
    while (tid < numBlocks/4) {
      index = 2 * s * tid;

      if (index + s < numBlocks/4) {
        sBestIndex1 = sBestIndex[index];
        sBestIndex2 = sBestIndex[index + s];
        if (dev_AcoSchdulr->shouldReplaceSchedule(
            dev_schedules[sBestIndex1],
            dev_schedules[sBestIndex2], false, RPTarget))
          sBestIndex[index] = sBestIndex2;
      }
      tid += hipBlockDim_x;
    }
    __syncthreads();
  }
  if (hipThreadIdx_x == 0)
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

__device__ int globalBestIndex, dev_noImprovement, dev_schedsUsed, dev_schedsFound;
__device__ pheromone_t dev_totalPherInTable;
__device__ bool lowerBoundSchedFound, isGlobalBest;

__global__ void
__launch_bounds__(NUMTHREADSPERBLOCK, 1)
Dev_ACO(SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
            ACOScheduler *dev_AcoSchdulr, InstSchedule **dev_schedules,
            InstSchedule *dev_bestSched, int noImprovementMax, 
            int *blockBestIndex) {
  #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("Crash very beginning\n");
    }
  #endif
  // holds cost and index of bestSched per block
  __shared__ int bestIndex;
  int dev_iterations;
  bool needsSLIL;
  needsSLIL = ((BBWithSpill *)dev_rgn)->needsSLIL();
  bool needsTarget = ((BBWithSpill *)dev_rgn)->needsTarget();
  bool IsSecondPass = dev_rgn->IsSecondPass();
  dev_rgn->SetDepGraph(dev_DDG);
  ((BBWithSpill *)dev_rgn)->SetRegFiles(dev_DDG->getRegFiles());
  dev_noImprovement = 0;
  dev_iterations = 0;
  lowerBoundSchedFound = false;
  isGlobalBest = false;
  // Used to synchronize all launched threads
  auto threadGroup = cg::this_grid();
  // Get RPTarget
  InstCount RPTarget;
  dev_schedsUsed = 0;
  dev_schedsFound = 0;

  // If in second pass and not using SLIL, set RPTarget
  if (!needsSLIL)
    RPTarget = dev_bestSched->GetSpillCost();
  else
    RPTarget = INT_MAX;

  #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("Crash before while loop\n");
    }
  #endif
  // Start ACO
  while (dev_noImprovement < noImprovementMax && !lowerBoundSchedFound) {
    // Reset schedules to post constructor state
    dev_schedules[GLOBALTID]->Initialize();
    #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("Before FindOneSchedule() loop\n");
    }
    #endif
    dev_AcoSchdulr->FindOneSchedule(RPTarget,
                                    dev_schedules[GLOBALTID]);
    // Sync threads after schedule creation
    threadGroup.sync();
    #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("After FindOneSchedule()\n");
    }
    #endif
    globalBestIndex = INVALID_VALUE;
    // reduce dev_schedules to 1 best schedule per block
    if (GLOBALTID < dev_AcoSchdulr->GetNumThreads()/2)
      reduceToBestSchedPerBlock(dev_schedules, blockBestIndex, dev_AcoSchdulr, RPTarget);

    threadGroup.sync();
    #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("After reduceToBestSchedPerBlock() loop\n");
    }
    #endif
    // one block to reduce blockBest schedules to one best schedule
    if (hipBlockIdx_x == 0)
      reduceToBestSched(dev_schedules, blockBestIndex, dev_AcoSchdulr, dev_AcoSchdulr->GetNumBlocks(), RPTarget);

    threadGroup.sync();

    #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("After reduceToBestSched() loop\n");
    }
    #endif
    if (GLOBALTID == 0 && 
        dev_schedules[blockBestIndex[0]]->GetCost() != INVALID_VALUE)
      globalBestIndex = blockBestIndex[0];

    // 1 thread compares iteration best to overall bestsched
    if (GLOBALTID == 0) {
      // Compare to initialSched/current best
      if (globalBestIndex != INVALID_VALUE &&
          dev_AcoSchdulr->shouldReplaceSchedule(dev_bestSched, 
                                                dev_schedules[globalBestIndex], 
                                                true, RPTarget)) {
        #ifdef DEBUG_0_PERP
          InstCount NewCost = dev_schedules[globalBestIndex]->GetExecCost();
          InstCount OldCost = dev_bestSched->GetExecCost();
          InstCount NewSpillCost = dev_schedules[globalBestIndex]->GetNormSpillCost();
          InstCount OldSpillCost = dev_bestSched->GetNormSpillCost();
          if (needsSLIL && dev_bestSched->getIsZeroPerp() && NewCost < OldCost) {
            if (NewSpillCost < OldSpillCost)
              printf("Shorter schedule found with 0 PERP. New RP: %d, Old RP: %d\n", NewSpillCost, OldSpillCost);
            else if ( NewSpillCost > OldSpillCost) {
              printf("Shorter schedule found with 0 PERP. Old was better. New RP: %d, Old RP: %d\n", NewSpillCost, OldSpillCost);
            }
          }
        #endif
        dev_bestSched->Copy(dev_schedules[globalBestIndex]);
        // update RPTarget if we are in second pass and not using SLIL
        if (!needsSLIL)
          RPTarget = dev_bestSched->GetSpillCost();
        int globalStalls = dev_bestSched->getTotalStalls();
        if (globalStalls < dev_AcoSchdulr->GetGlobalBestStalls())
          dev_AcoSchdulr->SetGlobalBestStalls(globalStalls);
        printf("New best sched found by thread %d\n", globalBestIndex);
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, exec cost: %d, and "
               "iteration:%d"
               " (sched length: %d, abs rp cost: %d, rplb: %d)"
               " stalls: %d, unnecessary stalls: %d\n",
               dev_bestSched->GetCost(), dev_bestSched->GetNormSpillCost(),
               dev_bestSched->GetExecCost(), dev_iterations,
               dev_bestSched->GetCrntLngth(), dev_bestSched->GetSpillCost(),
               dev_rgn->GetRPCostLwrBound(),
               dev_bestSched->getTotalStalls(), dev_bestSched->getUnnecessaryStalls());
#if !RUNTIME_TESTING
          dev_noImprovement = 0;
#else
          // for testing compile times disable resetting dev_noImprovement to
          // allow the same number of iterations every time
          dev_noImprovement++;
#endif
        // if a schedule is found with the cost at the lower bound
        // exit the loop after the current iteration is finished
        if ( dev_bestSched && (!IsSecondPass && dev_bestSched->GetNormSpillCost() == 0  || ( IsSecondPass && dev_bestSched->GetExecCost() == 0 ) ) ) {
          lowerBoundSchedFound = true;
        } else if (!needsTarget && !IsSecondPass  && ((BBWithSpill *)dev_rgn)->ReturnPeakSpillCost() == 0) {
          lowerBoundSchedFound = true;
          printf("Schedule with 0 PERP was found\n");
        }
        isGlobalBest = true;
      } else {
        dev_noImprovement++;
        if (dev_noImprovement > noImprovementMax)
          break;
      }
      #ifdef DEBUG_ACO_CRASH_LOCATIONS
        if (hipThreadIdx_x == 0) {
          printf("After Global TID 0 selects best schedule\n");
        }
      #endif
    }
    // perform pheremone update based on selected scheme
#if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
    // Another hard sync point after iteration best selection
    threadGroup.sync();
    if (globalBestIndex != INVALID_VALUE)
      dev_AcoSchdulr->UpdatePheromone(dev_schedules[globalBestIndex], isGlobalBest);
#elif (PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
    // each block finds its blockIterationBest
    if (threadIdx.x == 0) {
      auto bestCost = dev_schedules[GLOBALTID]->GetCost();
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
    if (bestIndex == globalBestIndex) {
      dev_AcoSchdulr->UpdatePheromone(dev_schedules[bestIndex], isGlobalBest);
    }
    else {
      dev_AcoSchdulr->UpdatePheromone(dev_schedules[bestIndex], false);
    }

#elif (PHER_UPDATE_SCHEME == ALL)
    // each block loops over all schedules created by its threads and
    // updates pheromones in block level parallel
    for (int i = blockIdx.x * NUMTHREADSPERBLOCK;
         i < ((blockIdx.x + 1) * NUMTHREADSPERBLOCK); i++) {
      // if sched is within 10% of rp cost and sched length, use it to update pheromone table
      if (dev_schedules[i]->GetNormSpillCost() <= dev_bestSched->GetNormSpillCost() &&
         (!IsSecondPass || dev_schedules[i]->GetExecCost() <= dev_bestSched->GetExecCost())) {
        dev_AcoSchdulr->UpdatePheromone(dev_schedules[i], false);
        atomicAdd(&dev_schedsUsed, 1);
      }
      else {
        atomicAdd(&dev_schedsFound, 1);
      }
    }
  #endif
    threadGroup.sync();
    #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("After UpdatePheromone() loop\n");
    }
    #endif
    dev_AcoSchdulr->ScalePheromoneTable();
    // wait for other blocks to finish before starting next iteration
    threadGroup.sync();
    #ifdef DEBUG_ACO_CRASH_LOCATIONS
    if (hipThreadIdx_x == 0) {
      printf("After ScalePheromoneTable() loop\n");
    }
    #endif
    // make sure no threads reset schedule before above operations complete
    dev_schedules[GLOBALTID]->resetTotalStalls();
    dev_schedules[GLOBALTID]->resetUnnecessaryStalls();
    if (GLOBALTID == 0) {
      dev_iterations++;
      #ifdef DEBUG_INSTR_SELECTION
      printf("Iterations: %d\n", dev_iterations);
      #endif
    }
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
  decay_factor = schedIni.GetInt(IsFirst ? "ACO_DECAY_FACTOR" : "ACO2P_DECAY_FACTOR");
  if (dev_AcoSchdulr)
    dev_AcoSchdulr->fixed_bias = fixed_bias;
  if (count_ < 50)
    noImprovementMax = schedIni.GetInt(IsFirst ? "ACO_STOP_ITERATIONS_RANGE1"
                                             : "ACO2P_STOP_ITERATIONS_RANGE1");
  else if (count_ < 100)
    noImprovementMax = schedIni.GetInt(IsFirst ? "ACO_STOP_ITERATIONS_RANGE2"
                                             : "ACO2P_STOP_ITERATIONS_RANGE2");
  else if (count_ < 1000)
    noImprovementMax = schedIni.GetInt(IsFirst ? "ACO_STOP_ITERATIONS_RANGE3"
                                             : "ACO2P_STOP_ITERATIONS_RANGE3");
  else
    noImprovementMax = schedIni.GetInt(IsFirst ? "ACO_STOP_ITERATIONS_RANGE4"
                                             : "ACO2P_STOP_ITERATIONS_RANGE4");
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
  InstCount TargetSC = InitialSchedule ? InitialSchedule->GetSpillCost()
                                        : heuristicSched->GetSpillCost();  

#if USE_ACS
  initialValue_ = 2.0 / ((double)count_ * heuristicCost);
#else
  initialValue_ = (double)numThreads_ / heuristicCost;
#endif
  for (int i = 0; i < pheromone_size; i++)
    pheromone_[i] = initialValue_;
  std::cerr << "initialValue_" << initialValue_ << std::endl;
  InstSchedule *bestSchedule = InitialSchedule;

  // check if heuristic schedule is better than the initial
  // schedule passed from the list scheduler
  if (shouldReplaceSchedule(InitialSchedule, heuristicSched,
                                /*IsGlobal=*/true, InitialSchedule->GetSpillCost())) {
    bestSchedule = std::move(heuristicSched);
    printf("Heuristic schedule is better\n");
  }
  else {
    bestSchedule = std::move(InitialSchedule);
    printf("Initial schedule is better\n");
  }
  if (bestSchedule) {
    UpdatePheromone(bestSchedule, false);
  }
  bestSchedule->setIsZeroPerp(false);
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  InstSchedule *iterationBest = nullptr;

  // set bestStallsValue to max of lower bound or critical path distance
  InstCount bestStallsValue = std::max(dataDepGraph_->GetSchedLwrBound(), dataDepGraph_->GetRootInst()->GetCrntLwrBound(DIR_BKWRD) + 1)* 6 / 5 - dataDepGraph_->GetInstCnt();
  if (dev_AcoSchdulr)
    // dev_AcoSchdulr->SetGlobalBestStalls(std::max(0, bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt()));
    dev_AcoSchdulr->SetGlobalBestStalls(std::min(bestStallsValue, bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt()));
  else
    // SetGlobalBestStalls(std::max(0, bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt()));
    SetGlobalBestStalls(std::min(bestStallsValue, bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt()));
  printf("bestStallsValue is: %d, initial sched is: %d\n", bestStallsValue, bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt());
  
  if (use_dev_ACO && count_ >= REGION_MIN_SIZE) { // Run ACO on device
    size_t memSize;
    // Update pheromones on device
    CopyPheromonesToDevice(dev_AcoSchdulr);
    Logger::Info("Creating and copying schedules to device"); 
    // An array to temporarily hold schedules to be copied over
    memSize = sizeof(InstSchedule) * numThreads_;
    InstSchedule *temp_schedules = (InstSchedule *)malloc(memSize);
    // An array of pointers to schedules which are copied over
    InstSchedule **host_schedules = new InstSchedule *[numThreads_];
    // Allocate one large array that will be split up between the dev arrays
    // of all InstSchedules. Massively decrease calls to hipMalloc/Free
    InstCount *dev_temp;
    size_t sizePerSched = bestSchedule->GetSizeOfDevArrays();
    memSize = sizePerSched * numThreads_ * sizeof(InstCount);
    gpuErrchk(hipMalloc(&dev_temp, memSize));
    memSize = sizeof(InstSchedule);
    for (int i = 0; i < numThreads_; i++) {
      // Create new schedule
      host_schedules[i] = new InstSchedule(machMdl_, dataDepGraph_, true);
      // Pass a dev array to the schedule to be divided up between the required
      // dev arrays for InstSchedule
      host_schedules[i]->SetDevArrayPointers(dev_MM_, 
                                             &dev_temp[i*sizePerSched]);
      // Copy to temp_schedules array to later copy to device with 1 hipMemcpy
      memcpy(&temp_schedules[i], host_schedules[i], memSize);
    }
    // Allocate and Copy array of schedules to device
    // A device array of schedules
    InstSchedule *dev_schedules_arr;
    memSize = sizeof(InstSchedule) * numThreads_;
    gpuErrchk(hipMalloc(&dev_schedules_arr, memSize));
    // Copy schedules to device
    gpuErrchk(hipMemcpy(dev_schedules_arr, temp_schedules, memSize,
                         hipMemcpyHostToDevice));
    free(temp_schedules);
    // Create a dev array of pointers to dev_schedules_arr
    // Passing and array of schedules and dereferencing the array
    // to get pointers slows down the kernel significantly
    InstSchedule **dev_schedules;
    memSize = sizeof(InstSchedule *) * numThreads_;
    gpuErrchk(hipMallocManaged(&dev_schedules, memSize));
    for (int i = 0; i < numThreads_; i++)
      dev_schedules[i] = &dev_schedules_arr[i];
    gpuErrchk(hipMemPrefetchAsync(dev_schedules, memSize, 0));
    // Copy over best schedule
    // holds device copy of best sched, to be copied back to host after kernel
    InstSchedule *dev_bestSched;
    bestSchedule = new InstSchedule(machMdl_, dataDepGraph_, true);
    bestSchedule->Copy(InitialSchedule);
    bestSchedule->AllocateOnDevice(dev_MM_);
    bestSchedule->CopyArraysToDevice();
    memSize = sizeof(InstSchedule);
    gpuErrchk(hipMalloc((void**)&dev_bestSched, memSize));
    gpuErrchk(hipMemcpy(dev_bestSched, bestSchedule, memSize,
                         hipMemcpyHostToDevice));
    // Create a global mem array for device to use in parallel reduction
    int *dev_blockBestIndex;
    memSize = (NUMBLOCKSMANYANTS/2) * sizeof(int);
    gpuErrchk(hipMalloc(&dev_blockBestIndex, memSize));
    // Allocate array to hold if the only choices we have are negative for RP
    memSize = sizeof(int) * numThreads_;
    gpuErrchk(hipMalloc(&dev_AcoSchdulr->dev_RP0OrPositiveCount, memSize));
    // Make sure managed memory is copied to device before kernel start
    memSize = sizeof(ACOScheduler);
    gpuErrchk(hipMemPrefetchAsync(dev_AcoSchdulr, memSize, 0));
    Logger::Info("Launching Dev_ACO with %d blocks of %d threads", numBlocks_,
                                                           NUMTHREADSPERBLOCK);
    // Using Cooperative Grid Groups requires launching with
    // hipLaunchCooperativeKernel which requires kernel args to be an array
    // of void pointers to host memory locations of the arguments
    dim3 gridDim(numBlocks_);
    dim3 blockDim(NUMTHREADSPERBLOCK);
    void *dArgs[7];
    dArgs[0] = (void*)&dev_rgn_;
    dArgs[1] = (void*)&dev_DDG_;
    dArgs[2] = (void*)&dev_AcoSchdulr;
    dArgs[3] = (void*)&dev_schedules;
    dArgs[4] = (void*)&dev_bestSched;
    dArgs[5] = (void*)&noImprovementMax;
    dArgs[6] = (void*)&dev_blockBestIndex;
    gpuErrchk(hipLaunchCooperativeKernel((void*)Dev_ACO, gridDim, blockDim, 
                                          dArgs, 0, NULL));
    hipDeviceSynchronize();
    Logger::Info("Post Kernel Error: %s", 
                 hipGetErrorString(hipGetLastError()));
    // Copy dev_bestSched back to host
    memSize = sizeof(InstSchedule);
    gpuErrchk(hipMemcpy(bestSchedule, dev_bestSched, memSize,
                         hipMemcpyDeviceToHost));
    bestSchedule->CopyArraysToHost();
    // Free allocated memory that is no longer needed
    bestSchedule->FreeDeviceArrays();
    hipFree(dev_bestSched);
    for (int i = 0; i < numThreads_; i++) {
      delete host_schedules[i];
    }
    delete[] host_schedules;
    // delete the large array shared by all schedules
    hipFree(dev_temp);
    hipFree(dev_schedules);

  } else { // Run ACO on cpu
    Logger::Info("Running host ACO with %d ants per iteration", numThreads_);
    InstCount RPTarget;
    if (!((BBWithSpill *)rgn_)->needsSLIL())
      RPTarget = bestSchedule->GetSpillCost();
    else
      RPTarget = MaxRPTarget;
    #ifdef CHECK_DIFFERENT_SCHEDULES
      std::unordered_map<string, int> schedMap;
      int diffSchedCount = 0;
    #endif
    while (noImprovement < noImprovementMax) {
      iterations++;
      iterationBest = nullptr;
      for (int i = 0; i < numThreads_; i++) {
        InstSchedule *schedule = FindOneSchedule(RPTarget);

        #ifdef CHECK_DIFFERENT_SCHEDULES
          // check if schedule is in Map
          InstCount instNum, cycleNum, slotNum;
          // get first instruction in string
          instNum = schedule->GetFrstInst(cycleNum, slotNum);
          std::string schedString = std::to_string(instNum);
          // prepare next instruction in comma separated list
          instNum = schedule->GetNxtInst(cycleNum, slotNum);
          while (instNum != INVALID_VALUE) {
            schedString.append(",");
            schedString.append(std::to_string(instNum));
            instNum = schedule->GetNxtInst(cycleNum, slotNum);
          }
          schedule->ResetInstIter();
          if (schedMap.find(schedString) == schedMap.end()) {
            schedMap[schedString] = 1;
            diffSchedCount++;
          }
          else {
            schedMap[schedString] = schedMap[schedString] + 1;
          }
        #endif

        if (print_aco_trace)
          PrintSchedule(schedule);
        if (shouldReplaceSchedule(iterationBest, schedule, false, RPTarget)) {
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
        UpdatePheromone(iterationBest, false);
#endif
      if (shouldReplaceSchedule(bestSchedule, iterationBest, true, RPTarget)) {
        if (bestSchedule && bestSchedule != InitialSchedule)
          delete bestSchedule;
        bestSchedule = std::move(iterationBest);
        if (!((BBWithSpill *)rgn_)->needsSLIL())
          RPTarget = bestSchedule->GetSpillCost();

        int globalStalls = bestSchedule->getTotalStalls();
        if (globalStalls < GetGlobalBestStalls())
          SetGlobalBestStalls(bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt());
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, exec cost: %d, and "
               "iteration:%d"
               " (sched length: %d, abs rp cost: %d, rplb: %d)"
               " stalls: %d, unnecessary stalls: %d\n",
               bestSchedule->GetCost(), bestSchedule->GetNormSpillCost(),
               bestSchedule->GetExecCost(), iterations,
               bestSchedule->GetCrntLngth(), bestSchedule->GetSpillCost(),
               rgn_->GetRPCostLwrBound(),
               bestSchedule->getTotalStalls(), bestSchedule->getUnnecessaryStalls());
#if !RUNTIME_TESTING
          noImprovement = 0;
#else
          // Disable resetting noImp to lock iterations to 10
          noImprovement++;
#endif
        if (bestSchedule && ( IsFirst && (bestSchedule->GetNormSpillCost() == 0 ||
        ((BBWithSpill *)rgn_)->ReturnPeakSpillCost() == 0) ||
        ( !IsFirst && bestSchedule->GetExecCost() == 0 ) ) )
          break;
      } else {
        delete iterationBest;
        noImprovement++;
      }
#if USE_ACS
      UpdatePheromone(bestSchedule, false);
#endif
    }
    Logger::Info("%d ants terminated early", numAntsTerminated_);
    #ifdef CHECK_DIFFERENT_SCHEDULES
    Logger::Info("%d different schedules for %d total ants", diffSchedCount, (iterations + 1) * numThreads_ - numAntsTerminated_);
    #endif
  } // End run on CPU

  printf("Best schedule: ");
  printf("Absolute RP Cost: %d, Length: %d, Cost: ", bestSchedule->GetSpillCost(), bestSchedule->GetCrntLngth());
  PrintSchedule(bestSchedule);
  schedule_out->Copy(bestSchedule);
  if (bestSchedule != InitialSchedule)
    delete bestSchedule;
  if (!use_dev_ACO || count_ < REGION_MIN_SIZE)
    printf("ACO finished after %d iterations\n", iterations);

  return RES_SUCCESS;
}
__device__
double dmax(double d1, double d2) {
  return d1 >= d2 ? d1 : d2;
}

__device__
double dmin(double d1, double d2) {
  return d1 <= d2 ? d1 : d2;
}

__host__ __device__
void ACOScheduler::UpdatePheromone(InstSchedule *schedule, bool isIterationBest) {
#ifdef __HIP_DEVICE_COMPILE__ // device version of function
#if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
  // parallel on global level
  int instNum = GLOBALTID;
#elif (PHER_UPDATE_SCHEME == ALL || PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
  // parallel on block level
  int instNum = hipThreadIdx_x;
#endif
  // Each thread updates pheromone table for 1 instruction
  // For the case numThreads < count_, increase instNum by
  // numThreads at the end of the loop.
  InstCount lastInstNum = -1;
  pheromone_t portion = schedule->GetCost() / (ScRelMax * 1.5);
  pheromone_t deposition;
    if (isIterationBest)
    deposition = 100000;
  else {
    if (portion < 1)
      deposition = (1 - portion) * MAX_DEPOSITION_MINUS_MIN + MIN_DEPOSITION;
    else {
      #if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
        deposition = MIN_DEPOSITION;
      #elif (PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
        deposition = MIN_DEPOSITION / 80;
      #elif (PHER_UPDATE_SCHEME == ALL)
        deposition = MIN_DEPOSITION / numThreads_;
      #endif
    }
  }

  pheromone_t *pheromone;
  while (instNum < count_) {
    // Get the instruction that comes before inst in the schedule
    // if instNum == count_ - 2 it has the root inst and lastInstNum = -1
    lastInstNum = schedule->GetPrevInstNum(instNum);
    // Get corresponding pheromone and update it
    pheromone = &Pheromone(lastInstNum, instNum);
    atomicAdd(pheromone, deposition);
#if (PHER_UPDATE_SCHEME == ONE_PER_ITER)
    // parallel on global level
    // Increase instNum by numThreads_ until over count_
    instNum += numThreads_;
#elif (PHER_UPDATE_SCHEME == ALL || PHER_UPDATE_SCHEME == ONE_PER_BLOCK)
    // parallel on block level
    instNum += NUMTHREADSPERBLOCK;
#endif
  }
  if (print_aco_trace && GLOBALTID==0)
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
      pheromone = &Pheromone(i, j);
      *pheromone *= (1 - decay_factor);
    }
  }
#endif
  if (print_aco_trace)
    PrintPheromone();
#endif
}

__host__ __device__
void ACOScheduler::ScalePheromoneTable() {
#ifdef __HIP_DEVICE_COMPILE__ // device version of function
  int instNum = GLOBALTID;
  dev_totalPherInTable = 0;
  // Each thread updates pheromone table for 1 instruction
  // For the case numThreads < count_, increase instNum by
  // numThreads at the end of the loop.
  pheromone_t *pheromone;
  while (instNum < count_) {
    // decay pheromone for all trails leading to instNum
    for (int j = 0; j < count_; j++) {
      pheromone = &Pheromone(j, instNum);
      *pheromone *= (1 - decay_factor);
      // clamp pheromone values to be between 1 and 8
      *pheromone = dmax(1, dmin(8, *pheromone));
      atomicAdd(&dev_totalPherInTable, *pheromone);
    }
    // parallel on global level
    // Increase instNum by numThreads_ until over count_
    instNum += numThreads_;
  }
  // adjust pheromone table by scaling factor
  // pheromone_t scalingFactor = (double) (count_ * count_ * 4.5)/dev_totalPherInTable;
  pheromone_t scalingAdjustment = 4.5 - dev_totalPherInTable / (count_ * count_);
  instNum = GLOBALTID;
  while (instNum < count_) {
    // restrict pheromone for all trails leading to instNum
    // to be within range
    for (int j = 0; j < count_; j++) {
      pheromone = &Pheromone(j, instNum);
      *pheromone = *pheromone + scalingAdjustment;
      *pheromone = dmax(1, dmin(8, *pheromone));
    }
    // parallel on global level
    // Increase instNum by numThreads_ until over count_
    instNum += numThreads_;
  }
  if (print_aco_trace && GLOBALTID==0)
    PrintPheromone();

#else // host version of function

  pheromone_t *pheromone;
  pheromone_t totalPherInTable;
  totalPherInTable = 0;
  // clamp pheromone to range
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      pheromone = &Pheromone(i, j);
      *pheromone *= (1 - decay_factor);
      *pheromone = fmax(1, fmin(8, *pheromone));
      totalPherInTable += *pheromone;
    }
  }

  // pheromone_t scalingFactor = (double) (count_ * count_ * 4.5)/totalPherInTable;
  pheromone_t scalingAdjustment = 4.5 - totalPherInTable / (count_ * count_);
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      pheromone = &Pheromone(i, j);
      *pheromone = *pheromone + scalingAdjustment;
      *pheromone = fmax(1, fmin(8, *pheromone));
    }
  }
  if (print_aco_trace)
    PrintPheromone();
#endif
}

__device__
void ACOScheduler::CopyPheromonesToSharedMem(double *s_pheromone) {
  InstCount toInstNum = hipThreadIdx_x;
  while (toInstNum < count_) {
    for (int fromInstNum = -1; fromInstNum < count_; fromInstNum++)
      s_pheromone[((fromInstNum + 1) * count_) + toInstNum] = 
                                        Pheromone(fromInstNum, toInstNum);
    toInstNum += NUMTHREADSPERBLOCK;
  }
}

__host__ __device__
inline void ACOScheduler::UpdateACOReadyList(SchedInstruction *inst) {
  InstCount prdcsrNum, scsrRdyCycle;
  
  #ifdef __HIP_DEVICE_COMPILE__ // device version of function
    // Notify each successor of this instruction that it has been scheduled.
    #ifdef DEBUG_INSTR_SELECTION
    if (GLOBALTID==0) {
      printf("successors of %d:", inst->GetNum());
    }
    #endif
    #ifdef DEBUG_ACO_CRASH_LOCATIONS
      if (hipThreadIdx_x == 0) {
        printf("Before for loop inside UpdateACOReadyList()\n");
      }
    #endif
    int i = 0;
    for (SchedInstruction *crntScsr = GetScsr(inst, i++, &prdcsrNum);
          crntScsr != NULL; crntScsr = GetScsr(inst, i++, &prdcsrNum)) {
        #ifdef DEBUG_INSTR_SELECTION
        if (GLOBALTID==0) {
          printf(" %d,", crntScsr->GetNum());
        }
        #endif
        bool wasLastPrdcsr =
            crntScsr->PrdcsrSchduld(prdcsrNum, dev_crntCycleNum_[GLOBALTID], scsrRdyCycle, dev_DDG_->ltncyPerPrdcsr_);

        if (wasLastPrdcsr) {
          // If all other predecessors of this successor have been scheduled then
          // we now know in which cycle this successor will become ready.
          HeurType HeurWOLuc = dev_kHelper->computeKey(crntScsr, false, dev_DDG_->RegFiles, dev_DDG_);
          dev_readyLs->addInstructionToReadyList(ACOReadyListEntry{crntScsr->GetNum(), scsrRdyCycle, HeurWOLuc, 0});
        }
    }
    #ifdef DEBUG_INSTR_SELECTION
    if (GLOBALTID==0) {
      printf("\n");
    }
    #endif
    // Make sure the scores are valid.  The scheduling of an instruction may
    // have increased another instruction's LUC Score
    PriorityEntry LUCEntry = dev_kHelper->getPriorityEntry(LSH_LUC);
    dev_RP0OrPositiveCount[GLOBALTID] = 0;
    for (InstCount I = 0; I < dev_readyLs->getReadyListSize(); ++I) {
      //we first get the heuristic without the LUC component, add the LUC
      //LUC component, and then compute the score
      HeurType Heur = *dev_readyLs->getInstHeuristicAtIndex(I);
      InstCount CandidateId = *dev_readyLs->getInstIdAtIndex(I);
      if (LUCEntry.Width) {
        SchedInstruction *ScsrInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType LUCVal = ScsrInst->CmputLastUseCnt(dev_DDG_->RegFiles, dev_DDG_);
        LUCVal <<= LUCEntry.Offset;
        Heur &= LUCVal;
      }
      if (dev_RP0OrPositiveCount[GLOBALTID]) {
        if (*dev_readyLs->getInstReadyOnAtIndex(I) > dev_crntCycleNum_[GLOBALTID])
          continue;

        SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType candidateLUC = candidateInst->GetLastUseCnt();
        int16_t candidateDefs = candidateInst->GetDefCnt();
        if (candidateDefs <= candidateLUC) {
          dev_RP0OrPositiveCount[GLOBALTID] = dev_RP0OrPositiveCount[GLOBALTID] + 1;
        }
      }
    }
  #else // host version of function
    // Notify each successor of this instruction that it has been scheduled.
    for (SchedInstruction *crntScsr = inst->GetFrstScsr(&prdcsrNum);
          crntScsr != NULL; crntScsr = inst->GetNxtScsr(&prdcsrNum)) {
        bool wasLastPrdcsr =
            crntScsr->PrdcsrSchduld(prdcsrNum, crntCycleNum_, scsrRdyCycle);

        if (wasLastPrdcsr) {
          // If all other predecessors of this successor have been scheduled then
          // we now know in which cycle this successor will become ready.
          HeurType HeurWOLuc = kHelper->computeKey(crntScsr, false, dataDepGraph_->RegFiles);
          readyLs->addInstructionToReadyList(ACOReadyListEntry{crntScsr->GetNum(), scsrRdyCycle, HeurWOLuc, 0});
        }
    }

    // Make sure the scores are valid.  The scheduling of an instruction may
    // have increased another instruction's LUC Score
    PriorityEntry LUCEntry = kHelper->getPriorityEntry(LSH_LUC);
    RP0OrPositiveCount = 0;
    for (InstCount I = 0; I < readyLs->getReadyListSize(); ++I) {
      //we first get the heuristic without the LUC component, add the LUC
      //LUC component, and then compute the score
      HeurType Heur = *readyLs->getInstHeuristicAtIndex(I);
      InstCount CandidateId = *readyLs->getInstIdAtIndex(I);
      if (LUCEntry.Width) {
        SchedInstruction *ScsrInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType LUCVal = ScsrInst->CmputLastUseCnt(dataDepGraph_->RegFiles);
        LUCVal <<= LUCEntry.Offset;
        Heur &= LUCVal;
      }
      if (RP0OrPositiveCount) {
        if (*dev_readyLs->getInstReadyOnAtIndex(I) > crntCycleNum_)
          continue;

        SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType candidateLUC = candidateInst->GetLastUseCnt();
        int16_t candidateDefs = candidateInst->GetDefCnt();
        if (candidateDefs <= candidateLUC) {
          RP0OrPositiveCount = RP0OrPositiveCount + 1;
        }
      }
    }
  #endif
}

// copied from Enumerator
inline void ACOScheduler::UpdtRdyLst_(InstCount cycleNum, int slotNum) {
  assert(false); // do not use this function with aco
  // it is only implemented b/c it is a pure virtual in ConstrainedScheduler
}

__host__ __device__
void ACOScheduler::PrintPheromone() {
  printf("Pher: ");
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      //std::cerr << std::scientific << std::setprecision(8) << Pheromone(i, j)
      //          << " ";
      printf("%.1f ", Pheromone(i, j));
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

  /*std::cerr << " defs ";
  llvm::opt_sched::Register **defs;
  uint16_t defsCount = inst->GetDefs(defs);
  for (uint16_t i = 0; i < defsCount; i++) {
    std::cerr << defs[i]->GetNum() << defs[i]->GetType();
    if (i != defsCount - 1)
      std::cerr << ", ";
  }

  std::cerr << " uses ";
  llvm::opt_sched::Register **uses;
  uint16_t usesCount = inst->GetUses(uses);
  for (uint16_t i = 0; i < usesCount; i++) {
    std::cerr << uses[i]->GetNum() << uses[i]->GetType();
    if (i != usesCount - 1)
      std::cerr << ", ";
  }*/
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
  memSize = sizeof(InstCount) * numThreads_;
  gpuErrchk(hipMalloc(&dev_schduldInstCnt_, memSize));
  // Alloc dev array for crntCycleNum_;
  memSize = sizeof(InstCount) * numThreads_;
  gpuErrchk(hipMalloc(&dev_crntCycleNum_, memSize));
  // Alloc dev array for crntSlotNum_;
  memSize = sizeof(InstCount) * numThreads_;
  gpuErrchk(hipMalloc(&dev_crntSlotNum_, memSize));
  // Alloc dev array for isCrntCycleBlkd_;
  memSize = sizeof(bool) * numThreads_;
  gpuErrchk(hipMalloc(&dev_isCrntCycleBlkd_, memSize));
  
  // Alloc dev array for readyLs;
  readyLs->AllocDevArraysForParallelACO(numThreads_);
  // Alloc dev arrays for MaxScoringInst
  memSize = sizeof(InstCount) * numThreads_;
  gpuErrchk(hipMalloc(&dev_MaxScoringInst, memSize));
  // Alloc dev array for avlblSlotsInCrntCycle_
  memSize = sizeof(int16_t) * issuTypeCnt_ * numThreads_;
  gpuErrchk(hipMalloc(&dev_avlblSlotsInCrntCycle_, memSize));
  // Alloc dev arrays for rsrvSlots_
  memSize = sizeof(ReserveSlot) * issuRate_ * numThreads_;
  gpuErrchk(hipMalloc(&dev_rsrvSlots_, memSize));
  memSize = sizeof(int16_t) * numThreads_;
  gpuErrchk(hipMalloc(&dev_rsrvSlotCnt_, memSize));
}

void ACOScheduler::CopyPheromonesToDevice(ACOScheduler *dev_AcoSchdulr) {
  size_t memSize;
  // Free allocated mem sinve pheromone size can change
  if (dev_AcoSchdulr->dev_pheromone_elmnts_alloced_ == true)
    hipFree(dev_AcoSchdulr->pheromone_.elmnts_);

  memSize = sizeof(DeviceVector<pheromone_t>);
  gpuErrchk(hipMemcpy(&dev_AcoSchdulr->pheromone_, &pheromone_, memSize,
            hipMemcpyHostToDevice));

  memSize = sizeof(pheromone_t) * pheromone_.alloc_;
  gpuErrchk(hipMalloc(&(dev_AcoSchdulr->pheromone_.elmnts_), memSize));
  gpuErrchk(hipMemcpy(dev_AcoSchdulr->pheromone_.elmnts_, pheromone_.elmnts_,
		       memSize, hipMemcpyHostToDevice));
  
  dev_AcoSchdulr->dev_pheromone_elmnts_alloced_ = true;
}

void ACOScheduler::CopyPointersToDevice(ACOScheduler *dev_ACOSchedulr) {
  size_t memSize;
  dev_ACOSchedulr->machMdl_ = dev_MM_;
  dev_ACOSchedulr->dataDepGraph_ = dev_DDG_;
  // Copy slotsPerTypePerCycle_
  int *dev_slotsPerTypePerCycle;
  memSize = sizeof(int) * issuTypeCnt_;
  gpuErrchk(hipMalloc(&dev_slotsPerTypePerCycle, memSize));
  gpuErrchk(hipMemcpy(dev_slotsPerTypePerCycle, slotsPerTypePerCycle_,
		       memSize, hipMemcpyHostToDevice));
  dev_ACOSchedulr->slotsPerTypePerCycle_ = dev_slotsPerTypePerCycle;
  // Copy instCntPerIssuType_
  InstCount *dev_instCntPerIssuType;
  memSize = sizeof(InstCount) * issuTypeCnt_;
  gpuErrchk(hipMalloc(&dev_instCntPerIssuType, memSize));
  gpuErrchk(hipMemcpy(dev_instCntPerIssuType, instCntPerIssuType_, memSize,
		       hipMemcpyHostToDevice));
  dev_ACOSchedulr->instCntPerIssuType_ = dev_instCntPerIssuType;
  // set root/leaf inst
  dev_ACOSchedulr->rootInst_ = dev_DDG_->GetRootInst();
  dev_ACOSchedulr->leafInst_ = dev_DDG_->GetLeafInst();
  // copy readyLs
  memSize = sizeof(ACOReadyList);
  gpuErrchk(hipMalloc(&dev_ACOSchedulr->dev_readyLs, memSize));
  gpuErrchk(hipMemcpy(dev_ACOSchedulr->dev_readyLs, readyLs, memSize,
		       hipMemcpyHostToDevice));
  // copy khelper
  memSize = sizeof(KeysHelper);
  gpuErrchk(hipMalloc(&dev_ACOSchedulr->dev_kHelper, memSize));
  gpuErrchk(hipMemcpy(dev_ACOSchedulr->dev_kHelper, kHelper, memSize,
		       hipMemcpyHostToDevice));
}

void ACOScheduler::FreeDevicePointers() {
  hipFree(dev_schduldInstCnt_);
  hipFree(dev_crntCycleNum_);
  hipFree(dev_crntSlotNum_);
  hipFree(dev_isCrntCycleBlkd_);
  hipFree(slotsPerTypePerCycle_);
  hipFree(instCntPerIssuType_);
  hipFree(dev_MaxScoringInst);
  readyLs->FreeDevicePointers();
  hipFree(dev_avlblSlotsInCrntCycle_);
  hipFree(dev_rsrvSlots_);
  hipFree(dev_rsrvSlotCnt_);
  hipFree(dev_readyLs);
  hipFree(dev_kHelper);
  hipFree(pheromone_.elmnts_);
}
