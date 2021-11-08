#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
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
			   MachineModel *dev_MM, curandState_t *dev_states)
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

  //construct the ACOReadyList member and a key helper
  int maxIndpntInstrs = 0;
  for (int i = 0; i < count_; i++) {
    int indpntInstrs = count_ - dataDepGraph->GetInstByIndx(i)->GetRcrsvPrdcsrCnt() - dataDepGraph->GetInstByIndx(i)->GetRcrsvScsrCnt();
    maxIndpntInstrs = indpntInstrs > maxIndpntInstrs ? indpntInstrs : maxIndpntInstrs;
  }
  Logger::Info("Ready List Size is: %d, Percent of total number of instructions: %f", maxIndpntInstrs, double(maxIndpntInstrs)/double(count_));
  readyLs = new ACOReadyList(maxIndpntInstrs);
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
InstCount ACOScheduler::SelectInstruction(SchedInstruction *lastInst, InstCount totalStalls, SchedRegion *rgn, bool &unnecessarilyStalling, bool closeToRPTarget, bool currentlyWaiting) {
#ifdef __CUDA_ARCH__
  // loop through instructions to see if all fully-ready instructions have a negative effect on RP
  // if there are no fully-ready instructions, then onlyRPNegative will be true
  bool onlyRPNegative = true;
  for (InstCount I = 0; I < dev_readyLs->getReadyListSize(); ++I) {
    // For this, do not consider instructions that are semi-ready,
    // consider them once they become fully ready
    if (*dev_readyLs->getInstReadyOnAtIndex(I) > dev_crntCycleNum_[GLOBALTID])
      continue;
    
    InstCount CandidateId = *dev_readyLs->getInstIdAtIndex(I);
    SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
    HeurType candidateLUC = candidateInst->GetLastUseCnt();
    int16_t candidateDefs = candidateInst->GetDefCnt();
    if (candidateDefs <= candidateLUC) {
      onlyRPNegative = false;
      break;
    }
  }
  // if we are waiting and have no fully-ready instruction that is 
  // net 0 or benefit to RP, then return -1 to schedule a stall
  if (currentlyWaiting && onlyRPNegative)
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
  bool tooManyStalls = totalStalls >= globalBestStalls_ * 5 / 10;
  dev_readyLs->dev_ScoreSum[GLOBALTID] = 0;

  for (InstCount I = 0; I < dev_readyLs->getReadyListSize(); ++I) {
    RPIsHigh = false;
    InstCount CandidateId = *dev_readyLs->getInstIdAtIndex(I);
    SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
    HeurType candidateLUC = candidateInst->GetLastUseCnt();
    int16_t candidateDefs = candidateInst->GetDefCnt();
    if (currentlyWaiting) {
      // if currently waiting on an instruction, do not consider semi-ready instructions 
      if (*dev_readyLs->getInstReadyOnAtIndex(I) > dev_crntCycleNum_[GLOBALTID])
        continue;

      // as well as instructions with a net negative impact on RP
      if (candidateDefs > candidateLUC)
        continue;
    }
    
    // compute the score
    HeurType Heur = *dev_readyLs->getInstHeuristicAtIndex(I);
    pheromone_t IScore = Score(lastInstId, *dev_readyLs->getInstIdAtIndex(I), Heur);
    if (!onlyRPNegative && candidateDefs > candidateLUC)
      IScore = IScore * 9/10;

    *dev_readyLs->getInstScoreAtIndex(I) = IScore;
    dev_readyLs->dev_ScoreSum[GLOBALTID] += IScore;
    // add a score penalty for instructions that are not ready yet
    // unnecessary stalls should not be considered if current RP is low, or if we already have too many stalls
    if (*dev_readyLs->getInstReadyOnAtIndex(I) > dev_crntCycleNum_[GLOBALTID]) {
      if (!onlyRPNegative) {
        IScore = 0.0000001;
      }
      else {
        int cyclesNeededToWait = *dev_readyLs->getInstReadyOnAtIndex(I) - dev_crntCycleNum_[GLOBALTID];
        if (cyclesNeededToWait < globalBestStalls_)
          IScore = IScore * (globalBestStalls_ - cyclesNeededToWait) / globalBestStalls_;
        else 
          IScore = IScore / globalBestStalls_;

        // check if any reg types used by the instructions are above the physical register limit
        SchedInstruction *tempInst = dataDepGraph_->GetInstByIndx(*dev_readyLs->getInstIdAtIndex(I));
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
            IScore = IScore * (globalBestStalls_ - totalStalls) / globalBestStalls_;
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
    *dev_readyLs->getInstScoreAtIndex(I) = IScore;
    dev_readyLs->dev_ScoreSum[GLOBALTID] += IScore;
    
    if(IScore > MaxScore) {
      MaxScoreIndx = I;
      MaxScore = IScore;
    }
  }
#else
  // loop through instructions to see if all fully-ready instructions have a negative effect on RP
  // if there are no fully-ready instructions, then onlyRPNegative will be true
  bool onlyRPNegative = true;
  for (InstCount I = 0; I < readyLs->getReadyListSize(); ++I) {
    // For this, do not consider instructions that are semi-ready,
    // consider them once they become fully ready
    if (*readyLs->getInstReadyOnAtIndex(I) > crntCycleNum_)
      continue;
    
    InstCount CandidateId = *readyLs->getInstIdAtIndex(I);
    SchedInstruction *candidateInst = dataDepGraph_->GetInstByIndx(CandidateId);
    HeurType candidateLUC = candidateInst->GetLastUseCnt();
    int16_t candidateDefs = candidateInst->GetDefCnt();
    if (candidateDefs <= candidateLUC) {
      onlyRPNegative = false;
      break;
    }
  }
  // if we are waiting and have no fully-ready instruction that is 
  // net 0 or benefit to RP, then return -1 to schedule a stall
  if (currentlyWaiting && onlyRPNegative)
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
    if (currentlyWaiting) {
      // if currently waiting on an instruction, do not consider semi-ready instructions 
      if (*readyLs->getInstReadyOnAtIndex(I) > crntCycleNum_)
        continue;

      // as well as instructions with a net negative impact on RP
      if (candidateDefs > candidateLUC)
        continue;
    }
    
    // compute the score
    HeurType Heur = *readyLs->getInstHeuristicAtIndex(I);
    pheromone_t IScore = Score(lastInstId, *readyLs->getInstIdAtIndex(I), Heur);
    if (!onlyRPNegative && candidateDefs > candidateLUC)
      IScore = IScore * 9/10;

    *readyLs->getInstScoreAtIndex(I) = IScore;
    readyLs->ScoreSum += IScore;
    // add a score penalty for instructions that are not ready yet
    // unnecessary stalls should not be considered if current RP is low, or if we already have too many stalls
    if (*readyLs->getInstReadyOnAtIndex(I) > crntCycleNum_) {
      if (!onlyRPNegative) {
        IScore = 0.0000001;
      }
      else {
        int cyclesNeededToWait = *readyLs->getInstReadyOnAtIndex(I) - crntCycleNum_;
        if (cyclesNeededToWait < globalBestStalls_)
          IScore = IScore * (globalBestStalls_ - cyclesNeededToWait) / globalBestStalls_;
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
            IScore = IScore * (globalBestStalls_ - totalStalls) / globalBestStalls_;
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
#ifdef __CUDA_ARCH__
  rand = curand_uniform(&dev_states_[GLOBALTID]);
  point = dev_readyLs->dev_ScoreSum[GLOBALTID] * curand_uniform(&dev_states_[GLOBALTID]);
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
  #ifdef __CUDA_ARCH__
    size_t fpIndx=0;
    for (size_t i = 0; i < dev_readyLs->getReadyListSize(); ++i) {
      // if we are avoiding unnecessary stalls, skip over instructions that are not latency ready
      // if (couldAvoidStalling && avoidStalling && *dev_readyLs->getInstReadyOnAtIndex(i))
      //   continue;
      point -= *dev_readyLs->getInstScoreAtIndex(i);
      if (point <= 0) {
        if (couldAvoidStalling && *dev_readyLs->getInstReadyOnAtIndex(i) > dev_crntCycleNum_[GLOBALTID]) {
          unnecessarilyStalling = true;
        }
        else
          unnecessarilyStalling = false;
        fpIndx = i;
        break;
      }
    }
  #else
    size_t fpIndx=0;
    for (size_t i = 0; i < readyLs->getReadyListSize(); ++i) {
      // if we are avoiding unnecessary stalls, skip over instructions that are not latency ready
      // if (couldAvoidStalling && avoidStalling && *readyLs->getInstReadyOnAtIndex(i))
      //   continue;
      point -= *readyLs->getInstScoreAtIndex(i);
      if (point <= 0) {
        if (couldAvoidStalling && *readyLs->getInstReadyOnAtIndex(i) > crntCycleNum_) {
          unnecessarilyStalling = true;
        }
        else
          unnecessarilyStalling = false;
        fpIndx = i;
        break;
      }
    }
  #endif
  //finally we pick whether we will return the fp choice or max score inst w/o using a branch
  bool UseMax = (rand < choose_best_chance) || currentlyWaiting;
  size_t indx = UseMax ? MaxScoreIndx : fpIndx;
  // #ifdef __CUDA_ARCH__
  //   if (GLOBALTID==0)
  //     printf("UseMax: %s, maxInstNum: %d, maxInstScore: %f, fpInstNum: %d, readySize: %d\n", UseMax ? "true" : "false", *dev_readyLs->getInstIdAtIndex(MaxScoreIndx), *dev_readyLs->getInstScoreAtIndex(MaxScoreIndx), *dev_readyLs->getInstIdAtIndex(fpIndx), dev_readyLs->getReadyListSize());
  // #else
  //   printf("UseMax: %s, maxInstNum: %d, maxInstScore: %f, fpInstNum: %d, readySize: %d\n", UseMax ? "true" : "false", *readyLs->getInstIdAtIndex(MaxScoreIndx), *readyLs->getInstScoreAtIndex(MaxScoreIndx), *readyLs->getInstIdAtIndex(fpIndx), dev_readyLs->getReadyListSize());
  // #endif
  #ifdef __CUDA_ARCH__
    if (couldAvoidStalling && *dev_readyLs->getInstReadyOnAtIndex(indx) > dev_crntCycleNum_[GLOBALTID])
      unnecessarilyStalling = true;
    else
      unnecessarilyStalling = false;
    
    // if (GLOBALTID==0)
    //   printf("UseMax: %s, instNum: %d, unnecessarilyStalling: %s\n", UseMax ? "true" : "false", *dev_readyLs->getInstIdAtIndex(indx), unnecessarilyStalling ? "true" : "false");
  #else
    if (couldAvoidStalling && *readyLs->getInstReadyOnAtIndex(indx) > crntCycleNum_)
      unnecessarilyStalling = true;
    else
      unnecessarilyStalling = false;
    // printf("UseMax: %s, instNum: %d, unnecessarilyStalling: %s\n", UseMax ? "true" : "false", *readyLs->getInstIdAtIndex(indx), unnecessarilyStalling ? "true" : "false");
  #endif
  return indx;
}

__host__ __device__
InstSchedule *ACOScheduler::FindOneSchedule(InstCount RPTarget, 
                                            InstSchedule *dev_schedule) {
#ifdef __CUDA_ARCH__ // device version of function
  SchedInstruction *inst = NULL;
  SchedInstruction *lastInst = NULL;
  ACOReadyListEntry LastInstInfo;
  InstSchedule *schedule = dev_schedule;
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
  HeurType RootHeuristic = dev_kHelper->computeKey(rootInst_, true);
  pheromone_t RootScore = Score(-1, RootId, RootHeuristic);
  ACOReadyListEntry InitialRoot{RootId, 0, RootHeuristic, RootScore};
  dev_readyLs->addInstructionToReadyList(InitialRoot);
  dev_readyLs->dev_ScoreSum[GLOBALTID] = RootScore;
  dev_MaxScoringInst[GLOBALTID] = 0;
  lastInst = dataDepGraph_->GetInstByIndx(RootId);
  bool closeToRPTarget = false;

  while (!IsSchedComplete_()) {

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    if (!waitFor && waitUntil <= dev_crntCycleNum_[GLOBALTID]) {
      assert(dev_readyLs->getReadyListSize());
      
      InstCount closeToRPCheck = RPTarget - 2 < RPTarget * 9 / 10 ? RPTarget - 2 : RPTarget * 9 / 10;
      closeToRPTarget = ((BBWithSpill *)dev_rgn_)->GetCrntSpillCost() >= closeToRPCheck;
      // select the instruction and get info on it
      InstCount SelIndx = SelectInstruction(lastInst, schedule->getTotalStalls(), dev_rgn_, unnecessarilyStalling, closeToRPTarget, waitFor ? true: false);

      if (SelIndx != -1) {
        LastInstInfo = dev_readyLs->removeInstructionAtIndex(SelIndx);
        
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

      // new readylist update
      UpdateACOReadyList(inst);
    }
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
  }
  dev_rgn_->UpdateScheduleCost(schedule);
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
  HeurType RootHeuristic = kHelper->computeKey(rootInst_, true);
  pheromone_t RootScore = Score(-1, RootId, RootHeuristic);
  ACOReadyListEntry InitialRoot{RootId, 0, RootHeuristic, RootScore};
  readyLs->addInstructionToReadyList(InitialRoot);
  readyLs->ScoreSum = RootScore;
  MaxScoringInst = 0;
  lastInst = dataDepGraph_->GetInstByIndx(RootId);
  bool closeToRPTarget = false;

  SchedInstruction *inst = NULL;
  while (!IsSchedComplete_()) {

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    inst = NULL;
    if (!(waitFor && waitUntil <= crntCycleNum_)) {
      // If an instruction is ready select it
      assert(readyLs->getReadyListSize()); // we should always have something in the rl

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
__device__ bool lowerBoundSchedFound;

__global__
void Dev_ACO(SchedRegion *dev_rgn, DataDepGraph *dev_DDG,
            ACOScheduler *dev_AcoSchdulr, InstSchedule **dev_schedules,
            InstSchedule *dev_bestSched, int noImprovementMax, 
            int *blockBestIndex) {
  // holds cost and index of bestSched per block
  __shared__ int bestIndex, dev_iterations;
  __shared__ bool needsSLIL;
  needsSLIL = ((BBWithSpill *)dev_rgn)->needsSLIL();
  bool IsSecondPass = dev_rgn->IsSecondPass();
  dev_rgn->SetDepGraph(dev_DDG);
  ((BBWithSpill *)dev_rgn)->SetRegFiles(dev_DDG->getRegFiles());
  dev_noImprovement = 0;
  dev_iterations = 0;
  lowerBoundSchedFound = false;
  // Used to synchronize all launched threads
  auto threadGroup = cg::this_grid();
  // Get RPTarget
  InstCount RPTarget;

  // If in second pass and not using SLIL, set RPTarget
  if (!needsSLIL)
    RPTarget = dev_bestSched->GetSpillCost();
  else
    RPTarget = INT_MAX;
  
  dev_AcoSchdulr->SetGlobalBestStalls(dev_bestSched->GetCrntLngth() - dev_DDG->GetInstCnt());
  // Start ACO
  while (dev_noImprovement < noImprovementMax && !lowerBoundSchedFound) {
    // Reset schedules to post constructor state
    dev_schedules[GLOBALTID]->Initialize();
    dev_AcoSchdulr->FindOneSchedule(RPTarget,
                                    dev_schedules[GLOBALTID]);
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
        InstCount globalStalls = 1 > dev_bestSched->getTotalStalls() ? 1 : dev_bestSched->getTotalStalls();
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
          atomicAdd(&dev_noImprovement, 1);
#endif
        // if a schedule is found with the cost at the lower bound
        // exit the loop after the current iteration is finished
        if ( dev_bestSched && ( dev_bestSched->GetCost() == 0 || ( IsSecondPass && dev_bestSched->GetExecCost() == 0 ) ) ) {
          lowerBoundSchedFound = true;
        }
      } else {
        atomicAdd(&dev_noImprovement, 1);
        if (dev_noImprovement > noImprovementMax)
          break;
      }
    }
    // wait for other blocks to finish before starting next iteration
    threadGroup.sync();
    // make sure no threads reset schedule before above operations complete
    dev_schedules[GLOBALTID]->resetTotalStalls();
    dev_schedules[GLOBALTID]->resetUnnecessaryStalls();
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
  InstCount TargetSC = InitialSchedule ? InitialSchedule->GetSpillCost()
                                        : heuristicSched->GetSpillCost();  

#if USE_ACS
  initialValue_ = 2.0 / ((double)count_ * heuristicCost);
#else
  initialValue_ = (double)NUMTHREADS / heuristicCost;
#endif
  for (int i = 0; i < pheromone_size; i++)
    pheromone_[i] = initialValue_;
  std::cerr << "initialValue_" << initialValue_ << std::endl;
  InstSchedule *bestSchedule = InitialSchedule;

  // check if heuristic schedule is better than the initial
  // schedule passed from the list scheduler
  if (shouldReplaceSchedule(InitialSchedule, heuristicSched,
                                /*IsGlobal=*/true)) {
    bestSchedule = std::move(heuristicSched);
    printf("Heuristic schedule is better\n");
  }
  else {
    bestSchedule = std::move(InitialSchedule);
    printf("Initial schedule is better\n");
  }
  if (bestSchedule) {
    UpdatePheromone(bestSchedule);
  }
  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  InstSchedule *iterationBest = nullptr;
  SetGlobalBestStalls(std::max(1, bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt()));
  
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
    void *dArgs[7];
    dArgs[0] = (void*)&dev_rgn_;
    dArgs[1] = (void*)&dev_DDG_;
    dArgs[2] = (void*)&dev_AcoSchdulr;
    dArgs[3] = (void*)&dev_schedules;
    dArgs[4] = (void*)&dev_bestSched;
    dArgs[5] = (void*)&noImprovementMax;
    dArgs[6] = (void*)&dev_blockBestIndex;
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
        SetGlobalBestStalls(bestSchedule->GetCrntLngth() - dataDepGraph_->GetInstCnt());
        printf("ACO found schedule "
               "cost:%d, rp cost:%d, exec cost: %d, and "
               "iteration:%d"
               " (sched length: %d, abs rp cost: %d, rplb: %d)\n",
               bestSchedule->GetCost(), bestSchedule->GetNormSpillCost(),
               bestSchedule->GetExecCost(), iterations,
               bestSchedule->GetCrntLngth(), bestSchedule->GetSpillCost(),
               rgn_->GetRPCostLwrBound());
#if !RUNTIME_TESTING
          noImprovement = 0;
#else
          // Disable resetting noImp to lock iterations to 10
          noImprovement++;
#endif
        if (bestSchedule && ( bestSchedule->GetCost() == 0 || ( !IsFirst && bestSchedule->GetExecCost() == 0 ) ) )
          break;
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
      pheromone = &Pheromone(i, j);
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

inline void ACOScheduler::UpdateACOReadyList(SchedInstruction *inst) {
  InstCount prdcsrNum, scsrRdyCycle;
  
  #ifdef __CUDA_ARCH__ // device version of function
    // Notify each successor of this instruction that it has been scheduled.
    for (SchedInstruction *crntScsr = inst->GetFrstScsr(&prdcsrNum);
          crntScsr != NULL; crntScsr = inst->GetNxtScsr(&prdcsrNum)) {
        bool wasLastPrdcsr =
            crntScsr->PrdcsrSchduld(prdcsrNum, dev_crntCycleNum_[GLOBALTID], scsrRdyCycle);

        if (wasLastPrdcsr) {
          // If all other predecessors of this successor have been scheduled then
          // we now know in which cycle this successor will become ready.
          HeurType HeurWOLuc = dev_kHelper->computeKey(crntScsr, false);
          dev_readyLs->addInstructionToReadyList(ACOReadyListEntry{crntScsr->GetNum(), scsrRdyCycle, HeurWOLuc, 0});
        }
    }

    // Make sure the scores are valid.  The scheduling of an instruction may
    // have increased another instruction's LUC Score
    PriorityEntry LUCEntry = dev_kHelper->getPriorityEntry(LSH_LUC);
    for (InstCount I = 0; I < dev_readyLs->getReadyListSize(); ++I) {
      //we first get the heuristic without the LUC component, add the LUC
      //LUC component, and then compute the score
      HeurType Heur = *dev_readyLs->getInstHeuristicAtIndex(I);
      InstCount CandidateId = *dev_readyLs->getInstIdAtIndex(I);
      if (LUCEntry.Width) {
        SchedInstruction *ScsrInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType LUCVal = ScsrInst->CmputLastUseCnt();
        LUCVal <<= LUCEntry.Offset;
        Heur &= LUCVal;
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
          HeurType HeurWOLuc = kHelper->computeKey(crntScsr, false);
          readyLs->addInstructionToReadyList(ACOReadyListEntry{crntScsr->GetNum(), scsrRdyCycle, HeurWOLuc, 0});
        }
    }

    // Make sure the scores are valid.  The scheduling of an instruction may
    // have increased another instruction's LUC Score
    PriorityEntry LUCEntry = kHelper->getPriorityEntry(LSH_LUC);
    for (InstCount I = 0; I < readyLs->getReadyListSize(); ++I) {
      //we first get the heuristic without the LUC component, add the LUC
      //LUC component, and then compute the score
      HeurType Heur = *readyLs->getInstHeuristicAtIndex(I);
      InstCount CandidateId = *readyLs->getInstIdAtIndex(I);
      if (LUCEntry.Width) {
        SchedInstruction *ScsrInst = dataDepGraph_->GetInstByIndx(CandidateId);
        HeurType LUCVal = ScsrInst->CmputLastUseCnt();
        LUCVal <<= LUCEntry.Offset;
        Heur &= LUCVal;
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
  
  // Alloc dev array for readyLs;
  readyLs->AllocDevArraysForParallelACO(NUMTHREADS);
  // Alloc dev arrays for MaxScoringInst
  memSize = sizeof(InstCount) * NUMTHREADS;
  gpuErrchk(cudaMalloc(&dev_MaxScoringInst, memSize));
  // Alloc dev array for avlblSlotsInCrntCycle_
  memSize = sizeof(int16_t *) * NUMTHREADS;
  gpuErrchk(cudaMallocManaged(&dev_avlblSlotsInCrntCycle_, memSize));
  // Alloc dev arrays of avlblSlotsInCrntCycle_ for each thread
  memSize = sizeof(int16_t) * issuTypeCnt_;
  for (int i = 0; i < NUMTHREADS; i++) {
    gpuErrchk(cudaMalloc(&dev_avlblSlotsInCrntCycle_[i], memSize));
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
  // copy readyLs
  memSize = sizeof(ACOReadyList);
  gpuErrchk(cudaMalloc(&dev_ACOSchedulr->dev_readyLs, memSize));
  gpuErrchk(cudaMemcpy(dev_ACOSchedulr->dev_readyLs, readyLs, memSize,
		       cudaMemcpyHostToDevice));
  // copy khelper
  memSize = sizeof(KeysHelper);
  gpuErrchk(cudaMalloc(&dev_ACOSchedulr->dev_kHelper, memSize));
  gpuErrchk(cudaMemcpy(dev_ACOSchedulr->dev_kHelper, kHelper, memSize,
		       cudaMemcpyHostToDevice));
  // make sure cudaMallocManaged memory is copied to device before kernel start
  memSize = sizeof(int16_t *) * NUMTHREADS;
  gpuErrchk(cudaMemPrefetchAsync(dev_avlblSlotsInCrntCycle_, memSize, 0));
  memSize = sizeof(ReserveSlot *) * NUMTHREADS;
  gpuErrchk(cudaMemPrefetchAsync(dev_rsrvSlots_, memSize, 0));
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
    cudaFree(dev_avlblSlotsInCrntCycle_[i]);
    cudaFree(dev_rsrvSlots_[i]);
  }
  cudaFree(dev_MaxScoringInst);
  readyLs->FreeDevicePointers();
  cudaFree(dev_avlblSlotsInCrntCycle_);
  cudaFree(dev_rsrvSlots_);
  cudaFree(dev_rsrvSlotCnt_);
  cudaFree(dev_readyLs);
  cudaFree(dev_kHelper);
  cudaFree(pheromone_.elmnts_);
}
