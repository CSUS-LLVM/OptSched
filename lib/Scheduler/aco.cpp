#include "opt-sched/Scheduler/aco.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/ready_list.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/sched_region.h"
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
#define TWO_STEP 1
#define MIN_DEPOSITION 1
#define MAX_DEPOSITION 6
#define MAX_DEPOSITION_MINUS_MIN (MAX_DEPOSITION - MIN_DEPOSITION)
#define ACO_SCHED_STALLS 1

#define SCALE (2<<24)

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
                           bool IsPostBB)
    : ConstrainedScheduler(dataDepGraph, machineModel, upperBound) {
  VrfySched_ = vrfySched;
  this->IsPostBB = IsPostBB;
  prirts_ = priorities;
  rdyLst_ = new ReadyList(dataDepGraph_, priorities);
  count_ = dataDepGraph->GetInstCnt();
  Config &schedIni = SchedulerOptions::getInstance();

  use_fixed_bias = schedIni.GetBool("ACO_USE_FIXED_BIAS");
  use_tournament = schedIni.GetBool("ACO_TOURNAMENT");
  bias_ratio = schedIni.GetFloat("ACO_BIAS_RATIO");
  local_decay = schedIni.GetFloat("ACO_LOCAL_DECAY");
  decay_factor = schedIni.GetFloat("ACO_DECAY_FACTOR");
  ants_per_iteration1p = schedIni.GetInt("ACO_ANT_PER_ITERATION");
  ants_per_iteration2p = schedIni.GetInt("ACO2P_ANT_PER_ITERATION", ants_per_iteration1p);
  ants_per_iteration = ants_per_iteration1p;
  print_aco_trace = schedIni.GetBool("ACO_TRACE");
  IsTwoPassEn = schedIni.GetBool("USE_TWO_PASS");
  DCFOption = ParseDCFOpt(schedIni.GetString("ACO_DUAL_COST_FN_ENABLE", "OFF"));

  // pheromone Graph Debugging start
  std::string TgtRgns = schedIni.GetString("ACO_DBG_REGIONS");
  OutPath = schedIni.GetString("ACO_DBG_REGIONS_OUT_PATH");
  if (TgtRgns != "NONE") {
    std::size_t StartIdx = 0;
    std::size_t SepIdx = TgtRgns.find("|");
    while (SepIdx != std::string::npos) {
      DbgRgns.insert(TgtRgns.substr(StartIdx, SepIdx - StartIdx));
      StartIdx = SepIdx + 1;
      SepIdx = TgtRgns.find("|", StartIdx);
    }
  }
  IsDbg = DbgRgns.count(dataDepGraph_->GetDagID());
  // pheromone Graph Debugging end

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
pheromone_t &ACOScheduler::Pheromone(SchedInstruction *from,
                                     SchedInstruction *to) {
  assert(to != NULL);
  int fromNum = -1;
  if (from != NULL)
    fromNum = from->GetNum();
  return Pheromone(fromNum, to->GetNum());
}

pheromone_t &ACOScheduler::Pheromone(InstCount from, InstCount to) {
  int row = 0;
  if (from != -1)
    row = from + 1;
  return pheromone_[(row * count_) + to];
}


double ACOScheduler::Score(SchedInstruction *from, Choice choice) {
// tuneable heuristic importance is temporarily disabled
//  return Pheromone(from, choice.inst) *
//         pow(choice.heuristic, heuristicImportance_);
  double hf = heuristicImportance_ ? choice.heuristic : 1.0;
  return Pheromone(from, choice.inst) * hf;
}

#define DBG_SRS 0

bool ACOScheduler::shouldReplaceSchedule(InstSchedule *OldSched,
                                         InstSchedule *NewSched,
                                         bool IsGlobal) {
#if DBG_SRS
  std::string CmpLn = "SRS/";
  CmpLn += IsGlobal ? "g/" : "";
#endif // DBG_SRS

  // return true if the old schedule is null (eg:there is no old schedule)
  // return false if the new schedule is is NULL
  if (!OldSched) {
#if DBG_SRS
    Logger::Info("SRS/Old:null, New:%d",
                 !NewSched ? -1
                           : ((!IsTwoPassEn) ? NewSched->GetCost()
                                             : NewSched->GetNormSpillCost()));
#endif // DBG_SRS
    return true;
  } else if (!NewSched) {
    // not likely to happen
#if DBG_SRS
    Logger::Info("SRS/Old:%d, New:null", (!IsTwoPassEn)
                                             ? OldSched->GetCost()
                                             : OldSched->GetNormSpillCost());
#endif // DBG_SRS
    return false;
  }

  // if it is the 1st pass return the cost comparison
  // if it is the 2nd pass return true if the RP cost and ILP cost is less
  if (!IsTwoPassEn || !rgn_->IsSecondPass()) {
    InstCount NewCost =
        (!IsTwoPassEn) ? NewSched->GetCost() : NewSched->GetNormSpillCost();
    InstCount OldCost =
        (!IsTwoPassEn) ? OldSched->GetCost() : OldSched->GetNormSpillCost();
#if DBG_SRS
    CmpLn +=
        "Old:" + std::to_string(OldCost) + ", New:" + std::to_string(NewCost);
#endif // DBG_SRS

    if (NewCost < OldCost) {
#if DBG_SRS
      Logger::Info(CmpLn.c_str());
#endif // DBG_SRS
      return true;
    } else if (NewCost == OldCost &&
               ((DCFOption == DCF_OPT::GLOBAL_ONLY && IsGlobal) ||
                DCFOption == DCF_OPT::GLOBAL_AND_TIGHTEN ||
                DCFOption == DCF_OPT::GLOBAL_AND_ITERATION)) {
      InstCount NewDCFCost = NewSched->GetExtraSpillCost(DCFCostFn);
      InstCount OldDCFCost = OldSched->GetExtraSpillCost(DCFCostFn);

#if DBG_SRS
      CmpLn += ", OldDCF:" + std::to_string(OldDCFCost) +
               ", NewDCF:" + std::to_string(NewDCFCost);
      Logger::Info(CmpLn.c_str());
#endif // DBG_SRS
      return (NewDCFCost < OldDCFCost);

    } else {
#if DBG_SRS
      Logger::Info(CmpLn.c_str());
#endif // DBG_SRS
      return false;
    }
  } else {
    InstCount NewCost = NewSched->GetExecCost();
    InstCount OldCost = OldSched->GetExecCost();
    InstCount NewSpillCost = NewSched->GetNormSpillCost();
    InstCount OldSpillCost = OldSched->GetNormSpillCost();
#if DBG_SRS
    Logger::Info("SRS2P/%sOld:%d,New:%d,OldNSC:%d,NewNSC:%d", IsGlobal ? "g/" : "",
                 OldCost, NewCost, OldSpillCost, NewSpillCost);
#endif // DBG_SRS
    return (NewCost < OldCost && NewSpillCost <= OldSpillCost) || NewSpillCost < OldSpillCost;
  }
}

DCF_OPT ACOScheduler::ParseDCFOpt(const std::string &opt) {
  if (opt == "OFF")
    return DCF_OPT::OFF;
  else if (opt == "GLOBAL_ONLY")
    return DCF_OPT::GLOBAL_ONLY;
  else if (opt == "GLOBAL_AND_TIGHTEN")
    return DCF_OPT::GLOBAL_AND_TIGHTEN;
  else if (opt == "GLOBAL_AND_ITERATION")
    return DCF_OPT::GLOBAL_AND_ITERATION;

  llvm::report_fatal_error("Unrecognized Dual Cost Function Option: " + opt,
                           false);
}

Choice ACOScheduler::SelectInstruction(const llvm::ArrayRef<Choice> &ready,
                                       SchedInstruction *lastInst) {
#if TWO_STEP
  double choose_best_chance;
  if (use_fixed_bias)
    choose_best_chance = fmax(0, 1 - (double)fixed_bias / count_);
  else
    choose_best_chance = bias_ratio;

  if (RandDouble(0, 1) < choose_best_chance) {
    if (print_aco_trace)
      std::cerr << "choose_best, use fixed bias: " << use_fixed_bias << "\n";
    pheromone_t max = -1;
    Choice maxChoice;
    for (auto &choice : ready) {
      if (Score(lastInst, choice) > max) {
        max = Score(lastInst, choice);
        maxChoice = choice;
      }
    }
    return maxChoice;
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
      std::cerr << "tournament Start \n";
      std::cerr << "array_size:" << POPULATION_SIZE << "\n";
      std::cerr << "r:\t" << r_pos << "\n";
      std::cerr << "s:\t" << s_pos << "\n";
      //        std::cerr<<"t:\t"<<t_pos<<"\n";

      std::cerr << "Score r" << Score(lastInst, r) << "\n";
      std::cerr << "Score s" << Score(lastInst, s) << "\n";
      //         std::cerr<<"Score t"<<Score(lastInst, t)<<"\n";
    }
    if (Score(lastInst, r) >=
        Score(lastInst, s)) //&& Score(lastInst, r) >= Score(lastInst, t))
      return r;
    //     else if (Score(lastInst, s) >= Score(lastInst, r) && Score(lastInst,
    //     s) >= Score(lastInst, t))
    //         return s;
    else
      return s;
  }
  pheromone_t sum = 0;
  for (auto choice : ready)
    sum += Score(lastInst, choice);
  pheromone_t point = RandDouble(0, sum);
  for (auto choice : ready) {
    point -= Score(lastInst, choice);
    if (point <= 0)
      return choice;
  }
  std::cerr << "returning last instruction" << std::endl;
  assert(point < 0.001); // floats should not be this inaccurate
  return ready.back();
}

std::unique_ptr<InstSchedule> ACOScheduler::FindOneSchedule(InstCount TargetRPCost) {
  SchedInstruction *lastInst = NULL;
  std::unique_ptr<InstSchedule> schedule =
      llvm::make_unique<InstSchedule>(machMdl_, dataDepGraph_, true);
  InstCount maxPriority = rdyLst_->MaxPriority();
  if (maxPriority == 0)
    maxPriority = 1; // divide by 0 is bad
  Initialize_();
  rgn_->InitForSchdulng();

  SchedInstruction *waitFor = NULL;
  InstCount waitUntil = 0;
  double maxPriorityInv = 1 / maxPriority;
  llvm::SmallVector<Choice, 0> ready;
  while (!IsSchedComplete_()) {
    UpdtRdyLst_(crntCycleNum_, crntSlotNum_);

    // there are two steps to scheduling an instruction:
    // 1)Select the instruction(if we are not waiting on another instruction)
    SchedInstruction *inst = NULL;
    if (!waitFor) {
      // if we have not already committed to schedule an instruction
      // next then pick one. First add ready instructions.  Including
      //"illegal" e.g. blocked instructions

      // convert the ready list from a custom priority queue to a std::vector,
      // much nicer for this particular scheduler
      ready.reserve(rdyLst_->GetInstCnt());
      unsigned long heuristic;
      SchedInstruction *rInst = rdyLst_->GetNextPriorityInst(heuristic);
      while (rInst != NULL) {
        if (ACO_SCHED_STALLS || ChkInstLglty_(rInst)) {
          Choice c;
          c.inst = rInst;
          c.heuristic = (double)heuristic * maxPriorityInv + 1;
          c.readyOn = 0;
          ready.push_back(c);
          if (IsDbg && lastInst)
            LastHeu[std::make_pair(lastInst->GetNum(), rInst->GetNum())] =
                c.heuristic;
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
        LinkedList<SchedInstruction> *futureReady =
            frstRdyLstPerCycle_[crntCycleNum_ + fCycle];
        if (!futureReady)
          continue;

        for (SchedInstruction *fIns = futureReady->GetFrstElmnt(); fIns;
             fIns = futureReady->GetNxtElmnt()) {
          bool changed;
          unsigned long heuristic = rdyLst_->CmputKey_(fIns, false, changed);
          Choice c;
          c.inst = fIns;
          c.heuristic = (double)heuristic * maxPriorityInv + 1;
          c.readyOn = crntCycleNum_ + fCycle;
          ready.push_back(c);
          if (IsDbg && lastInst)
            LastHeu[std::make_pair(lastInst->GetNum(), fIns->GetNum())] =
                c.heuristic;
        }
        futureReady->ResetIterator();
      }
#endif

      if (!ready.empty()) {
        Choice Sel = SelectInstruction(ready, lastInst);
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
        *pheromone =
            (1 - local_decay) * *pheromone + local_decay * initialValue_;
#endif
        if (IsDbg && lastInst != NULL) {
          AntEdges.insert(std::make_pair(lastInst->GetNum(), inst->GetNum()));
          CrntAntEdges.insert(
              std::make_pair(lastInst->GetNum(), inst->GetNum()));
        }
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
      DoRsrvSlots_(inst);
      // this is annoying
      SchedInstruction *blah = rdyLst_->GetNextPriorityInst();
      while (blah != NULL && blah != inst) {
        blah = rdyLst_->GetNextPriorityInst();
      }
      if (blah == inst)
        rdyLst_->RemoveNextPriorityInst();
      UpdtSlotAvlblty_(inst);

#define STOP_CONSTRUCTION_IF_INFEASIBLE 1

//this is a bugged compile time optimization.  It causes crashes.
#if STOP_CONSTRUCTION_IF_INFEASIBLE
      if (rgn_->getUnnormalizedIncrementalRPCost() > TargetRPCost) {
        delete rdyLst_;
        rdyLst_ = new ReadyList(dataDepGraph_, prirts_);
        return nullptr;
      }
#endif

    }
    /* Logger::Info("Chose instruction %d (for some reason)", instNum); */
    schedule->AppendInst(instNum);
    if (MovToNxtSlot_(inst))
      InitNewCycle_();
    rdyLst_->ResetIterator();
    ready.clear();

  }
  rgn_->UpdateScheduleCost(schedule.get());
  return schedule;
}

FUNC_RESULT ACOScheduler::FindSchedule(InstSchedule *schedule_out,
                                       SchedRegion *region) {
  rgn_ = region;

  // get settings
  Config &schedIni = SchedulerOptions::getInstance();
  bool IsFirst = !rgn_->IsSecondPass();
  heuristicImportance_ = schedIni.GetInt(
      IsFirst ? "ACO_HEURISTIC_IMPORTANCE" : "ACO2P_HEURISTIC_IMPORTANCE");
  fixed_bias = schedIni.GetInt(IsFirst ? "ACO_FIXED_BIAS" : "ACO2P_FIXED_BIAS");
  ants_per_iteration = IsFirst ? ants_per_iteration1p : ants_per_iteration2p;
  noImprovementMax = schedIni.GetInt(IsFirst ? "ACO_STOP_ITERATIONS"
                                             : "ACO2P_STOP_ITERATIONS");
  Logger::Info("ants/it:%d,stop_iter:%d",ants_per_iteration,noImprovementMax);
  if (DCFOption != DCF_OPT::OFF) {
    std::string DcfFnString =
        schedIni.GetString(IsFirst ? "ACO_DUAL_COST_FN" : "ACO2P_DUAL_COST_FN");
    if (DcfFnString != "NONE")
      DCFCostFn = ParseSCFName(DcfFnString);
    else
      DCFOption = DCF_OPT::OFF;
  }

  // compute the relative maximum score inverse
  ScRelMax = rgn_->GetHeuristicCost();

  // initialize pheromone
  // for this, we need the cost of the pure heuristic schedule
  int pheromone_size = (count_ + 1) * count_;
  for (int i = 0; i < pheromone_size; i++)
    pheromone_[i] = 1;
  initialValue_ = 1;
  InstCount MaxRPTarget = std::numeric_limits<InstCount>::max();
  std::unique_ptr<InstSchedule> heuristicSched = FindOneSchedule(MaxRPTarget);
  InstCount heuristicCost =
      heuristicSched->GetCost() + 1; // prevent divide by zero
  InstCount InitialCost = InitialSchedule ? InitialSchedule->GetCost() : 0;

#if USE_ACS
  initialValue_ = 2.0 / ((double)count_ * heuristicCost);
#else
  initialValue_ = (double)ants_per_iteration / heuristicCost;
#endif
  for (int i = 0; i < pheromone_size; i++)
    pheromone_[i] = initialValue_;
  std::cerr << "initialValue_" << initialValue_ << std::endl;

  std::unique_ptr<InstSchedule> bestSchedule = std::move(InitialSchedule);
  if (bestSchedule) {
    UpdatePheromone(bestSchedule.get());
  }
  writePheromoneGraph("initial");

  int noImprovement = 0; // how many iterations with no improvement
  int iterations = 0;
  while (true) {
    std::unique_ptr<InstSchedule> iterationBest;
    for (int i = 0; i < ants_per_iteration; i++) {
      CrntAntEdges.clear();
      std::unique_ptr<InstSchedule> schedule = FindOneSchedule(i && iterationBest && rgn_->GetSpillCostFunc() != SCF_SLIL ? iterationBest->GetNormSpillCost() : MaxRPTarget);
      if (print_aco_trace)
        PrintSchedule(schedule.get());
      ++localCmp;
      if (iterationBest && bestSchedule && !(!IsFirst && iterationBest->GetNormSpillCost() <= bestSchedule->GetNormSpillCost()))
        ++localCmpRej;
      if (shouldReplaceSchedule(iterationBest.get(), schedule.get(),
                                /*IsGlobal=*/false)) {
        iterationBest = std::move(schedule);
        if (IsDbg)
          IterAntEdges = CrntAntEdges;
      }
    }
    ++globalCmp;
    if (!IsFirst && iterationBest->GetNormSpillCost() <= bestSchedule->GetNormSpillCost()) {
      UpdatePheromone(iterationBest.get());
    }
    else ++globalCmpRej;
    /* PrintSchedule(iterationBest); */
    /* std::cout << iterationBest->GetCost() << std::endl; */
    // TODO DRY
    if (shouldReplaceSchedule(bestSchedule.get(), iterationBest.get(),
                              /*IsGlobal=*/true)) {
      bestSchedule = std::move(iterationBest);
      Logger::Info("ACO found schedule with spill cost %d",
                   bestSchedule->GetCost());
      Logger::Info("ACO found schedule "
                   "cost:%d, rp cost:%d, exec cost: %d, and "
                   "iteration:%d"
                   " (sched length: %d, abs rp cost: %d, rplb: %d)",
                   bestSchedule->GetCost(), bestSchedule->GetNormSpillCost(),
                   bestSchedule->GetExecCost(), iterations,
                   bestSchedule->GetCrntLngth(), bestSchedule->GetSpillCost(),
                   rgn_->GetRPCostLwrBound());
      if (IsDbg)
        BestAntEdges = IterAntEdges;

      noImprovement = 0;
      if (bestSchedule && bestSchedule->GetCost() == 0)
        break;
    } else {
      noImprovement++;
      /* if (*iterationBest == *bestSchedule) */
      /*   std::cout << "same" << std::endl; */
      if (noImprovement > noImprovementMax)
        break;
    }

    writePheromoneGraph("iteration" + std::to_string(iterations));
    iterations++;
  }

  Logger::Info("localCmp:%d,localCmpRej:%d,globalCmp:%d,globalCmpRej:%d", localCmp, localCmpRej, globalCmp, globalCmpRej);

  Logger::Event(IsPostBB ? "AcoPostSchedComplete" : "ACOSchedComplete", "cost",
                bestSchedule->GetCost(), "iterations", iterations,
                "improvement", InitialCost - bestSchedule->GetCost());
  PrintSchedule(bestSchedule.get());
  schedule_out->Copy(bestSchedule.release());

  Logger::Info("ACO finished after %d iterations", iterations);
  return RES_SUCCESS;
}

void ACOScheduler::UpdatePheromone(InstSchedule *schedule) {
  // I wish InstSchedule allowed you to just iterate over it, but it's got this
  // cycle and slot thing which needs to be accounted for
  InstCount instNum, cycleNum, slotNum;
  instNum = schedule->GetFrstInst(cycleNum, slotNum);

  SchedInstruction *lastInst = NULL;
  pheromone_t portion = schedule->GetCost() / (ScRelMax * 1.5);
  pheromone_t deposition =
      fmax((1 - portion) * MAX_DEPOSITION_MINUS_MIN, 0) + MIN_DEPOSITION;

  while (instNum != INVALID_VALUE) {
    SchedInstruction *inst = dataDepGraph_->GetInstByIndx(instNum);

    pheromone_t *pheromone = &Pheromone(lastInst, inst);
#if USE_ACS
    // ACS update rule includes decay
    // only the arcs on the current solution are decayed
    *pheromone = (1 - decay_factor) * *pheromone +
                 decay_factor / (schedule->GetCost() + 1);
#else
    *pheromone += deposition;
#endif
    lastInst = inst;

    instNum = schedule->GetNxtInst(cycleNum, slotNum);
  }
  schedule->ResetInstIter();

#if !USE_ACS
  // decay pheromone
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      pheromone_t &PhPtr = Pheromone(i, j);
      PhPtr *= (1 - decay_factor);
      PhPtr = fmax(1, fmin(8, PhPtr));
    }
  }
#endif
  if (print_aco_trace)
    PrintPheromone();
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

void ACOScheduler::PrintPheromone() {
  for (int i = 0; i < count_; i++) {
    for (int j = 0; j < count_; j++) {
      std::cerr << std::scientific << std::setprecision(8) << Pheromone(i, j)
                << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

#ifndef NDEBUG
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static void PrintInstruction(SchedInstruction *inst) {
  std::cerr << std::setw(2) << inst->GetNum() << " ";
  std::cerr << std::setw(20) << std::left << inst->GetOpCode();

  std::cerr << " defs ";
  for (auto def : llvm::enumerate(inst->GetDefs())) {
    if (def.index() != 0)
      std::cerr << ", ";
    std::cerr << def.value()->GetNum() << def.value()->GetType();
  }

  std::cerr << " uses ";
  for (auto use : llvm::enumerate(inst->GetUses())) {
    if (use.index() != 0)
      std::cerr << ", ";
    std::cerr << use.value()->GetNum() << use.value()->GetType();
  }
  std::cerr << std::endl;
}
#endif

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

void ACOScheduler::setInitialSched(InstSchedule *Sched) {
  if (Sched) {
    InitialSchedule =
        llvm::make_unique<InstSchedule>(machMdl_, dataDepGraph_, VrfySched_);
    InitialSchedule->Copy(Sched);
  }
}

void ACOScheduler::writePheromoneGraph(std::string Stage) {
  if (!IsDbg)
    return;

  std::string FullOutPath =
      OutPath + "/" + dataDepGraph_->GetDagID() + "@" + Stage + ".dot";
  FILE *Out = fopen(FullOutPath.c_str(), "w");
  if (!Out) {
    Logger::Error("Could now open file to write pheromone display at %s."
                  " Skipping.",
                  FullOutPath.c_str());
    return;
  }

  // already added set
  llvm::SetVector<SchedInstruction *> Visited;

  // header for .dot file
  fprintf(Out, "digraph pheromone_matrix {\n");
  fprintf(Out, "label=\"%s@%s\"\n", dataDepGraph_->GetDagID(), Stage.c_str());
  fprintf(Out, "ranksep=1.0\n");

  // find the recursive neighbors
  dataDepGraph_->FindRcrsvNghbrs(DIR_FRWRD);
  dataDepGraph_->FindRcrsvNghbrs(DIR_BKWRD);

  writePGraphRecursive(Out, dataDepGraph_->GetRootInst(), Visited);

  // footer for .dot file
  fprintf(Out, "}\n");
  fclose(Out);

  // wipe out ant edges
  AntEdges.clear();
  CrntAntEdges.clear();
  IterAntEdges.clear();
  LastHeu.clear();
}

std::string ACOScheduler::graphDisplayAnnotation(int Frm, int To) {
  std::pair<InstCount, InstCount> ThisEdge = std::make_pair(Frm, To);

  if (BestAntEdges.count(ThisEdge) && IterAntEdges.count(ThisEdge))
    return "penwidth=2.0 color=\"#00FFFF\"";
  else if (IterAntEdges.count(ThisEdge))
    return "penwidth=2.0 color=\"#00FF00\"";
  else if (BestAntEdges.count(ThisEdge))
    return "penwidth=2.0 color=\"#0000FF\"";
  else if (AntEdges.count(ThisEdge))
    return "color=\"#FF0000\"";
  else
    return "color=\"#000000\"";
}

std::string ACOScheduler::getHeuIfPossible(int Frm, int To) {
  std::pair<InstCount, InstCount> ThisEdge = std::make_pair(Frm, To);
  if (LastHeu.count(ThisEdge)) {
    return std::string("|") + std::to_string(LastHeu[ThisEdge]);
  } else
    return "";
}

void ACOScheduler::writePGraphRecursive(
    FILE *Out, SchedInstruction *Ins,
    llvm::SetVector<SchedInstruction *> &Visited) {
  InstCount I = Ins->GetNum();

  // do not add edges out twice
  if (Visited.count(Ins))
    return;
  // add self to set so edges are not double counted
  Visited.insert(Ins);

  // create edges for other orderings
  for (SchedInstruction *VIns : Visited) {
    if (!(Ins->IsRcrsvPrdcsr(VIns) || Ins->IsRcrsvScsr(VIns))) {
      InstCount VVtx = VIns->GetNum();
      std::string ToAnno = graphDisplayAnnotation(I, VVtx);
      std::string FrmAnno = graphDisplayAnnotation(VVtx, I);
      std::string ToHeu = getHeuIfPossible(I, VVtx);
      std::string FrmHeu = getHeuIfPossible(VVtx, I);

      fprintf(Out,
              "\t%d -> %d [label=\"%f%s\" constraint=false style=dotted %s];\n",
              I, VVtx, Pheromone(I, VVtx), ToHeu.c_str(), ToAnno.c_str());
      fprintf(Out,
              "\t%d -> %d [label=\"%f%s\" constraint=false style=dotted %s];\n",
              VVtx, I, Pheromone(VVtx, I), FrmHeu.c_str(), FrmAnno.c_str());
    }
  }

  // add edges to children
  for (SchedInstruction *Child = Ins->GetFrstScsr(); Child != NULL;
       Child = Ins->GetNxtScsr()) {
    InstCount J = Child->GetNum();
    std::string SAnno = graphDisplayAnnotation(I, J);
    std::string SHeu = getHeuIfPossible(I, J);
    fprintf(Out, "\t%d -> %d [label=\"%f%s\" %s];\n", I, J, Pheromone(I, J),
            SHeu.c_str(), SAnno.c_str());
    writePGraphRecursive(Out, Child, Visited);
  }
}
