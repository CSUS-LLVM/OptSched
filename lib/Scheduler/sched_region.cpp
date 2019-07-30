#include <memory>

#include <algorithm>
#include <memory>
#include <utility>

#include "opt-sched/Scheduler/bb_spill.h"
#include "opt-sched/Scheduler/config.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/list_sched.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/random.h"
#include "opt-sched/Scheduler/reg_alloc.h"
#include "opt-sched/Scheduler/relaxed_sched.h"
#include "opt-sched/Scheduler/sched_region.h"
#include "opt-sched/Scheduler/stats.h"
#include "opt-sched/Scheduler/utilities.h"

extern bool OPTSCHED_gPrintSpills;

using namespace llvm::opt_sched;

SchedRegion::SchedRegion(MachineModel *machMdl, DataDepGraph *dataDepGraph,
                         long rgnNum, int16_t sigHashSize, LB_ALG lbAlg,
                         SchedPriorities hurstcPrirts,
                         SchedPriorities enumPrirts, bool vrfySched,
                         Pruning PruningStrategy, SchedulerType HeurSchedType) {
  machMdl_ = machMdl;
  dataDepGraph_ = dataDepGraph;
  rgnNum_ = rgnNum;
  sigHashSize_ = sigHashSize;
  lbAlg_ = lbAlg;
  hurstcPrirts_ = hurstcPrirts;
  enumPrirts_ = enumPrirts;
  vrfySched_ = vrfySched;
  prune_ = PruningStrategy;
  HeurSchedType_ = HeurSchedType;
  isSecondPass = false;

  totalSimSpills_ = INVALID_VALUE;
  bestCost_ = 0;
  bestSchedLngth_ = 0;
  hurstcCost_ = 0;
  hurstcSchedLngth_ = 0;
  enumCrntSched_ = nullptr;
  enumBestSched_ = nullptr;

  schedLwrBound_ = 0;
  schedUprBound_ = INVALID_VALUE;

  instCnt_ = dataDepGraph_->GetInstCnt();

  needTrnstvClsr_ = false;
}

void SchedRegion::UseFileBounds_() {
  InstCount fileLwrBound, fileUprBound;

  dataDepGraph_->UseFileBounds();
  dataDepGraph_->GetFileSchedBounds(fileLwrBound, fileUprBound);
  assert(fileLwrBound >= schedLwrBound_);
  schedLwrBound_ = fileLwrBound;
}

InstSchedule *SchedRegion::AllocNewSched_() {
  InstSchedule *newSched =
      new InstSchedule(machMdl_, dataDepGraph_, vrfySched_);
  if (newSched == nullptr)
    Logger::Fatal("Out of memory.");
  return newSched;
}

void SchedRegion::CmputAbslutUprBound_() {
  abslutSchedUprBound_ = dataDepGraph_->GetAbslutSchedUprBound();
}

FUNC_RESULT SchedRegion::FindOptimalSchedule(
    Milliseconds rgnTimeout, Milliseconds lngthTimeout, bool &isLstOptml,
    InstCount &bestCost, InstCount &bestSchedLngth, InstCount &hurstcCost,
    InstCount &hurstcSchedLngth, InstSchedule *&bestSched, bool filterByPerp,
    const BLOCKS_TO_KEEP blocksToKeep) {
  ConstrainedScheduler *lstSchdulr;
  InstSchedule *lstSched = nullptr;
  FUNC_RESULT rslt = RES_SUCCESS;
  Milliseconds hurstcTime = 0;
  Milliseconds boundTime = 0;
  Milliseconds enumTime = 0;
  Milliseconds vrfyTime = 0;

  enumCrntSched_ = nullptr;
  enumBestSched_ = nullptr;
  bestSched = bestSched_ = nullptr;

  Logger::Info("---------------------------------------------------------------"
               "------------");
  Logger::Info("Processing DAG %s with %d insts and max latency %d.",
               dataDepGraph_->GetDagID(), dataDepGraph_->GetInstCnt(),
               dataDepGraph_->GetMaxLtncy());

  stats::problemSize.Record(dataDepGraph_->GetInstCnt());

  const auto *GraphTransformations = dataDepGraph_->GetGraphTrans();
  if (rgnTimeout > 0 || GraphTransformations->size() > 0 ||
      spillCostFunc_ == SCF_SLIL)
    needTrnstvClsr_ = true;

  rslt = dataDepGraph_->SetupForSchdulng(needTrnstvClsr_);
  if (rslt != RES_SUCCESS) {
    Logger::Info("Invalid input DAG");
    return rslt;
  }

  // Apply graph transformations
  for (auto &GT : *GraphTransformations) {
    rslt = GT->ApplyTrans();

    if (rslt != RES_SUCCESS)
      return rslt;

    // Update graph after each transformation
    rslt = dataDepGraph_->UpdateSetupForSchdulng(needTrnstvClsr_);
    if (rslt != RES_SUCCESS) {
      Logger::Info("Invalid DAG after graph transformations");
      return rslt;
    }
  }

  SetupForSchdulng_();
  CmputAbslutUprBound_();
  schedLwrBound_ = dataDepGraph_->GetSchedLwrBound();
  Milliseconds hurstcStart = Utilities::GetProcessorTime();
  lstSched = new InstSchedule(machMdl_, dataDepGraph_, vrfySched_);
  if (lstSched == nullptr)
    Logger::Fatal("Out of memory.");

  lstSchdulr = AllocHeuristicScheduler_();

  // Step #1: Find the heuristic schedule.
  rslt = lstSchdulr->FindSchedule(lstSched, this);

  if (rslt != RES_SUCCESS) {
    Logger::Info("List scheduling failed");
    delete lstSchdulr;
    delete lstSched;
    return rslt;
  }

  hurstcTime = Utilities::GetProcessorTime() - hurstcStart;
  stats::heuristicTime.Record(hurstcTime);
  if (hurstcTime > 0)
    Logger::Info("Heuristic_Time %d", hurstcTime);

#ifdef IS_DEBUG_SLIL_PRINTOUT
  if (OPTSCHED_gPrintSpills) {
    const auto &slilVector = this->GetSLIL_();
    for (int j = 0; j < slilVector.size(); j++) {
      Logger::Info(
          "SLIL after Heuristic Scheduler for dag %s Type %d %s is %d.",
          dataDepGraph_->GetDagID(), j, machMdl_->GetRegTypeName(j).c_str(),
          slilVector[j]);
    }
  }
#endif

  Milliseconds boundStart = Utilities::GetProcessorTime();
  hurstcSchedLngth_ = lstSched->GetCrntLngth();
  bestSchedLngth_ = hurstcSchedLngth_;
  assert(bestSchedLngth_ >= schedLwrBound_);
  bestSched = bestSched_ = lstSched;

  // Step #2: Compute the lower bounds and cost upper bound.
  if (rgnTimeout == 0)
    costLwrBound_ = CmputCostLwrBound();
  else
    CmputLwrBounds_(false);
  assert(schedLwrBound_ <= lstSched->GetCrntLngth());

  InstCount hurstcExecCost;
  Config &schedIni = SchedulerOptions::getInstance();
  if (!schedIni.GetBool("USE_ACO", false)) {
    CmputNormCost_(lstSched, CCM_DYNMC, hurstcExecCost, true);
  } else {
    CmputNormCost_(lstSched, CCM_STTC, hurstcExecCost, false);
  }
  hurstcCost_ = lstSched->GetCost();
  isLstOptml = CmputUprBounds_(lstSched, false);
  boundTime = Utilities::GetProcessorTime() - boundStart;
  stats::boundComputationTime.Record(boundTime);

  FinishHurstc_();

  //  #ifdef IS_DEBUG_SOLN_DETAILS_1
  Logger::Info(
      "The list schedule is of length %d and spill cost %d. Tot cost = %d",
      bestSchedLngth_, lstSched->GetSpillCost(), bestCost_);
  //  #endif

#ifdef IS_DEBUG_PRINT_SCHEDS
  lstSched->Print(Logger::GetLogStream(), "Heuristic");
#endif
#ifdef IS_DEBUG_PRINT_BOUNDS
  dataDepGraph_->PrintLwrBounds(DIR_FRWRD, Logger::GetLogStream(),
                                "CP Lower Bounds");
#endif

  if (rgnTimeout == 0)
    isLstOptml = true;

  // (Chris): If the cost function is SLIL, then the list schedule is considered
  // optimal if PERP is 0.
  if (filterByPerp && !isLstOptml && spillCostFunc_ == SCF_SLIL) {
    const InstCount *regPressures = nullptr;
    auto regTypeCount = lstSched->GetPeakRegPressures(regPressures);
    InstCount sumPerp = 0;
    for (int i = 0; i < regTypeCount; ++i) {
      int perp = regPressures[i] - machMdl_->GetPhysRegCnt(i);
      if (perp > 0)
        sumPerp += perp;
    }
    if (sumPerp == 0) {
      isLstOptml = true;
      Logger::Info("Marking SLIL list schedule as optimal due to zero PERP.");
    }
  }

#if defined(IS_DEBUG_SLIL_OPTIMALITY)
  // (Chris): This code prints a statement when a schedule is SLIL-optimal but
  // not PERP-optimal.
  if (spillCostFunc_ == SCF_SLIL && bestCost_ == 0) {
    const InstCount *regPressures = nullptr;
    auto regTypeCount = lstSched->GetPeakRegPressures(regPressures);
    InstCount sumPerp = 0;
    for (int i = 0; i < regTypeCount; ++i) {
      int perp = regPressures[i] - machMdl_->GetPhysRegCnt(i);
      if (perp > 0)
        sumPerp += perp;
    }
    if (sumPerp > 0) {
      Logger::Info("Dag %s is SLIL optimal but not PERP optimal (PERP=%d).",
                   dataDepGraph_->GetDagID(), sumPerp);
    }
  }
#endif
  if (!EnableEnum_()) {
    delete lstSchdulr;
    return RES_FAIL;
  }

  // Step #3: Find the optimal schedule if the heuristc was not optimal.
  Milliseconds enumStart = Utilities::GetProcessorTime();

#ifdef IS_DEBUG_BOUNDS
  Logger::Info("Sched LB = %d, Sched UB = %d", schedLwrBound_, schedUprBound_);
#endif

  if (!isLstOptml) {
    dataDepGraph_->SetHard(true);
    rslt = Optimize_(enumStart, rgnTimeout, lngthTimeout);
    enumTime = Utilities::GetProcessorTime() - enumStart;

    if (hurstcTime > 0) {
      enumTime /= hurstcTime;
      stats::enumerationToHeuristicTimeRatio.Record(enumTime);
    }

    if (bestCost_ < hurstcCost_) {
      assert(enumBestSched_ != nullptr);
      bestSched = bestSched_ = enumBestSched_;
#ifdef IS_DEBUG_PRINT_SCHEDS
      enumBestSched_->Print(Logger::GetLogStream(), "Optimal");
#endif
    }
  } else if (rgnTimeout == 0) {
    Logger::Info(
        "Bypassing optimal scheduling due to zero time limit with cost %d",
        bestCost_);
  } else {
    Logger::Info("The list schedule of length %d and cost %d is optimal.",
                 bestSchedLngth_, bestCost_);
  }

  if (rgnTimeout != 0) {
    bool optimalSchedule = isLstOptml || (rslt == RES_SUCCESS);
    Logger::Info("Best schedule for DAG %s has cost %d and length %d. The "
                 "schedule is %s",
                 dataDepGraph_->GetDagID(), bestCost_, bestSchedLngth_,
                 optimalSchedule ? "optimal" : "not optimal");
  }

#ifdef IS_DEBUG_PRINT_PERP_AT_EACH_STEP
  Logger::Info("Printing PERP at each step in the schedule.");

  int costSum = 0;
  for (int i = 0; i < dataDepGraph_->GetInstCnt(); ++i) {
    Logger::Info("Cycle: %lu Cost: %lu", i, bestSched_->GetSpillCost(i));
    costSum += bestSched_->GetSpillCost(i);
  }
  Logger::Info("Cost Sum: %lu", costSum);
#endif

  if (SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") != "NO") {
    //#ifdef IS_DEBUG
    RegAlloc_(bestSched, lstSched);
    //#endif
  }

  // FIXME: Unused expression result - is this supposed to do something? Should
  // the result be assigned?
  Utilities::GetProcessorTime() - enumStart;
  stats::enumerationTime.Record(enumTime);

  Milliseconds vrfyStart = Utilities::GetProcessorTime();

  if (vrfySched_) {
    bool isValidSchdul = bestSched->Verify(machMdl_, dataDepGraph_);

    if (!isValidSchdul) {
      stats::invalidSchedules++;
    }
  }

  vrfyTime = Utilities::GetProcessorTime() - vrfyStart;
  stats::verificationTime.Record(vrfyTime);

  InstCount finalLwrBound = costLwrBound_;
  InstCount finalUprBound = costLwrBound_ + bestCost_;
  if (rslt == RES_SUCCESS)
    finalLwrBound = finalUprBound;

  dataDepGraph_->SetFinalBounds(finalLwrBound, finalUprBound);

  FinishOptml_();

  bool tookBest = ChkSchedule_(bestSched, lstSched);
  if (!tookBest) {
    bestCost_ = hurstcCost_;
    bestSchedLngth_ = hurstcSchedLngth_;
  }

  delete lstSchdulr;
  if (bestSched != lstSched)
    delete lstSched;
  if (enumBestSched_ != nullptr && bestSched != enumBestSched_)
    delete enumBestSched_;
  if (enumCrntSched_ != nullptr)
    delete enumCrntSched_;

  bestCost = bestCost_;
  bestSchedLngth = bestSchedLngth_;
  hurstcCost = hurstcCost_;
  hurstcSchedLngth = hurstcSchedLngth_;
  // (Chris): Experimental. Discard the schedule based on sched.ini setting.
  if (spillCostFunc_ == SCF_SLIL) {
    bool optimal = isLstOptml || (rslt == RES_SUCCESS);
    if ((blocksToKeep == BLOCKS_TO_KEEP::ZERO_COST && bestCost != 0) ||
        (blocksToKeep == BLOCKS_TO_KEEP::OPTIMAL && !optimal) ||
        (blocksToKeep == BLOCKS_TO_KEEP::IMPROVED && bestCost >= hurstcCost) ||
        (blocksToKeep == BLOCKS_TO_KEEP::IMPROVED_OR_OPTIMAL &&
         !(optimal || bestCost < hurstcCost))) {
      delete bestSched;
      bestSched = nullptr;
      return rslt;
    }
  }
#if defined(IS_DEBUG_COMPARE_SLIL_BB)
  {
    const auto &status = [&]() {
      switch (rslt) {
      case RES_SUCCESS:
        return "optimal";
      case RES_TIMEOUT:
        return "timeout";
      default:
        return "failed";
      }
    }();
    if (!isLstOptml) {
      Logger::Info("Dag %s %s cost %d time %lld", dataDepGraph_->GetDagID(),
                   status, bestCost_, enumTime);
      Logger::Info("Dag %s %s absolute cost %d time %lld",
                   dataDepGraph_->GetDagID(), status, bestCost_ + costLwrBound_,
                   enumTime);
    }
  }
  {
    if (spillCostFunc_ == SCF_SLIL && rgnTimeout != 0) {
      // costLwrBound_: static lower bound
      // bestCost_: total cost of the best schedule relative to static lower
      // bound

      auto isEnumerated = [&]() { return (!isLstOptml) ? "True" : "False"; }();

      auto isOptimal = [&]() {
        return (isLstOptml || (rslt == RES_SUCCESS)) ? "True" : "False";
      }();

      auto isPerpHigherThanHeuristic = [&]() {
        auto getSumPerp = [&](InstSchedule *sched) {
          const InstCount *regPressures = nullptr;
          auto regTypeCount = sched->GetPeakRegPressures(regPressures);
          InstCount sumPerp = 0;
          for (int i = 0; i < regTypeCount; ++i) {
            int perp = regPressures[i] - machMdl_->GetPhysRegCnt(i);
            if (perp > 0)
              sumPerp += perp;
          }
          return sumPerp;
        };

        if (lstSched == bestSched)
          return "False";

        auto heuristicPerp = getSumPerp(lstSched);
        auto bestPerp = getSumPerp(bestSched);

        return (bestPerp > heuristicPerp) ? "True" : "False";
      }();

      Logger::Info("SLIL stats: DAG %s static LB %d gap size %d enumerated %s "
                   "optimal %s PERP higher %s",
                   dataDepGraph_->GetDagID(), costLwrBound_, bestCost_,
                   isEnumerated, isOptimal, isPerpHigherThanHeuristic);
    }
  }
#endif
#if defined(IS_DEBUG_FINAL_SPILL_COST)
  // (Chris): Unconditionally Print out the spill cost of the final schedule.
  // This makes it easy to compare results.
  Logger::Info("Final spill cost is %d for DAG %s.", bestSched_->GetSpillCost(),
               dataDepGraph_->GetDagID());
#endif
#if defined(IS_DEBUG_PRINT_PEAK_FOR_ENUMERATED)
  if (!isLstOptml) {
    InstCount maxSpillCost = 0;
    for (int i = 0; i < dataDepGraph_->GetInstCnt(); ++i) {
      if (bestSched->GetSpillCost(i) > maxSpillCost)
        maxSpillCost = bestSched->GetSpillCost(i);
    }
    Logger::Info("DAG %s PEAK %d", dataDepGraph_->GetDagID(), maxSpillCost);
  }
#endif
  return rslt;
}

FUNC_RESULT SchedRegion::Optimize_(Milliseconds startTime,
                                   Milliseconds rgnTimeout,
                                   Milliseconds lngthTimeout) {
  Enumerator *enumrtr;
  FUNC_RESULT rslt = RES_SUCCESS;

  enumCrntSched_ = AllocNewSched_();
  enumBestSched_ = AllocNewSched_();

  InstCount initCost = bestCost_;
  enumrtr = AllocEnumrtr_(lngthTimeout);
  rslt = Enumerate_(startTime, rgnTimeout, lngthTimeout);

  Milliseconds solnTime = Utilities::GetProcessorTime() - startTime;

#ifdef IS_DEBUG_NODES
  Logger::Info("Examined %lld nodes.", enumrtr->GetNodeCnt());
#endif
  stats::nodeCount.Record(enumrtr->GetNodeCnt());
  stats::solutionTime.Record(solnTime);

  InstCount imprvmnt = initCost - bestCost_;
  if (rslt == RES_SUCCESS) {
    Logger::Info("DAG solved optimally in %lld ms with "
                 "length=%d, spill cost = %d, tot cost = %d, cost imp=%d.",
                 solnTime, bestSchedLngth_, bestSched_->GetSpillCost(),
                 bestCost_, imprvmnt);
    stats::solvedProblemSize.Record(dataDepGraph_->GetInstCnt());
    stats::solutionTimeForSolvedProblems.Record(solnTime);
  } else {
    if (rslt == RES_TIMEOUT) {
      Logger::Info("DAG timed out with "
                   "length=%d, spill cost = %d, tot cost = %d, cost imp=%d.",
                   bestSchedLngth_, bestSched_->GetSpillCost(), bestCost_,
                   imprvmnt);
    }
    stats::unsolvedProblemSize.Record(dataDepGraph_->GetInstCnt());
  }

  return rslt;
}

void SchedRegion::CmputLwrBounds_(bool useFileBounds) {
  RelaxedScheduler *rlxdSchdulr = nullptr;
  RelaxedScheduler *rvrsRlxdSchdulr = nullptr;
  InstCount rlxdUprBound = dataDepGraph_->GetAbslutSchedUprBound();

  switch (lbAlg_) {
  case LBA_LC:
    rlxdSchdulr = new LC_RelaxedScheduler(dataDepGraph_, machMdl_, rlxdUprBound,
                                          DIR_FRWRD);
    rvrsRlxdSchdulr = new LC_RelaxedScheduler(dataDepGraph_, machMdl_,
                                              rlxdUprBound, DIR_BKWRD);
    break;
  case LBA_RJ:
    rlxdSchdulr = new RJ_RelaxedScheduler(dataDepGraph_, machMdl_, rlxdUprBound,
                                          DIR_FRWRD, RST_STTC);
    rvrsRlxdSchdulr = new RJ_RelaxedScheduler(
        dataDepGraph_, machMdl_, rlxdUprBound, DIR_BKWRD, RST_STTC);
    break;
  }

  if (rlxdSchdulr == nullptr || rvrsRlxdSchdulr == nullptr) {
    Logger::Fatal("Out of memory.");
  }

  InstCount frwrdLwrBound = 0;
  InstCount bkwrdLwrBound = 0;
  frwrdLwrBound = rlxdSchdulr->FindSchedule();
  bkwrdLwrBound = rvrsRlxdSchdulr->FindSchedule();
  InstCount rlxdLwrBound = std::max(frwrdLwrBound, bkwrdLwrBound);

  assert(rlxdLwrBound >= schedLwrBound_);

  if (rlxdLwrBound > schedLwrBound_)
    schedLwrBound_ = rlxdLwrBound;

#ifdef IS_DEBUG_PRINT_BOUNDS
  dataDepGraph_->PrintLwrBounds(DIR_FRWRD, Logger::GetLogStream(),
                                "Relaxed Forward Lower Bounds");
  dataDepGraph_->PrintLwrBounds(DIR_BKWRD, Logger::GetLogStream(),
                                "Relaxed Backward Lower Bounds");
#endif

  if (useFileBounds)
    UseFileBounds_();

  costLwrBound_ = CmputCostLwrBound();

  delete rlxdSchdulr;
  delete rvrsRlxdSchdulr;
}

bool SchedRegion::CmputUprBounds_(InstSchedule *lstSched, bool useFileBounds) {
  if (useFileBounds) {
    hurstcCost_ = dataDepGraph_->GetFileCostUprBound();
    hurstcCost_ -= GetCostLwrBound();
  }

  bestCost_ = hurstcCost_;
  bestSchedLngth_ = hurstcSchedLngth_;

  if (bestCost_ == 0) {
    // If the heuristic schedule is optimal, we are done!
    schedUprBound_ = lstSched->GetCrntLngth();
    return true;
  } else if (isSecondPass) {
    // In the second pass, the upper bound is the length of the min-RP schedule
    // that was found in the first pass with stalls inserted.
    schedUprBound_ = lstSched->GetCrntLngth();
    return false;
  } else {
    CmputSchedUprBound_();
    return false;
  }
}

void SchedRegion::UpdateScheduleCost(InstSchedule *schedule) {
  InstCount crntExecCost;
  CmputNormCost_(schedule, CCM_STTC, crntExecCost, false);
  // no need to return anything as all results can be found in the schedule
}

SPILL_COST_FUNCTION SchedRegion::GetSpillCostFunc() { return spillCostFunc_; }

void SchedRegion::HandlEnumrtrRslt_(FUNC_RESULT rslt, InstCount trgtLngth) {
  switch (rslt) {
  case RES_FAIL:
    //    #ifdef IS_DEBUG_ENUM_ITERS
    Logger::Info("No feasible solution of length %d was found.", trgtLngth);
    //    #endif
    break;
  case RES_SUCCESS:
#ifdef IS_DEBUG_ENUM_ITERS
    Logger::Info("Found a feasible solution of length %d.", trgtLngth);
#endif
    break;
  case RES_TIMEOUT:
    //    #ifdef IS_DEBUG_ENUM_ITERS
    Logger::Info("Enumeration timedout at length %d.", trgtLngth);
    //    #endif
    break;
  case RES_ERROR:
    Logger::Info("The processing of DAG \"%s\" was terminated with an error.",
                 dataDepGraph_->GetDagID(), rgnNum_);
    break;
  case RES_END:
    //    #ifdef IS_DEBUG_ENUM_ITERS
    Logger::Info("Enumeration ended at length %d.", trgtLngth);
    //    #endif
    break;
  }
}

void SchedRegion::RegAlloc_(InstSchedule *&bestSched, InstSchedule *&lstSched) {
  std::unique_ptr<LocalRegAlloc> u_regAllocBest = nullptr;
  std::unique_ptr<LocalRegAlloc> u_regAllocList = nullptr;

  if (SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "HEURISTIC" ||
      SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "BOTH" ||
      SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "TAKE_SCHED_WITH_LEAST_SPILLS") {
    // Simulate register allocation using the heuristic schedule.
    u_regAllocList = std::make_unique<LocalRegAlloc>(lstSched, dataDepGraph_);

    u_regAllocList->SetupForRegAlloc();
    u_regAllocList->AllocRegs();

    std::string id(dataDepGraph_->GetDagID());
    std::string heur_ident(" ***heuristic_schedule***");
    std::string ident(id + heur_ident);

    u_regAllocList->PrintSpillInfo(ident.c_str());
  }
  if (SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "BEST" ||
      SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "BOTH" ||
      SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "TAKE_SCHED_WITH_LEAST_SPILLS") {
    // Simulate register allocation using the best schedule.
    u_regAllocBest = std::make_unique<LocalRegAlloc>(bestSched, dataDepGraph_);

    u_regAllocBest->SetupForRegAlloc();
    u_regAllocBest->AllocRegs();

    u_regAllocBest->PrintSpillInfo(dataDepGraph_->GetDagID());
    totalSimSpills_ = u_regAllocBest->GetCost();
  }

  if (SchedulerOptions::getInstance().GetString(
          "SIMULATE_REGISTER_ALLOCATION") == "TAKE_SCHED_WITH_LEAST_SPILLS")
    if (u_regAllocList->GetCost() < u_regAllocBest->GetCost()) {
      bestSched = lstSched;
#ifdef IS_DEBUG
      Logger::Info(
          "Taking list schedule becuase of less spilling with simulated RA.");
#endif
    }
}

void SchedRegion::InitSecondPass() { isSecondPass = true; }
