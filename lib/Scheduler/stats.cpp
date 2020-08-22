#include "opt-sched/Scheduler/stats.h"
#include <iomanip>
#include <limits>

using namespace llvm::opt_sched::stats;

template <class T>
DistributionStat<T>::DistributionStat(const string name) : Stat(name) {
  sum_ = 0;
  count_ = 0;
  min_ = std::numeric_limits<T>::max();
  max_ = std::numeric_limits<T>::min();
}

template <class T> void DistributionStat<T>::Record(T value) {
  count_++;
  sum_ += value;
  if (value < min_)
    min_ = value;
  if (value > max_)
    max_ = value;
}

template <class T> void DistributionStat<T>::Print(std::ostream &out) const {
  out << std::setprecision(4);
  out << name_ << ": ";
  if (count_ == 0) {
    out << "[no records]";
  } else if (count_ == 1) {
    assert(min_ == max_ && max_ == sum_);
    out << sum_ << ".";
  } else {
    double avg = (double)sum_ / count_;
    out << "[" << min_ << ", " << max_ << "]"
        << ", count: " << count_ << ", sum: " << sum_ << ", avg: " << avg
        << ".";
  }
  out << "\n";
}

template <class T> void SetStat<T>::Print(std::ostream &out) const {
  out << name_ << ": ";
  if (values_.size() == 0) {
    out << "[no records]";
  } else {
    for (typename std::set<T>::const_iterator it = values_.begin();
         it != values_.end(); it++) {
      if (it != values_.begin())
        out << ", ";
      out << *it;
    }
  }
  out << "\n";
}

template <class T> void IndexedSetStat<T>::Print(std::ostream &out) const {
  out << name_ << ":";
  if (values_.size() == 0) {
    out << " [no records]\n";
  } else {
    out << "\n";
    typedef typename std::map<string, std::set<T>>::const_iterator MapIter;
    for (MapIter it = values_.begin(); it != values_.end(); it++) {
      out << "  " << it->first << ": ";
      if (it->second.size() == 0) {
        out << "[no records]";
      } else {
        for (typename std::set<T>::const_iterator it2 = it->second.begin();
             it2 != it->second.end(); it2++) {
          if (it2 != it->second.begin())
            out << ", ";
          out << *it2;
        }
      }
      out << "\n";
    }
  }
}

void TimeoutStat::Record(int regionNumber, InstCount instCount, int lowerBound,
                         int upperBound) {
  entries_.push_back(Entry(regionNumber, instCount, lowerBound, upperBound));
}

void TimeoutStat::Print(ostream &out) const {
  out << name_ << ":\n";
  for (std::list<Entry>::const_iterator it = entries_.begin();
       it != entries_.end(); it++) {
    out << "  Region #" << it->regionNumber << " (" << it->instCount
        << " instructions):"
        << " final range = [" << it->lowerBound << ", " << it->upperBound
        << "].\n";
  }
}

template <class T> void IndexedNumericStat<T>::Print(std::ostream &out) const {
  out << std::setprecision(4);
  out << name_ << ":";
  if (values_.size() == 0) {
    out << " [no records]";
  } else {
    out << "\n";
    T total = 0;
    for (typename std::map<string, T>::const_iterator it = values_.begin();
         it != values_.end(); it++) {
      total += it->second;
    }
    for (typename std::map<string, T>::const_iterator it = values_.begin();
         it != values_.end(); it++) {
      out << "  " << it->first << ": " << it->second << " ("
          << (it->second * 100 / (double)total) << "%)\n";
    }
  }
}

namespace llvm {
namespace opt_sched {
namespace stats {

ostream &operator<<(ostream &out, const Stat &stat) {
  stat.Print(out);
  return out;
}

// Make sure we explicitly instantiate the allowed template parameters.
template class DistributionStat<int64_t>;
template class DistributionStat<float>;
template class SetStat<int64_t>;
template class SetStat<float>;
template class IndexedSetStat<int64_t>;
template class IndexedSetStat<float>;

// Definitions of the actual statistical records.
IntDistributionStat nodeCount("Nodes per problem");
IntDistributionStat nodesPerLength("Nodes per length");
IntDistributionStat solutionTime("Solution time");
IntDistributionStat
    solutionTimeForSolvedProblems("Solution time for solved problems");

IntDistributionStat iterations("Iterations");
IntDistributionStat enumerations("Enumerations");
IntDistributionStat lengths("Lengths");

IntDistributionStat feasibleSchedulesPerLength("Feasible schedules per length");
IntDistributionStat improvementsPerLength("Improvements per length");

IntDistributionStat costChecksPerLength("Cost checks per length");
IntDistributionStat costPruningsPerLength("Cost prunings per length");
IntDistributionStat
    dynamicLBIterationsPerPath("Dynamic LB iterations per path");

IntDistributionStat problemSize("Problem size");
IntDistributionStat solvedProblemSize("Solved problem size");
IntDistributionStat unsolvedProblemSize("Unsolved problem size");

IntDistributionStat traceCostLowerBound("Trace cost lower bound");

IntDistributionStat traceHeuristicCost("Trace heuristic cost");
IntDistributionStat traceOptimalCost("Trace optimal cost");

IntDistributionStat
    traceHeuristicScheduleLength("Trace heuristic schedule length");
IntDistributionStat traceOptimalScheduleLength("Trace optimal schedule length");

IntDistributionStat regionBuildTime("Region build time");
IntDistributionStat heuristicTime("Heuristic time");
IntDistributionStat AcoTime("ACO time");
IntDistributionStat boundComputationTime("Bound computation time");
IntDistributionStat enumerationTime("Enumeration time");
IntDistributionStat
    enumerationToHeuristicTimeRatio("Enumeration to heuristic time ratio");
IntDistributionStat verificationTime("Verification time");

IntDistributionStat historyEntriesPerIteration("History entries per iteration");
IntDistributionStat historyListSize("History list size");
IntDistributionStat maximumHistoryListSize("Maximum history list size");
IntDistributionStat traversedHistoryListSize("Traversed history list size");
IntDistributionStat historyDominationPosition("History domination position");
IntDistributionStat historyDominationPositionToListSize(
    "History domination position to list size");
IntDistributionStat
    historyTableInitializationTime("History table initialization time");

IntDistributionStat scheduledLatency("Scheduled latency");

TimeoutStat timeouts("Timeouts");

IntStat perfectMatchCount("Perfect match count");
IntStat positiveMismatchCount("Positive mismatch count");
IntStat negativeMismatchCount("Negative mismatch count");
IntStat positiveOptimalMismatchCount("Positive optimal mismatch count");
IntStat negativeOptimalMismatchCount("Negative optimal mismatch count");
IntStat positiveUpperBoundMismatchCount("Positive upper bound mismatch count");
IntStat negativeUpperBoundMismatchCount("Negative upper bound mismatch count");
IntStat positiveLowerBoundMismatchCount("Positive lower bound mismatch count");
IntStat negativeLowerBoundMismatchCount("Negative lower bound mismatch count");

IntStat signatureDominationTests("Signature domination tests");
IntStat signatureMatches("Signature matches");
IntStat signatureAliases("Signature aliases");
IntStat subsetMatches("Subset matches");
IntStat absoluteDominationHits("Absolute domination hits");
IntStat positiveDominationHits("Positive domination hits");
IntStat negativeDominationHits("Negative domination hits");
IntStat dominationPruningHits("Domination pruning hits");
IntStat invalidDominationHits("Invalid domination hits");

IntStat stalls("Stalls");
IntStat feasibilityTests("Feasibility tests");
IntStat feasibilityHits("Feasibility hits");
IntStat nodeSuperiorityInfeasibilityHits("Node superiority infeasibility hits");
IntStat rangeTighteningInfeasibilityHits("Range tightening infeasibility hits");
IntStat
    historyDominationInfeasibilityHits("History domination infeasibility hits");
IntStat
    relaxedSchedulingInfeasibilityHits("Relaxed scheduling infeasibility hits");
IntStat slotCountInfeasibilityHits("Slot count infeasibility hits");
IntStat forwardLBInfeasibilityHits("Forward LB infeasibility hits");
IntStat backwardLBInfeasibilityHits("Backward LB infeasibility hits");

IntStat invalidSchedules("Invalid schedules");

IntStat totalInstructions("Total instructions");
IntStat instructionsWithTighterFileLB("Instructions with tighter file LB");
IntStat cyclesTightenedForTighterFileLB(
    "Cycles tightened for instructions with tighter file LB");
IntStat
    instructionsWithTighterRelaxedLB("Instructions with tighter relaxed LB");
IntStat cyclesTightenedForTighterRelaxedLB(
    "Cycles tightened for instructions with tighter relaxed LB");
IntStat instructionsWithEqualLB("Instructions with equal LB");
IntStat instructionsWithTighterFileUB("Instructions with tighter file UB");
IntStat cyclesTightenedForTighterFileUB(
    "Cycles tightened for instructions with tighter file UB");
IntStat
    instructionsWithTighterRelaxedUB("Instructions with tighter relaxed UB");
IntStat cyclesTightenedForTighterRelaxedUB(
    "Cycles tightened for instructions with tighter relaxed UB");
IntStat instructionsWithEqualUB("Instructions with equal UB");

IntStat maxReadyListSize("Max ready list size");

IntStat negativeNodeDominationHits("Negative node domination hits");

IndexedIntStat instructionTypeCounts("Instruction type counts");
IndexedIntSetStat instructionTypeLatencies("Instruction type latencies");
IndexedIntSetStat dependenceTypeLatencies("Dependence type latencies");

IntStat
    legalListSchedulerInstructionHits("Legal list scheduler instruction hits");
IntStat illegalListSchedulerInstructionHits(
    "Illegal list scheduler instruction hits");

} // namespace stats
} // namespace opt_sched
} // namespace llvm
