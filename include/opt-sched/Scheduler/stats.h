/*******************************************************************************
Description:  Provides a flexible set of classes to keep track of statistical
              records. All records are intended to be write-only to ensure that
              no hidden dependences are introduced due to stat records being
              global. The actual records are also defined here.
Author:       Max Shawabkeh
Created:      Mar. 2011
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_STATS_H
#define OPTSCHED_GENERIC_STATS_H

#include "opt-sched/Scheduler/defines.h"
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>

using std::ostream;
using std::string;

namespace llvm {
namespace opt_sched {
namespace stats {

// An abstract base class for statistical records.
class Stat {
public:
  // Constructs a stat with a given name.
  Stat(const string name) : name_(name) {}
  virtual ~Stat() {}
  // Shortcut for printing the stat to a stream.
  friend std::ostream &operator<<(std::ostream &out, const Stat &stat);
  // Prints the stat to a stream.
  virtual void Print(std::ostream &out) const = 0;

protected:
  // The human-friendly name of the stat.
  const string name_;
};

// A simple single-value numerical record. Holds only one value at a time.
template <class T> class NumericStat : public Stat {
public:
  // Constructs a simple stat record.
  NumericStat(const string name) : Stat(name), value_(0) {}
  // Sets the stat value.
  void Set(T value) { value_ = value; }
  // Sets the stat value to the maximum of the current and the argument.
  void SetMax(T value) {
    if (value > value_)
      value_ = value;
  }
  // Sets the stat value to the minimum of the current and the argument.
  void SetMin(T value) {
    if (value < value_)
      value_ = value;
  }
  // Increments the value in the record.
  NumericStat &operator++() {
    value_++;
    return *this;
  }
  NumericStat &operator++(int) {
    value_++;
    return *this;
  }
  // Decrements the value in the record.
  NumericStat &operator--() {
    value_--;
    return *this;
  }
  NumericStat &operator--(int) {
    value_--;
    return *this;
  }
  // Adds the specified amount to the value in the record.
  NumericStat &operator+=(T change) {
    value_ += change;
    return *this;
  }
  // Subtracts the specified amount from the value in the record.
  NumericStat &operator-=(T change) {
    value_ -= change;
    return *this;
  }

protected:
  // The value tracked by this record.
  T value_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const {
    out << name_ << ": " << value_ << "\n";
  }
};

typedef NumericStat<int64_t> IntStat;
typedef NumericStat<float> FloatStat;

// A statistical numeric distribution record. Calculates count, mean and
// extrema of all the recorded values.
template <class T> class DistributionStat : public Stat {
public:
  // Constructs a numerical stat.
  DistributionStat(const string name);
  // Records a new sample value.
  void Record(T value);

protected:
  // The number of recorded values.
  int count_;
  // The sum of the recorded values.
  T sum_;
  // The minimum of the recorded values.
  T min_;
  // The maximum  of the recorded values.
  T max_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const;
};

typedef DistributionStat<int64_t> IntDistributionStat;
typedef DistributionStat<float> FloatDistributionStat;

// A string-typed record. Holds only the latest value.
class StringStat : public Stat {
public:
  // Constructs a string stat record.
  StringStat(const string name) : Stat(name) {}
  // Sets the stat value.
  void Set(string &value) { value_ = value; }

protected:
  // The string tracked by this record.
  string value_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const {
    out << name_ << ": " << value_ << "\n";
  }
};

// A record to keep track of a set of values.
template <class T> class SetStat : public Stat {
public:
  // Constructs a set stat record.
  SetStat(const string name) : Stat(name) {}
  // Clears the values set.
  void Clear() { values_.clear(); }
  // Add a new value to the set.
  void Add(const T &value) { values_.insert(value); }

protected:
  // The values tracked by this record.
  std::set<T> values_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const;
};
typedef SetStat<int64_t> IntSetStat;
typedef SetStat<float> FloatSetStat;

// A record to keep track of a group of value sets, indexed by strings.
template <class T> class IndexedSetStat : public Stat {
public:
  // Constructs an indexed stat record.
  IndexedSetStat(const string name) : Stat(name) {}
  // Clears all the sets.
  void Clear() { values_.clear(); }
  // Clears a specified set.
  void Clear(const string &index) { values_[index].clear(); }
  // Add a new value to the set.
  void Add(const string &index, const T &value) {
    values_[index].insert(value);
  }

protected:
  // The sets tracked by this record.
  std::map<string, std::set<T>> values_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const;
};

typedef IndexedSetStat<int64_t> IndexedIntSetStat;
typedef IndexedSetStat<float> IndexedFloatSetStat;

// A statistical record of timeouts. Keeps track of all recorded entries.
class TimeoutStat : public Stat {
public:
  // A timed-out block description.
  struct Entry {
    // The number of the timed out region.
    int regionNumber;
    // The number of instructions in the block.
    InstCount instCount;
    // The final lower bound reached by the scheduler on this block.
    int lowerBound;
    // The final upper bound reached by the scheduler on this block.
    int upperBound;
    // Constructs an entry.
    Entry(int regionNumber, InstCount instCount, int lowerBound, int upperBound)
        : regionNumber(regionNumber), instCount(instCount),
          lowerBound(lowerBound), upperBound(upperBound) {}
  };

  // Constructs a timeout stat.
  TimeoutStat(const string name) : Stat(name) {}
  // Records a new sample value given the block count, instruction count,
  // final lower bound value and final upper bound value.
  void Record(int regionNumber, InstCount instCount, int lowerBound,
              int upperBound);

protected:
  std::list<Entry> entries_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const;
};

// A record to keep track of a group of numerical values, indexed by strings.
// Holds only the latest value for each index. Calculates percentages of total
// for each index when printed.
template <class T> class IndexedNumericStat : public Stat {
public:
  // Constructs an indexed stat record.
  IndexedNumericStat(const string name) : Stat(name) {}
  // Sets a stat value.
  void Set(const string &index, T value) { values_[index] = value; }
  // Sets a stat value to the maximum of the current and the supplied.
  void SetMax(const string &index, T value) {
    if (value > values_[index])
      values_[index] = value;
  }
  // Sets a stat value to the minimum of the current and the supplied.
  void SetMin(const string &index, T value) {
    if (value < values_[index])
      values_[index] = value;
  }
  // Increments a value in the record.
  void Increment(const string &index) { values_[index]++; }
  // Decrements a value in the record.
  void Decrement(const string &index) { values_[index]--; }
  // Adds the specified amount to a value in the record.
  void Add(const string &index, T delta) { values_[index] += delta; }
  // Subtracts the specified amount from a value in the record.
  void Subtract(const string &index, T delta) { values_[index] -= delta; }

protected:
  // The values tracked by this record.
  std::map<string, T> values_;
  // Prints the stat to a stream.
  void Print(std::ostream &out) const;
};

typedef IndexedNumericStat<int64_t> IndexedIntStat;
typedef IndexedNumericStat<float> IndexedFloatStat;

// Declarations of the actual statistical records.
// TODO(max): Document where ambiguous.
extern IntDistributionStat nodeCount;
extern IntDistributionStat nodesPerLength;
extern IntDistributionStat solutionTime;
extern IntDistributionStat solutionTimeForSolvedProblems;

extern IntDistributionStat iterations;
extern IntDistributionStat enumerations;
extern IntDistributionStat lengths;

extern IntDistributionStat feasibleSchedulesPerLength;
extern IntDistributionStat improvementsPerLength;

extern IntDistributionStat costChecksPerLength;
extern IntDistributionStat costPruningsPerLength;
extern IntDistributionStat dynamicLBIterationsPerPath;

extern IntDistributionStat problemSize;
extern IntDistributionStat solvedProblemSize;
extern IntDistributionStat unsolvedProblemSize;

extern IntDistributionStat traceCostLowerBound;

extern IntDistributionStat traceHeuristicCost;
extern IntDistributionStat traceOptimalCost;

extern IntDistributionStat traceHeuristicScheduleLength;
extern IntDistributionStat traceOptimalScheduleLength;

extern IntDistributionStat regionBuildTime;
extern IntDistributionStat heuristicTime;
extern IntDistributionStat AcoTime;
extern IntDistributionStat boundComputationTime;
extern IntDistributionStat enumerationTime;
extern IntDistributionStat enumerationToHeuristicTimeRatio;
extern IntDistributionStat verificationTime;

extern IntDistributionStat historyEntriesPerIteration;
extern IntDistributionStat historyListSize;
extern IntDistributionStat maximumHistoryListSize;
extern IntDistributionStat traversedHistoryListSize;
extern IntDistributionStat historyDominationPosition;
extern IntDistributionStat historyDominationPositionToListSize;
extern IntDistributionStat historyTableInitializationTime;

extern IntDistributionStat scheduledLatency;

// The number of regions whose scheduling timed out.
extern TimeoutStat timeouts;

// The number of perfectly matched blocks (when comparing to input).
extern IntStat perfectMatchCount;
// The number of positively mismatched blocks (when comparing to input).
extern IntStat positiveMismatchCount;
// The number of negatively mismatched blocks (when comparing to input).
extern IntStat negativeMismatchCount;
// The number of blocks that were scheduled optimally but were not optimal
// in the input (when comparing to input).
extern IntStat positiveOptimalMismatchCount;
// The number of blocks that were not scheduled optimally but were optimal
// in the input (when comparing to input).
extern IntStat negativeOptimalMismatchCount;
// The number of blocks that were scheduled to have a lower upper bound
// than the one specified in the input (when comparing to input).
extern IntStat positiveUpperBoundMismatchCount;
// The number of blocks that were scheduled to have a higher upper bound
// than the one specified in the input (when comparing to input).
extern IntStat negativeUpperBoundMismatchCount;
// The number of blocks that were scheduled to have a higher lower bound
// than the one specified in the input (when comparing to input).
extern IntStat positiveLowerBoundMismatchCount;
// The number of blocks that were scheduled to have a lower lower bound
// than the one specified in the input (when comparing to input).
extern IntStat negativeLowerBoundMismatchCount;

// Enumeration stats.
extern IntStat signatureDominationTests;
extern IntStat signatureMatches;
extern IntStat signatureAliases;
extern IntStat subsetMatches;
extern IntStat absoluteDominationHits;
extern IntStat positiveDominationHits;
extern IntStat negativeDominationHits;
extern IntStat dominationPruningHits;
extern IntStat invalidDominationHits;

extern IntStat stalls;
extern IntStat feasibilityTests;
extern IntStat feasibilityHits;
extern IntStat nodeSuperiorityInfeasibilityHits;
extern IntStat rangeTighteningInfeasibilityHits;
extern IntStat historyDominationInfeasibilityHits;
extern IntStat relaxedSchedulingInfeasibilityHits;
extern IntStat slotCountInfeasibilityHits;
extern IntStat forwardLBInfeasibilityHits;
extern IntStat backwardLBInfeasibilityHits;

extern IntStat invalidSchedules;

// File/relaxed bound comparisons.
extern IntStat totalInstructions;
extern IntStat instructionsWithTighterFileLB;
extern IntStat cyclesTightenedForTighterFileLB;
extern IntStat instructionsWithTighterRelaxedLB;
extern IntStat cyclesTightenedForTighterRelaxedLB;
extern IntStat instructionsWithEqualLB;
extern IntStat instructionsWithTighterFileUB;
extern IntStat cyclesTightenedForTighterFileUB;
extern IntStat instructionsWithTighterRelaxedUB;
extern IntStat cyclesTightenedForTighterRelaxedUB;
extern IntStat instructionsWithEqualUB;

extern IntStat maxReadyListSize;

extern IntStat negativeNodeDominationHits;

extern IndexedIntStat instructionTypeCounts;
extern IndexedIntSetStat instructionTypeLatencies;
extern IndexedIntSetStat dependenceTypeLatencies;

extern IntStat legalListSchedulerInstructionHits;
extern IntStat illegalListSchedulerInstructionHits;

} // namespace stats
} // namespace opt_sched
} // namespace llvm

#endif
