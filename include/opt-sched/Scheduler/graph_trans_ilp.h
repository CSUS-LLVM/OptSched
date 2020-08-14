#ifndef OPTSCHED_BASIC_GRAPH_TRANS_ILP_H
#define OPTSCHED_BASIC_GRAPH_TRANS_ILP_H

#include "opt-sched/Scheduler/array_ref2d.h"
#include "opt-sched/Scheduler/graph_trans.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace llvm {
namespace opt_sched {

// Node superiority ILP graph transformation.
class StaticNodeSupILPTrans : public GraphTrans {
public:
  StaticNodeSupILPTrans(DataDepGraph *dataDepGraph);

  const char *Name() const override { return "ilp.nodesup"; }

  FUNC_RESULT ApplyTrans() override;

  struct Statistics {
    int NumEdgesAdded = 0;
    int NumResourceEdgesAdded = 0;
    int NumEdgesRemoved = 0;
  };

  struct Data {
    DataDepGraph &DDG;
    MutableArrayRef2D<int> DistanceTable;
    MutableArrayRef2D<int> SuperiorArray;
    llvm::SmallVectorImpl<std::pair<int, int>> &SuperiorNodesList;
    Statistics &Stats;
  };

  static constexpr int SmallSize = 64;

  static llvm::SmallVector<int, SmallSize>
  createDistanceTable(DataDepGraph &DDG);

  static llvm::SmallVector<int, SmallSize>
  createSuperiorArray(DataDepGraph &DDG, ArrayRef2D<int> DistanceTable);

  static llvm::SmallVector<std::pair<int, int>, SmallSize>
  createSuperiorNodesList(ArrayRef2D<int> SuperiorArray);

  class DataAlloc {
    friend class StaticNodeSupILPTrans;

  public:
    explicit DataAlloc(DataDepGraph &DDG);
    Data &getData() { return *Data_; }

  public:
    llvm::SmallVector<int, SmallSize> DistanceTable;
    llvm::SmallVector<int, SmallSize> SuperiorArray;
    llvm::SmallVector<std::pair<int, int>, SmallSize> SuperiorNodesList;
    Statistics Stats = {};

  private:
    std::unique_ptr<Data> Data_;
  };

  static DataAlloc createData(DataDepGraph &DDG) { return DataAlloc(DDG); }

  static void updateSuperiorArray(Data &Data, int i, int j);

  static void setDistanceTable(Data &Data, int i, int j, int Val);

  static void updateDistanceTable(Data &Data, int i, int j);

  static void addZeroLatencyEdge(DataDepGraph &DDG, int i, int j,
                                 Statistics &Stats);

  static void addZeroLatencyEdge(Data &Data, int i, int j) {
    addZeroLatencyEdge(Data.DDG, i, j, Data.Stats);
  }

  static void addNecessaryResourceEdges(DataDepGraph &DDG, int i, int j,
                                        Statistics &Stats);

  static void addNecessaryResourceEdges(Data &Data, int i, int j) {
    addNecessaryResourceEdges(Data.DDG, i, j, Data.Stats);
  }

  static void removeRedundantEdges(DataDepGraph &DDG,
                                   ArrayRef2D<int> DistanceTable, int i, int j,
                                   Statistics &Stats);

  static void removeRedundantEdges(Data &Data, int i, int j) {
    removeRedundantEdges(Data.DDG, Data.DistanceTable, i, j, Data.Stats);
  }
};

} // namespace opt_sched
} // namespace llvm

#endif
