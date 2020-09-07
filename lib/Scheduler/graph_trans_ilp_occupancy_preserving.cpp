#include "opt-sched/Scheduler/graph_trans_ilp_occupancy_preserving.h"

#include "opt-sched/Scheduler/graph_trans_ilp.h"
#include "opt-sched/Scheduler/logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <vector>

using namespace llvm::opt_sched;

// #define IS_DEBUG_OCCUPANCY_PRESERVING_ILP_GRAPH_TRANS

#ifdef IS_DEBUG_OCCUPANCY_PRESERVING_ILP_GRAPH_TRANS
#define DEBUG_LOG(...) Logger::Info(__VA_ARGS__)
#else
#define DEBUG_LOG(...) static_cast<void>(0)
#endif

using ILP = StaticNodeSupILPTrans;
using RP = StaticNodeSupTrans;

StaticNodeSupOccupancyPreservingILPTrans::
    StaticNodeSupOccupancyPreservingILPTrans(DataDepGraph *DDG)
    : GraphTrans(DDG) {}

FUNC_RESULT StaticNodeSupOccupancyPreservingILPTrans::ApplyTrans() {
  Logger::Info("Performing occupancy-preserving ILP graph transformations");

  DataDepGraph &DDG = *GetDataDepGraph_();
  assert(GetNumNodesInGraph_() == DDG.GetNodeCnt());

  auto Data_ = ILP::createData(DDG);
  ILP::Data &Data = Data_.getData();

  int NumPassedILP = 0;
  int NumFailedRP = 0;

  DEBUG_LOG("Starting main algorithm");
  while (!Data.SuperiorNodesList.empty()) {
    auto ij = Data.SuperiorNodesList.pop_back_val();
    const int i = ij.first;
    const int j = ij.second;
    DEBUG_LOG("Considering adding a superior edge (%d, %d)", i, j);

    if (!areNodesIndependent(DDG.GetInstByIndx(i), DDG.GetInstByIndx(j))) {
      DEBUG_LOG("Skipping (%d, %d) because nodes are no longer independent\n",
                i, j);
      continue;
    }
    ++NumPassedILP;
    if (!RP::isNodeSuperior(DDG, i, j)) {
      DEBUG_LOG("(%d, %d) failed the occupancy-preserving conditions\n", i, j);
      ++NumFailedRP;
      continue;
    }

    ILP::addZeroLatencyEdge(Data, i, j);
    ILP::addNecessaryResourceEdges(Data, i, j);

    ILP::updateDistanceTable(Data, i, j);
    ILP::updateSuperiorArray(Data, i, j);
    // ILP redundant edges are also redundant from RP point of view.
    // This is because ILP redundant edges are transitive edges with more
    // conditions met, and the RP point of view considers transitive edges to be
    // redundant.
    ILP::removeRedundantEdges(Data, i, j);

    DEBUG_LOG("Finished iteration for (%d, %d)\n", i, j);
  }

  Logger::Info(
      "Finished occupancy-preserving ILP graph transformations. Added edges: "
      "%d. Removed redundant edges: %d. Resource edges utilized: %d. Passed "
      "ILP conditions: %d. Failed RP conditions: %d.",
      Data.Stats.NumEdgesAdded, Data.Stats.NumEdgesRemoved,
      Data.Stats.NumResourceEdgesAdded, NumPassedILP, NumFailedRP);

  return RES_SUCCESS;
}
