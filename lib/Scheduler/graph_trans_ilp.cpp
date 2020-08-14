#include "opt-sched/Scheduler/graph_trans_ilp.h"

#include "opt-sched/Scheduler/array_ref2d.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <vector>

using namespace llvm::opt_sched;

// #define IS_DEBUG_ILP_GRAPH_TRANS

#ifdef IS_DEBUG_ILP_GRAPH_TRANS
#define DEBUG_LOG(...) Logger::Info(__VA_ARGS__)
#else
#define DEBUG_LOG(...) static_cast<void>(0)
#endif

static constexpr auto SmallSize = StaticNodeSupILPTrans::SmallSize;

llvm::SmallVector<int, SmallSize>
StaticNodeSupILPTrans::createDistanceTable(DataDepGraph &DDG) {
  const int NegativeInfinity = std::numeric_limits<int>::lowest();
  const int NumNodes_ = DDG.GetNodeCnt();
  assert(NumNodes_ >= 0); // sanity check
  const size_t NumNodes = size_t(NumNodes_);

  const int MaxLatency = DDG.GetMaxLtncy();
  DEBUG_LOG("Creating DISTANCE() table with MaxLatency: %d", MaxLatency);

  llvm::SmallVector<int, SmallSize> DistanceTable_;
  DistanceTable_.resize(
      NumNodes * NumNodes,
      // DISTANCE(i, j) where no edge (i, j) exists = -infinity
      NegativeInfinity);
  MutableArrayRef2D<int> DistanceTable(DistanceTable_, NumNodes, NumNodes);

  // DISTANCE(i, i) = 0
  for (size_t I = 0; I < NumNodes; ++I) {
    for (size_t J = 0; J < NumNodes; ++J) {
      SchedInstruction *NodeI = DDG.GetInstByIndx(I);
      SchedInstruction *NodeJ = DDG.GetInstByIndx(J);

      DistanceTable[{I, J}] =
          NodeI->IsRcrsvScsr(NodeJ)
              ? std::min(NodeJ->GetRltvCrtclPath(DIR_FRWRD, NodeI), MaxLatency)
              : NegativeInfinity;
      DEBUG_LOG(" DISTANCE(%d, %d) = %d", I, J, (DistanceTable[{I, J}]));
    }
  }

  DEBUG_LOG("Finished creating DISTANCE() table\n");

  return DistanceTable_;
}

static size_t castUnsigned(int x) {
  assert(x >= 0); // sanity check
  return size_t(x);
}

static size_t getNum(GraphNode *Node) { return castUnsigned(Node->GetNum()); }

template <typename... Ts>
[[gnu::always_inline]] static inline void suppressUnused(Ts &&...) {}

static int computeSuperiorArrayValue(DataDepGraph &DDG,
                                     ArrayRef2D<int> DistanceTable, //
                                     const int i_, const int j_) {
  SchedInstruction *NodeI = DDG.GetInstByIndx(i_);
  SchedInstruction *NodeJ = DDG.GetInstByIndx(j_);

  if (NodeI->GetInstType() != NodeJ->GetInstType() ||
      !areNodesIndependent(NodeI, NodeJ)) {
    return -1;
  }

  const size_t i = castUnsigned(i_);
  const size_t j = castUnsigned(j_);

  DEBUG_LOG("   Counting bad IPred");
  const int NumBadPredecessors =
      llvm::count_if(NodeI->GetPredecessors(), [&](GraphEdge &e) {
        DEBUG_LOG("    LATENCY(%d, %d) = %d <> DISTANCE(%d, %d) = %d",
                  e.from->GetNum(), i, e.label, //
                  e.from->GetNum(), j, DistanceTable[{getNum(e.from), j}]);
        return e.label > DistanceTable[{getNum(e.from), j}];
      });

  DEBUG_LOG("   Counting bad ISucc");
  const int NumBadSuccessors =
      llvm::count_if(NodeJ->GetSuccessors(), [&](GraphEdge &e) {
        DEBUG_LOG("    LATENCY(%d, %d) = %d <> DISTANCE(%d, %d) = %d", //
                  j, getNum(e.from), e.label,                          //
                  i, getNum(e.from), DistanceTable[{i, getNum(e.to)}]);
        return e.label > DistanceTable[{i, getNum(e.to)}];
      });

  return NumBadPredecessors + NumBadSuccessors;
}

llvm::SmallVector<int, SmallSize>
StaticNodeSupILPTrans::createSuperiorArray(DataDepGraph &DDG,
                                           ArrayRef2D<int> DistanceTable) {
  DEBUG_LOG("Creating SUPERIOR() array");

  const size_t NumNodes = castUnsigned(DDG.GetNodeCnt());

  llvm::SmallVector<int, SmallSize> Superior_;
  Superior_.resize(NumNodes * NumNodes, -1);
  MutableArrayRef2D<int> Superior(Superior_, NumNodes, NumNodes);

  for (size_t I = 0; I < NumNodes; ++I) {
    for (size_t J = 0; J < NumNodes; ++J) {
      SchedInstruction *NodeI = DDG.GetInstByIndx(I);
      SchedInstruction *NodeJ = DDG.GetInstByIndx(J);

      if (NodeI->GetInstType() == NodeJ->GetInstType() &&
          areNodesIndependent(NodeI, NodeJ)) {
        Superior[{I, J}] = computeSuperiorArrayValue(DDG, DistanceTable, I, J);
        DEBUG_LOG(" SUPERIOR(%d, %d) = %d", I, J, Superior[{I, J}]);
      }
    }
  }
  DEBUG_LOG("Finished creating SUPERIOR() array\n");

  return Superior_;
}

llvm::SmallVector<std::pair<int, int>, SmallSize>
StaticNodeSupILPTrans::createSuperiorNodesList(ArrayRef2D<int> SuperiorArray) {
  DEBUG_LOG("Creating SuperiorList of nodes with superiority available");
  const size_t NumNodes = SuperiorArray.rows();

  llvm::SmallVector<std::pair<int, int>, SmallSize> SuperiorNodes;

  for (size_t I = 0; I < NumNodes; ++I) {
    for (size_t J = 0; J < NumNodes; ++J) {
      if (SuperiorArray[{I, J}] == 0) {
        SuperiorNodes.push_back({int(I), int(J)});
        DEBUG_LOG(" Tracking (%d, %d) as SUPERIOR(...) = 0", I, J);
      }
    }
  }
  DEBUG_LOG("Finished initial values for SuperiorList\n");

  return SuperiorNodes;
}

static MutableArrayRef2D<int> wrapAs2D(llvm::MutableArrayRef<int> Ref,
                                       int NumNodes) {
  return MutableArrayRef2D<int>(Ref, NumNodes, NumNodes);
}

StaticNodeSupILPTrans::DataAlloc::DataAlloc(DataDepGraph &DDG)
    : DistanceTable(createDistanceTable(DDG)),
      SuperiorArray(
          createSuperiorArray(DDG, wrapAs2D(DistanceTable, DDG.GetNodeCnt()))),
      SuperiorNodesList(
          createSuperiorNodesList(wrapAs2D(SuperiorArray, DDG.GetNodeCnt()))),
      Stats(), Data_(llvm::make_unique<Data>(Data{
                   DDG,
                   wrapAs2D(this->DistanceTable, DDG.GetNodeCnt()),
                   wrapAs2D(this->SuperiorArray, DDG.GetNodeCnt()),
                   this->SuperiorNodesList,
                   this->Stats,
               })) {}

void StaticNodeSupILPTrans::updateSuperiorArray(Data &Data, int i_, int j_) {
  const size_t i = castUnsigned(i_);
  const size_t j = castUnsigned(j_);

  const int OldValue = Data.SuperiorArray[{i, j}];
  suppressUnused(OldValue);

  const int NewValue =
      computeSuperiorArrayValue(Data.DDG, Data.DistanceTable, i, j);
  Data.SuperiorArray[{i, j}] = NewValue;
  DEBUG_LOG("  Updating SUPERIOR(%d, %d) = %d (old = %d)", i, j, NewValue,
            OldValue);
  // Not necessarily true due to the way we update the SUPERIOR array.
  // After fully updating the SUPERIOR array, it will be true.
  // However, in the process of doing so, we might increase.
  // assert(NewValue <= OldValue);

  if (NewValue == 0) {
    DEBUG_LOG("   Tracking (%d, %d) as a possible superior edge", i, j);
    Data.SuperiorNodesList.push_back({i, j});
  }
}

void StaticNodeSupILPTrans::addZeroLatencyEdge(DataDepGraph &DDG, int i, int j,
                                               Statistics &Stats) {
  SchedInstruction *NodeI = DDG.GetInstByIndx(i);
  SchedInstruction *NodeJ = DDG.GetInstByIndx(j);
  addSuperiorEdge(DDG, NodeI, NodeJ);
  ++Stats.NumEdgesAdded;
  DEBUG_LOG(" Added (%d, %d) superior edge", i, j);
}

void StaticNodeSupILPTrans::addNecessaryResourceEdges(DataDepGraph &DDG, //
                                                      int i, int j,
                                                      Statistics &stats) {
  DEBUG_LOG(" Resource edges not currently implemented");
}

void StaticNodeSupILPTrans::setDistanceTable(StaticNodeSupILPTrans::Data &Data,
                                             int i_, int j_, int Val) {
  const size_t i = castUnsigned(i_);
  const size_t j = castUnsigned(j_);
  const int OldDistance = Data.DistanceTable[{i, j}];
  Data.DistanceTable[{i, j}] = Val;
  DEBUG_LOG("  Updated DISTANCE(%d, %d) = %d (old = %d)", i, j, Val,
            OldDistance);

  if (Val > OldDistance) {
    SchedInstruction *NodeI = Data.DDG.GetInstByIndx(i);
    SchedInstruction *NodeJ = Data.DDG.GetInstByIndx(j);

    for (GraphEdge &e : NodeI->GetPredecessors()) {
      updateSuperiorArray(Data, e.from->GetNum(), j);
    }
    for (GraphEdge &e : NodeJ->GetSuccessors()) {
      updateSuperiorArray(Data, i, e.to->GetNum());
    }
  }
}

void StaticNodeSupILPTrans::updateDistanceTable(Data &Data, int i_, int j_) {
  DEBUG_LOG(" Updating DISTANCE() table");
  const size_t i = castUnsigned(i_);
  const size_t j = castUnsigned(j_);

  DataDepGraph &DDG = Data.DDG;
  MutableArrayRef2D<int> DistanceTable = Data.DistanceTable;

  // Adding the edge (i, j) increases DISTANCE(i, j) to 0 (from -infinity).
  setDistanceTable(Data, i, j, 0);

  const int MaxLatency = DDG.GetMaxLtncy();

  SchedInstruction *NodeI = DDG.GetInstByIndx(i);
  SchedInstruction *NodeJ = DDG.GetInstByIndx(j);

  LinkedList<GraphNode> *JSuccessors = NodeJ->GetRecursiveSuccessors();
  LinkedList<GraphNode> *IPredecessors = NodeI->GetRecursivePredecessors();

  for (GraphNode &Succ : *JSuccessors) {
    const size_t k = getNum(&Succ);
    const int OldDistance = DistanceTable[{i, k}];
    // The "new" DISTANCE(i, k) = DISTANCE(j, k) because we added a latency 0
    // edge (i, j), but only if this "new distance" is larger.
    const int NewDistance = std::min(MaxLatency, DistanceTable[{j, k}]);

    if (NewDistance > OldDistance) {
      DEBUG_LOG("  Increased DISTANCE(%d, %d) = %d (old = %d)", i, k,
                NewDistance, OldDistance);
      setDistanceTable(Data, i, k, NewDistance);

      for (GraphNode &Pred : *IPredecessors) {
        const size_t p = getNum(&Pred);
        const int NewPossiblePK =
            std::min(MaxLatency, NewDistance + DistanceTable[{p, i}]);
        const int OldPK = DistanceTable[{p, k}];

        if (NewPossiblePK > OldPK) {
          DEBUG_LOG("   Increased (i, k) distance resulted in increased "
                    "DISTANCE(%d, %d) = %d (old = %d)",
                    p, k, NewPossiblePK, OldPK);
          setDistanceTable(Data, p, k, NewPossiblePK);
        }
      }
    }
  }
}

void StaticNodeSupILPTrans::removeRedundantEdges(DataDepGraph &DDG,
                                                 ArrayRef2D<int> DistanceTable,
                                                 int i, int j,
                                                 Statistics &stats) {
  // We can't remove redundant edges at this time, because the LinkedList class
  // doesn't support removal if it uses its custom allocator.

  // DEBUG_LOG(" Removing redundant edges");
  // SchedInstruction *NodeI = DDG.GetInstByIndx(i);
  // SchedInstruction *NodeJ = DDG.GetInstByIndx(j);

  // for (GraphNode &Pred : *NodeI->GetRecursivePredecessors()) {
  //   LinkedList<GraphEdge> &PSuccs = Pred.GetSuccessors();

  //   for (auto it = PSuccs.begin(); it != PSuccs.end();) {
  //     GraphEdge &e = *it;

  //     if (NodeJ->IsRcrsvScsr(e.to) &&
  //         e.label <= DistanceTable[{e.from->GetNum(), e.to->GetNum()}]) {
  //       it = PSuccs.RemoveAt(it);
  //       e.to->RemovePredFrom(&Pred);
  //       DEBUG_LOG("  Deleting GraphEdge* at %p: (%d, %d)", (void *)&e,
  //                 e.from->GetNum(), e.to->GetNum());
  //       delete &e;
  //       ++stats.NumEdgesRemoved;
  //     } else {
  //       ++it;
  //     }
  //   }
  // }
}

StaticNodeSupILPTrans::StaticNodeSupILPTrans(DataDepGraph *dataDepGraph)
    : GraphTrans(dataDepGraph) {}

FUNC_RESULT StaticNodeSupILPTrans::ApplyTrans() {
  DataDepGraph &DDG = *GetDataDepGraph_();
  assert(GetNumNodesInGraph_() == DDG.GetNodeCnt());
  Logger::Event("GraphTransILPNodeSuperiority");

  auto Data_ = createData(DDG);
  Data &Data = Data_.getData();

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

    addZeroLatencyEdge(Data, i, j);
    addNecessaryResourceEdges(Data, i, j);

    updateDistanceTable(Data, i, j);
    updateSuperiorArray(Data, i, j);
    removeRedundantEdges(Data, i, j);

    DEBUG_LOG("Finished iteration for (%d, %d)\n", i, j);
  }

  Logger::Event("GraphTransILPNodeSuperiorityFinished",      //
                "superior_edges", Data.Stats.NumEdgesAdded,  //
                "removed_edges", Data.Stats.NumEdgesRemoved, //
                "resource_edges", Data.Stats.NumResourceEdgesAdded);

  return RES_SUCCESS;
}
