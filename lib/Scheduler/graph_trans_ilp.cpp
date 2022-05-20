#include "opt-sched/Scheduler/graph_trans_ilp.h"

#include "opt-sched/Scheduler/array_ref2d.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <cstddef> // std::size_t
#include <limits>
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
                  j, getNum(e.to), e.label,                            //
                  i, getNum(e.to), DistanceTable[{i, getNum(e.to)}]);
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
      AddedEdges(), Stats(),
      Data_(std::make_unique<Data>(Data{
          DDG,
          wrapAs2D(this->DistanceTable, DDG.GetNodeCnt()),
          wrapAs2D(this->SuperiorArray, DDG.GetNodeCnt()),
          this->SuperiorNodesList,
          this->AddedEdges,
          this->Stats,
      })) {}

static void decrementSuperiorArray(
    llvm::SmallVectorImpl<std::pair<int, int>> &SuperiorNodesList,
    MutableArrayRef2D<int> SuperiorArray, int i_, int j_) {
  const size_t i = castUnsigned(i_);
  const size_t j = castUnsigned(j_);

  const int OldValue = SuperiorArray[{i, j}];
  const int NewValue = OldValue - 1;

  SuperiorArray[{i, j}] = NewValue;
  DEBUG_LOG("  Updating SUPERIOR(%d, %d) = %d (old = %d)", i, j, NewValue,
            OldValue);
  assert(NewValue >= 0);

  if (NewValue == 0) {
    DEBUG_LOG("   Tracking (%d, %d) as a possible superior edge", i, j);
    SuperiorNodesList.push_back({i_, j_});
  }
}

void StaticNodeSupILPTrans::addZeroLatencyEdge(Data &Data, int i, int j) {
  SchedInstruction *NodeI = Data.DDG.GetInstByIndx(i);
  SchedInstruction *NodeJ = Data.DDG.GetInstByIndx(j);
  GraphEdge *e = addSuperiorEdge(Data.DDG, NodeI, NodeJ);
  Data.AddedEdges.insert(e);
  ++Data.Stats.NumEdgesAdded;
  DEBUG_LOG(" Added (%d, %d) superior edge", i, j);
}

void StaticNodeSupILPTrans::addNecessaryResourceEdges(DataDepGraph &DDG, //
                                                      int i, int j,
                                                      Statistics &stats) {
  DEBUG_LOG(" Resource edges not currently implemented");
}

void StaticNodeSupILPTrans::setDistanceTable(StaticNodeSupILPTrans::Data &Data,
                                             int i_, int j_, int NewDistance) {
  const size_t i = castUnsigned(i_);
  const size_t j = castUnsigned(j_);
  const int OldDistance = Data.DistanceTable[{i, j}];
  Data.DistanceTable[{i, j}] = NewDistance;
  DEBUG_LOG("  Updated DISTANCE(%d, %d) = %d (old = %d)", i, j, NewDistance,
            OldDistance);

  assert(NewDistance > OldDistance);

  SchedInstruction *NodeI = Data.DDG.GetInstByIndx(i);
  SchedInstruction *NodeJ = Data.DDG.GetInstByIndx(j);

  DEBUG_LOG("  . Checking I's successors");
  for (GraphEdge &e : NodeI->GetSuccessors()) {
    const int Latency = e.label;
    const size_t k = castUnsigned(e.to->GetNum());

    if (Data.AddedEdges.find(&e) != Data.AddedEdges.end()) {
      DEBUG_LOG("   Skipping; successor from superior edge: (%d, %d)", i, k);
      continue;
    }
    if (!areNodesIndependent(NodeJ, static_cast<SchedInstruction *>(e.to))) {
      DEBUG_LOG("   Skipping; not independent with J. K: %d J: %d", k, j);
      continue;
    }

    if (Latency <= NewDistance && Latency > OldDistance) {
      decrementSuperiorArray(Data.SuperiorNodesList, Data.SuperiorArray, k, j);
    }
  }

  DEBUG_LOG("  . Checking J's predecessors");
  for (GraphEdge &e : NodeJ->GetPredecessors()) {
    const int Latency = e.label;
    const size_t k = castUnsigned(e.from->GetNum());

    if (Data.AddedEdges.find(&e) != Data.AddedEdges.end()) {
      DEBUG_LOG("   Skipping; predecessor from superior edge: (%d, %d)", k, j);
      continue;
    }
    if (!areNodesIndependent(NodeI, static_cast<SchedInstruction *>(e.from))) {
      DEBUG_LOG("   Skipping; not independent with I. K: %d I: %d", k, i);
      continue;
    }

    if (Latency <= NewDistance && Latency > OldDistance) {
      decrementSuperiorArray(Data.SuperiorNodesList, Data.SuperiorArray, i, k);
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
  if (DistanceTable[{i, j}] < 0) {
    setDistanceTable(Data, i, j, 0);
  }

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

static bool isRedundant(SchedInstruction *NodeI, SchedInstruction *NodeJ,
                        ArrayRef2D<int> DistanceTable, GraphEdge &e) {
  // If this is the edge we just added, it's not redundant
  if (e.from == NodeI && e.to == NodeJ) {
    return false;
  }

  const size_t From = castUnsigned(e.from->GetNum());
  const size_t To = castUnsigned(e.to->GetNum());

  const size_t I = castUnsigned(NodeI->GetNum());
  const size_t J = castUnsigned(NodeJ->GetNum());

  // If this edge is not (I, J) and there is a path through From -> (I, J) -> To
  // which is at least as long as this edge's weight, then this edge is
  // redundant.
  // This is because this path implies that this edge is a transitive edge and
  // the length condition shows that this edge doesn't affect the critical path
  // distances.

  // Note: DistanceTable[{I, J}] should always be 0 at this point, but with
  // resource edges, this may not necessarily be true.
  // Note: we don't need to saturate at MaxLatency because it doesn't affect the
  // answer.
  const int DistThroughIJ =
      DistanceTable[{From, I}] + DistanceTable[{I, J}] + DistanceTable[{J, To}];

  return NodeJ->IsRcrsvScsr(e.to) && e.label <= DistThroughIJ;
}

static LinkedList<GraphEdge>::iterator
removeEdge(LinkedList<GraphEdge> &Succs, LinkedList<GraphEdge>::iterator it,
           StaticNodeSupILPTrans::Statistics &stats) {
  GraphEdge &e = *it;
  it = Succs.RemoveAt(it);
  e.to->RemovePredFrom(e.from);
  DEBUG_LOG("  Deleting GraphEdge* at %p: (%zu, %zu)", (void *)&e,
            e.from->GetNum(), e.to->GetNum());
  delete &e;
  ++stats.NumEdgesRemoved;

  return it;
}

void StaticNodeSupILPTrans::removeRedundantEdges(DataDepGraph &DDG,
                                                 ArrayRef2D<int> DistanceTable,
                                                 int i, int j,
                                                 Statistics &stats) {
  DEBUG_LOG(" Removing redundant edges");
  SchedInstruction *NodeI = DDG.GetInstByIndx(i);
  SchedInstruction *NodeJ = DDG.GetInstByIndx(j);

  // Check edges from I itself, since GetRecursivePredecessors() doesn't include
  // I.
  {
    LinkedList<GraphEdge> &ISuccs = NodeI->GetSuccessors();
    for (auto it = ISuccs.begin(); it != ISuccs.end();) {
      if (isRedundant(NodeI, NodeJ, DistanceTable, *it)) {
        it = removeEdge(ISuccs, it, stats);
      } else {
        ++it;
      }
    }
  }

  // Check edges from a predecessor of I to a successor of J (or J itself).
  // We don't need to explicitly check J itself in a separate step because
  // the isRedundant() check appropriately considers edges ending at J.
  for (GraphNode &Pred : *NodeI->GetRecursivePredecessors()) {
    LinkedList<GraphEdge> &PSuccs = Pred.GetSuccessors();

    for (auto it = PSuccs.begin(); it != PSuccs.end();) {
      if (isRedundant(NodeI, NodeJ, DistanceTable, *it)) {
        it = removeEdge(PSuccs, it, stats);
      } else {
        ++it;
      }
    }
  }

  // Don't need to repeat for successors of J, as those are already considered
  // by the prior loops. We could have checked the successors of J instead of
  // predecessors of I, but we don't need to explicitly check both.
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
    removeRedundantEdges(Data, i, j);

    DEBUG_LOG("Finished iteration for (%d, %d)\n", i, j);
  }

  assert(Data.AddedEdges.size() <=
         static_cast<std::size_t>(std::numeric_limits<int>::max()));
  assert(Data.Stats.NumEdgesAdded == static_cast<int>(Data.AddedEdges.size()));
  Logger::Event("GraphTransILPNodeSuperiorityFinished",      //
                "superior_edges", Data.Stats.NumEdgesAdded,  //
                "removed_edges", Data.Stats.NumEdgesRemoved, //
                "resource_edges", Data.Stats.NumResourceEdgesAdded);

  return RES_SUCCESS;
}
