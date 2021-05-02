#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/bit_vector.h"
#include "opt-sched/Scheduler/logger.h"
#include "opt-sched/Scheduler/register.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <list>
#include <vector>

// #define IS_DEBUG_GRAPH_TRANS

#ifdef IS_DEBUG_GRAPH_TRANS
#define DEBUG_LOG(...) Logger::Info(__VA_ARGS__)
#else
#define DEBUG_LOG(...) static_cast<void>(0)
#endif

using namespace llvm::opt_sched;

bool llvm::opt_sched::areNodesIndependent(const SchedInstruction *A,
                                          const SchedInstruction *B) {
  return A != B && !A->IsRcrsvPrdcsr(B) && !A->IsRcrsvScsr(B);
}

static void UpdateRecursiveNeighbors(SchedInstruction *A, SchedInstruction *B) {
  // Update lists for the nodes themselves.
  A->AddRcrsvScsr(B);
  B->AddRcrsvPrdcsr(A);

  for (GraphNode &X : *A->GetRecursivePredecessors()) {
    if (!B->IsRcrsvPrdcsr(&X)) {
      B->AddRcrsvPrdcsr(&X);
      X.AddRcrsvScsr(B);
    }
  }

  for (GraphNode &Y : *B->GetRecursiveSuccessors()) {
    if (!A->IsRcrsvScsr(&Y)) {
      A->AddRcrsvScsr(&Y);
      Y.AddRcrsvPrdcsr(A);
    }
  }

  for (GraphNode &X : *A->GetRecursivePredecessors()) {
    for (GraphNode &Y : *B->GetRecursiveSuccessors()) {
      if (!X.IsRcrsvScsr(&Y)) {
        Y.AddRcrsvPrdcsr(&X);
        X.AddRcrsvScsr(&Y);
      }
    }
  }
}

GraphEdge *llvm::opt_sched::addSuperiorEdge(DataDepGraph &DDG,
                                            SchedInstruction *A,
                                            SchedInstruction *B, int latency) {
  GraphEdge *e = DDG.CreateEdge(A, B, latency, DEP_OTHER);
  e->IsArtificial = true;
  UpdateRecursiveNeighbors(A, B);

  return e;
}

GraphTrans::GraphTrans(DataDepGraph *dataDepGraph) {
  assert(dataDepGraph != NULL);

  SetDataDepGraph(dataDepGraph);
  SetNumNodesInGraph(dataDepGraph->GetInstCnt());
}

StaticNodeSupTrans::StaticNodeSupTrans(DataDepGraph *dataDepGraph,
                                       bool IsMultiPass_)
    : GraphTrans(dataDepGraph) {
  IsMultiPass = IsMultiPass_;
}

static GraphEdge *addRPSuperiorEdge(DataDepGraph &DDG, SchedInstruction *A,
                                    SchedInstruction *B) {
  DEBUG_LOG("Node %d is superior to node %d", A->GetNum(), B->GetNum());
  return addSuperiorEdge(DDG, A, B);
}

GraphEdge *StaticNodeSupTrans::TryAddingSuperiorEdge_(SchedInstruction *nodeA,
                                                      SchedInstruction *nodeB) {
  if (nodeA->GetNodeID() > nodeB->GetNodeID())
    std::swap(nodeA, nodeB);

  if (NodeIsSuperior_(nodeA, nodeB)) {
    return addRPSuperiorEdge(*GetDataDepGraph_(), nodeA, nodeB);
  } else if (NodeIsSuperior_(nodeB, nodeA)) {
    return addRPSuperiorEdge(*GetDataDepGraph_(), nodeB, nodeA);
  }

  return nullptr;
}

FUNC_RESULT StaticNodeSupTrans::ApplyTrans() {
  InstCount numNodes = GetNumNodesInGraph_();
  DataDepGraph *graph = GetDataDepGraph_();
  // A list of independent nodes.
  std::list<std::pair<SchedInstruction *, SchedInstruction *>> indepNodes;
  Statistics stats;
  Logger::Event("GraphTransRPNodeSuperiority");

  // For the first pass visit all nodes. Add sets of independent nodes to a
  // list.
  for (int i = 0; i < numNodes; i++) {
    SchedInstruction *nodeA = graph->GetInstByIndx(i);
    for (int j = i + 1; j < numNodes; j++) {
      if (i == j)
        continue;
      SchedInstruction *nodeB = graph->GetInstByIndx(j);

      DEBUG_LOG("Checking nodes %d:%d", i, j);

      if (areNodesIndependent(nodeA, nodeB)) {
        GraphEdge *edge = TryAddingSuperiorEdge_(nodeA, nodeB);
        // If the nodes are independent and no superiority was found add the
        // nodes to a list for
        // future passes.
        if (!edge)
          indepNodes.push_back(std::make_pair(nodeA, nodeB));
        else {
          stats.NumEdgesAdded++;
          removeRedundantEdges(*graph, edge->from->GetNum(), edge->to->GetNum(),
                               stats);
        }
      }
    }
  }

  Logger::Event("GraphTransRPNodeSuperiorityFinished", "superior_edges",
                stats.NumEdgesAdded, "removed_edges", stats.NumEdgesRemoved);

  if (IsMultiPass)
    nodeMultiPass_(indepNodes);

  return RES_SUCCESS;
}

bool StaticNodeSupTrans::isNodeSuperior(DataDepGraph &DDG, int A, int B) {
  SchedInstruction *nodeA = DDG.GetInstByIndx(A);
  SchedInstruction *nodeB = DDG.GetInstByIndx(B);

  if (nodeA->GetIssueType() != nodeB->GetIssueType()) {
    DEBUG_LOG("Node %d is not of the same issue type as node %d",
              nodeA->GetNum(), nodeB->GetNum());
    return false;
  }

  // The predecessor list of A must be a sub-list of the predecessor list of B.
  BitVector *predsA = nodeA->GetRcrsvNghbrBitVector(DIR_BKWRD);
  BitVector *predsB = nodeB->GetRcrsvNghbrBitVector(DIR_BKWRD);
  if (!predsA->IsSubVector(predsB)) {
    DEBUG_LOG(
        "Pred list of node %d is not a sub-list of the pred list of node %d",
        nodeA->GetNum(), nodeB->GetNum());
    return false;
  }

  // The successor list of B must be a sub-list of the successor list of A.
  BitVector *succsA = nodeA->GetRcrsvNghbrBitVector(DIR_FRWRD);
  BitVector *succsB = nodeB->GetRcrsvNghbrBitVector(DIR_FRWRD);
  if (!succsB->IsSubVector(succsA)) {
    DEBUG_LOG(
        "Succ list of node %d is not a sub-list of the succ list of node %d",
        nodeB->GetNum(), nodeA->GetNum());
    return false;
  }

  // For every virtual register that belongs to the Use set of B but does not
  // belong to the Use set of A
  // there must be at least one instruction C that is distinct from A and B and
  // belongs to the
  // recursive successor lits of both A and B.
  //
  // For every virtual register that would have its live range lengthened by
  // scheduling B after A,
  // there must be a register of the same type that would have its live range
  // shortened by scheduling
  // A before B.

  // First find registers that belong to the Use Set of B but not to the Use Set
  // of A.
  // TODO (austin) modify wrapper code so it is easier to identify physical
  // registers.
  const int regTypes = DDG.GetRegTypeCnt();

  const llvm::ArrayRef<const Register *> usesA = nodeA->GetUses();
  const llvm::ArrayRef<const Register *> usesB = nodeB->GetUses();

  const llvm::ArrayRef<const Register *> defsA = nodeA->GetDefs();
  const llvm::ArrayRef<const Register *> defsB = nodeB->GetDefs();

  // (# lengthened registers) - (# shortened registers)
  // from scheduling B after A. Indexed by register type.
  llvm::SmallVector<int, 10> amountLengthenedBySwap(regTypes);

  // With B after A, some Use registers' live ranges might be lengthened. If it
  // could be lengthened, we must assume that it will be lengthened.
  auto bUsesLengthened =
      llvm::make_filter_range(usesB, [&](const Register *useB) {
        // Is this register also used by A?
        // If so, reordering A and B would have no effect on this register's
        // live range.
        const bool usedByA = llvm::find(usesA, useB) != usesA.end();
        // If this register isn't used by A, is it at least used
        // by some successor? If so, reordering A and B would have no effect on
        // this register's live range, as it must live until C.
        const auto usedByC = [&] {
          return llvm::any_of(
              useB->GetUseList(), [&](const SchedInstruction *user) {
                // Given: [... B ... A ...]
                // We need to prove that the register `useB` won't be used by an
                // instruction before A but after B. In the hypothetical
                // schedule we are considering, A currently appears after B.
                // Thus, it is sufficient to show that this register has a user
                // C that is a successor of A.
                //
                // This is more relaxed than showing that C is a successor of B,
                // as RcrsvScsr(B) is a subset of RcrsvScsr(A).
                return user != nodeB &&
                       nodeA->IsRcrsvScsr(const_cast<SchedInstruction *>(user));
              });
        };

        return !usedByA && !usedByC();
      });

  for (const Register *bLastUse : bUsesLengthened) {
    ++amountLengthenedBySwap[bLastUse->GetType()];
  }

  // For A's uses, we need to find registers whose live ranges are definitely
  // shortened. Possibly shortened isn't enough.
  auto aUsesShortened =
      llvm::make_filter_range(usesA, [&](const Register *reg) {
        // A given register definitely has its live range shortened if its last
        // use is A.
        // A must be the last use of R if every user of R is a recursive
        // predecessor of A or A itself.
        // However, we can relax this to recursive predecessor of B, since B
        // appears before A's final destination after the swap.
        // This is a strict relaxation, as RecPred(A) is a subset of RecPred(B).
        const auto &regUses = reg->GetUseList();

        return llvm::all_of(regUses, [&](const SchedInstruction *user) {
          // If this register is used both by A and by B, then its live range
          // will not be shortened.
          if (user == nodeB) {
            return false;
          }

          return user == nodeA ||
                 nodeB->IsRcrsvPrdcsr(const_cast<SchedInstruction *>(user));
        });
      });
  for (const Register *aLastUse : aUsesShortened) {
    --amountLengthenedBySwap[aLastUse->GetType()];
  }

  // Every register defined by A is moved earlier, lengthening their live ranges
  for (const Register *reg : defsA) {
    ++amountLengthenedBySwap[reg->GetType()];
  }

  // Every register defined by B is moved later, shortening their live ranges
  for (const Register *reg : defsB) {
    --amountLengthenedBySwap[reg->GetType()];
  }

  return llvm::all_of(amountLengthenedBySwap,
                      [](int netLengthened) { return netLengthened <= 0; });
}

void StaticNodeSupTrans::nodeMultiPass_(
    std::list<std::pair<SchedInstruction *, SchedInstruction *>> indepNodes) {
  Logger::Event("MultiPassGraphTransRPNodeSuperiority");
  // Try to add superior edges until there are no more independent nodes or no
  // edges can be added.
  bool didAddEdge = true;
  while (didAddEdge && indepNodes.size() > 0) {
    std::list<std::pair<SchedInstruction *, SchedInstruction *>>::iterator
        pair = indepNodes.begin();
    didAddEdge = false;

    while (pair != indepNodes.end()) {
      SchedInstruction *nodeA = (*pair).first;
      SchedInstruction *nodeB = (*pair).second;

      if (!areNodesIndependent(nodeA, nodeB)) {
        pair = indepNodes.erase(pair);
      } else {
        bool result = TryAddingSuperiorEdge_(nodeA, nodeB);
        // If a superior edge was added remove the pair of nodes from the list.
        if (result) {
          pair = indepNodes.erase(pair);
          didAddEdge = true;
        } else
          pair++;
      }
    }
  }
}

////////////////////////////////////
// Removal of redundant edges:
static size_t castUnsigned(int x) {
  assert(x >= 0); // sanity check
  return size_t(x);
}

static bool isRedundant(SchedInstruction *NodeI, SchedInstruction *NodeJ,
                        GraphEdge &e) {
  // If this is the edge we just added, it's not redundant
  if (e.from == NodeI && e.to == NodeJ) {
    return false;
  }

  return NodeJ->IsRcrsvScsr(e.to);
}

static LinkedList<GraphEdge>::iterator
removeEdge(LinkedList<GraphEdge> &Succs, LinkedList<GraphEdge>::iterator it,
           StaticNodeSupTrans::Statistics &stats) {
  GraphEdge &e = *it;
  it = Succs.RemoveAt(it);
  e.to->RemovePredFrom(e.from);
  DEBUG_LOG("  Deleting GraphEdge* at %p: (%zu, %zu)", (void *)&e,
            e.from->GetNum(), e.to->GetNum());
  delete &e;
  ++stats.NumEdgesRemoved;

  return it;
}

void StaticNodeSupTrans::removeRedundantEdges(DataDepGraph &DDG, //
                                              int i, int j, Statistics &stats) {
  DEBUG_LOG(" Removing redundant edges");
  SchedInstruction *NodeI = DDG.GetInstByIndx(i);
  SchedInstruction *NodeJ = DDG.GetInstByIndx(j);

  // Check edges from I itself, since GetRecursivePredecessors() doesn't include
  // I.
  {
    LinkedList<GraphEdge> &ISuccs = NodeI->GetSuccessors();
    for (auto it = ISuccs.begin(); it != ISuccs.end();) {
      if (isRedundant(NodeI, NodeJ, *it)) {
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
      if (isRedundant(NodeI, NodeJ, *it)) {
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
