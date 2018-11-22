#include "opt-sched/Scheduler/graph_trans.h"
#include "opt-sched/Scheduler/register.h"
#include "opt-sched/Scheduler/bit_vector.h"
#include "opt-sched/Scheduler/logger.h"
#include <list>

using namespace llvm::opt_sched;

GraphTrans::GraphTrans(DataDepGraph *dataDepGraph) {
  assert(dataDepGraph != NULL);

  SetDataDepGraph(dataDepGraph);
  SetNumNodesInGraph(dataDepGraph->GetInstCnt());
}

std::unique_ptr<GraphTrans>
GraphTrans::CreateGraphTrans(TRANS_TYPE type, DataDepGraph *dataDepGraph) {
  switch (type) {
  // Create equivalence detection graph transformation.
  case TT_NSP:
    return std::unique_ptr<GraphTrans>(new StaticNodeSupTrans(dataDepGraph));
  default:
    return nullptr;
  }
}

bool GraphTrans::AreNodesIndep_(SchedInstruction *inst1,
                                SchedInstruction *inst2) {
  // The nodes are independent if there is no path from srcInst to dstInst.
  if (inst1 != inst2 && !inst1->IsRcrsvPrdcsr(inst2) &&
      !inst1->IsRcrsvScsr(inst2)) {
#ifdef IS_DEBUG_GRAPH_TRANS
    Logger::Info("Nodes %d and %d are independent", inst1->GetNum(),
                 inst2->GetNum());
#endif
    return true;
  } else
    return false;
}

void GraphTrans::UpdatePrdcsrAndScsr_(SchedInstruction *nodeA,
                                      SchedInstruction *nodeB) {
  LinkedList<GraphNode> *nodeBScsrLst = nodeB->GetRcrsvNghbrLst(DIR_FRWRD);
  LinkedList<GraphNode> *nodeAPrdcsrLst = nodeA->GetRcrsvNghbrLst(DIR_BKWRD);

  // Update lists for the nodes themselves.
  nodeA->AddRcrsvScsr(nodeB);
  nodeB->AddRcrsvPrdcsr(nodeA);

  for (GraphNode *X = nodeAPrdcsrLst->GetFrstElmnt(); X != NULL;
       X = nodeAPrdcsrLst->GetNxtElmnt()) {

    for (GraphNode *Y = nodeBScsrLst->GetFrstElmnt(); Y != NULL;
         Y = nodeBScsrLst->GetNxtElmnt()) {
      // Check if Y is reachable f
      if (!X->IsRcrsvScsr(Y)) {
        Y->AddRcrsvPrdcsr(X);
        X->AddRcrsvScsr(Y);
      }
    }
  }
}

bool StaticNodeSupTrans::TryAddingSuperiorEdge_(SchedInstruction *nodeA,
                                                SchedInstruction *nodeB) {
  // Return this flag which designates whether an edge was added.
  bool edgeWasAdded = false;

  if (nodeA->GetNodeID() > nodeB->GetNodeID())
    std::swap(nodeA, nodeB);

  if (NodeIsSuperior_(nodeA, nodeB)) {
    AddSuperiorEdge_(nodeA, nodeB);
    edgeWasAdded = true;
  } else if (NodeIsSuperior_(nodeB, nodeA)) {
    AddSuperiorEdge_(nodeB, nodeA);
    // Swap nodeIDs
    //int tmp = nodeA->GetNodeID();
    //nodeA->SetNodeID(nodeB->GetNodeID());
    //nodeB->SetNodeID(tmp);
    edgeWasAdded = true;
  }

  return edgeWasAdded;
}

void StaticNodeSupTrans::AddSuperiorEdge_(SchedInstruction *nodeA,
                                          SchedInstruction *nodeB) {
#if defined(IS_DEBUG_GRAPH_TRANS_RES) || defined(IS_DEBUG_GRAPH_TRANS)
  Logger::Info("Node %d is superior to node %d", nodeA->GetNum(),
               nodeB->GetNum());
#endif
  GetDataDepGraph_()->CreateEdge(nodeA, nodeB, 0, DEP_OTHER);
  UpdatePrdcsrAndScsr_(nodeA, nodeB);
}

FUNC_RESULT StaticNodeSupTrans::ApplyTrans() {
  InstCount numNodes = GetNumNodesInGraph_();
  DataDepGraph *graph = GetDataDepGraph_();
  // A list of independent nodes.
  std::list<std::pair<SchedInstruction *, SchedInstruction *>> indepNodes;
  bool didAddEdge = false;
#ifdef IS_DEBUG_GRAPH_TRANS
  Logger::Info("Applying node superiority graph transformation.");
#endif

  // For the first pass visit all nodes. Add sets of independent nodes to a
  // list.
  for (int i = 0; i < numNodes; i++) {
    SchedInstruction *nodeA = graph->GetInstByIndx(i);
    for (int j = i + 1; j < numNodes; j++) {
      if (i == j)
        continue;
      SchedInstruction *nodeB = graph->GetInstByIndx(j);

#ifdef IS_DEBUG_GRAPH_TRANS
      Logger::Info("Checking nodes %d:%d", i, j);
#endif
      if (AreNodesIndep_(nodeA, nodeB)) {
        didAddEdge = TryAddingSuperiorEdge_(nodeA, nodeB);
        // If the nodes are independent and no superiority was found add the
        // nodes to a list for
        // future passes.
        if (!didAddEdge)
          indepNodes.push_back(std::make_pair(nodeA, nodeB));
      }
    }
  }
  
  // Multi pass node superiority.
  if (GRAPHTRANSFLAGS.multiPassNodeSup)
    nodeMultiPass_(indepNodes);

  return RES_SUCCESS;
}

bool StaticNodeSupTrans::NodeIsSuperior_(SchedInstruction *nodeA,
                                         SchedInstruction *nodeB) {
  DataDepGraph *graph = GetDataDepGraph_();

  if (nodeA->GetIssueType() != nodeB->GetIssueType()) {
#ifdef IS_DEBUG_GRAPH_TRANS
    Logger::Info("Node %d is not of the same issue type as node %d",
                 nodeA->GetNum(), nodeB->GetNum());
#endif
    return false;
  }

  // The predecessor list of A must be a sub-list of the predecessor list of B.
  BitVector *predsA = nodeA->GetRcrsvNghbrBitVector(DIR_BKWRD);
  BitVector *predsB = nodeB->GetRcrsvNghbrBitVector(DIR_BKWRD);
  if (!predsA->IsSubVector(predsB)) {
#ifdef IS_DEBUG_GRAPH_TRANS
    Logger::Info(
        "Pred list of node %d is not a sub-list of the pred list of node %d",
        nodeA->GetNum(), nodeB->GetNum());
#endif
    return false;
  }

  // The successor list of B must be a sub-list of the successor list of A.
  BitVector *succsA = nodeA->GetRcrsvNghbrBitVector(DIR_FRWRD);
  BitVector *succsB = nodeB->GetRcrsvNghbrBitVector(DIR_FRWRD);
  if (!succsB->IsSubVector(succsA)) {
#ifdef IS_DEBUG_GRAPH_TRANS
    Logger::Info(
        "Succ list of node %d is not a sub-list of the succ list of node %d",
        nodeB->GetNum(), nodeA->GetNum());
#endif
    return false;
  }

  // For every virtual register that belongs to the Use set of B but does not
  // belong to the Use set of A
  // there must be at least one instruction C that is distint from A nad B and
  // belongs to the
  // recurisve sucessor lits of both A and B.
  //
  // For every vitrual register that would have its live range lengthened by
  // scheduling B after A,
  // there must be a register of the same time that would have its live range
  // shortened by scheduling
  // A before B.

  // First find registers that belong to the Use Set of B but not to the Use Set
  // of A.
  // TODO (austin) modify wrapper code so it is easier to identify physical
  // registers.
  Register **usesA;
  Register **usesB;
  //Register **usesC;
  int useCntA = nodeA->GetUses(usesA);
  int useCntB = nodeB->GetUses(usesB);
  // Register used by B but not by A.
  std::list<Register *> usesOnlyB;
  // A list of registers that will have their live range lengthened
  // by scheduling B after A.
  std::list<Register *> lengthenedLiveRegisters;
  // The number of registers that will be lengthened by
  // scheduling B after A. Indexed by register type.
  std::vector<InstCount> lengthenedByB(graph->GetRegTypeCnt());
  std::fill(lengthenedByB.begin(), lengthenedByB.end(), 0);
  // The total number of live ranges that could be lengthened by
  // scheduling B after A.
  //InstCount totalLengthenedByB = 0;

  for (int i = 0; i < useCntB; i++) {
    Register *useB = usesB[i];
    // Flag for determining whether useB is used by node A.
    bool usedByA = false;
    for (int j = 0; j < useCntA; j++) {
      Register *useA = usesA[j];
      if (useA == useB) {
        usedByA = true;
        break;
      }
    }
    if (!usedByA) {
#ifdef IS_DEBUG_GRAPH_TRANS
      Logger::Info("Found reg used by nodeB but not nodeA");
#endif

      // For this register did we find a user C that is a successor of
      // A and B.
      bool foundC = false;
      for (const SchedInstruction *user : useB->GetUseList()) {
        if (user != nodeB &&
            nodeB->IsRcrsvScsr(const_cast<SchedInstruction *>(user))) {
          foundC = true;
          break;
        }
      }
      if (!foundC) {
#ifdef IS_DEBUG_GRAPH_TRANS
        Logger::Info("Found register that has its live range lengthend by "
                     "scheduling B after A");
#endif
        return false;
        //lengthenedByB[useB->GetType()]++;
        //totalLengthenedByB++;
      }
    }
  }
/*
  for (int j = 0; j < useCntA && totalLengthenedByB > 0; j++) {
    Register *useA = usesA[j];

    if (lengthenedByB[useA->GetType()] < 1)
      continue;

    // Try to find an instruction that must be scheduled after A
    // that uses register "useA".
    bool foundLaterUse = false;
    for (const SchedInstruction *user : useA->GetUseList()) {
      // If "nodeA" is not a recursive predecessor of "user" nodeA is not the
      // last
      // user of this register.
      if (user != nodeA &&
          !nodeA->IsRcrsvPrdcsr(const_cast<SchedInstruction *>(user))) {
        foundLaterUse = true;
        break;
      }
    }

    if (!foundLaterUse) {
      lengthenedByB[useA->GetType()]--;
      totalLengthenedByB--;
    }
  }

  if (totalLengthenedByB > 0) {
#ifdef IS_DEBUG_GRAPH_TRANS
    Logger::Info("Live range condition 1 failed");
#endif
    return false;
  }
*/

  // For each register type, the number of registers defined by A is less than
  // or equal to the number of registers defined by B.
  Register **defsA;
  Register **defsB;
  int defCntA = nodeA->GetDefs(defsA);
  int defCntB = nodeB->GetDefs(defsB);
  int regTypes = graph->GetRegTypeCnt();
  vector<InstCount> regTypeDefsA(regTypes);
  vector<InstCount> regTypeDefsB(regTypes);

  for (int i = 0; i < defCntA; i++)
    regTypeDefsA[defsA[i]->GetType()]++;

  for (int i = 0; i < defCntB; i++)
    regTypeDefsB[defsB[i]->GetType()]++;

  for (int i = 0; i < regTypes; i++) {
    // Logger::Info("Def count A for Type %d is %d and B is %d", i,
    // regTypeDefsA[i], regTypeDefsB[i]);
    if (regTypeDefsA[i] > regTypeDefsB[i]) {
#ifdef IS_DEBUG_GRAPH_TRANS
      Logger::Info("Live range condition 2 failed");
#endif
      return false;
    }
  }

  return true;
}

void StaticNodeSupTrans::nodeMultiPass_(std::list<std::pair<SchedInstruction *, SchedInstruction *>> indepNodes) {
#ifdef IS_DEBUG_GRAPH_TRANS
      Logger::Info("Applying multi-pass node superiority");
#endif
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

      if (!AreNodesIndep_(nodeA, nodeB)) {
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

GraphTransFlags GraphTrans::GRAPHTRANSFLAGS;
