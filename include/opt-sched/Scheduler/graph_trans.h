/*******************************************************************************
Description:  Implement graph transformations to be applied before scheduling.
Author:       Austin Kerbow
Created:      June. 2017
Last Update:  June. 2017
*******************************************************************************/

#ifndef OPTSCHED_BASIC_GRAPH_TRANS_H
#define OPTSCHED_BASIC_GRAPH_TRANS_H

#include "opt-sched/Scheduler/data_dep.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"
#include "opt-sched/Scheduler/sched_region.h"
#include <list>
#include <memory>

namespace llvm {
namespace opt_sched {

// A and B are independent iff there is no path between A and B (in either
// direction)
bool areNodesIndependent(const SchedInstruction *A, const SchedInstruction *B);

// Adds an edge (A --> B) to the graph, updating recursive neighbors.
// The type of the added edge is OTHER.
GraphEdge *addSuperiorEdge(DataDepGraph &DDG, SchedInstruction *A,
                           SchedInstruction *B, int latency = 0);

// An abstract graph transformation class.
class GraphTrans {

public:
  GraphTrans(DataDepGraph *dataDepGraph);
  virtual ~GraphTrans(){};

  virtual const char *Name() const = 0;

  // Apply the graph transformation to the DataDepGraph.
  virtual FUNC_RESULT ApplyTrans() = 0;

  void SetDataDepGraph(DataDepGraph *dataDepGraph);

  void SetSchedRegion(SchedRegion *schedRegion);

  void SetNumNodesInGraph(InstCount numNodesInGraph);

protected:
  DataDepGraph *GetDataDepGraph_() const;
  SchedRegion *GetSchedRegion_() const;
  InstCount GetNumNodesInGraph_() const;

private:
  // A pointer to the graph.
  DataDepGraph *dataDepGraph_;

  // A pointer to the scheduling region.
  SchedRegion *schedRegion_;

  // The total number of nodes in the graph.
  InstCount numNodesInGraph_;
};

inline DataDepGraph *GraphTrans::GetDataDepGraph_() const {
  return dataDepGraph_;
}
inline void GraphTrans::SetDataDepGraph(DataDepGraph *dataDepGraph) {
  dataDepGraph_ = dataDepGraph;
}

inline SchedRegion *GraphTrans::GetSchedRegion_() const { return schedRegion_; }
inline void GraphTrans::SetSchedRegion(SchedRegion *schedRegion) {
  schedRegion_ = schedRegion;
}

inline InstCount GraphTrans::GetNumNodesInGraph_() const {
  return numNodesInGraph_;
}
inline void GraphTrans::SetNumNodesInGraph(InstCount numNodesInGraph) {
  numNodesInGraph_ = numNodesInGraph;
}

// Node superiority graph transformation.
class StaticNodeSupTrans : public GraphTrans {
public:
  StaticNodeSupTrans(DataDepGraph *dataDepGraph, bool IsMultiPass);

  const char *Name() const override { return "rp.nodesup"; }

  FUNC_RESULT ApplyTrans() override;

  static bool isNodeSuperior(DataDepGraph &DDG, int A, int B);

  struct Statistics {
    int NumEdgesAdded = 0;
    int NumEdgesRemoved = 0;
  };
  static void removeRedundantEdges(DataDepGraph &DDG, int i, int j,
                                   Statistics &Stats);

private:
  // Are multiple passes enabled.
  bool IsMultiPass;

  // Return true if node A is superior to node B.
  bool NodeIsSuperior_(SchedInstruction *nodeA, SchedInstruction *nodeB) {
    return isNodeSuperior(*GetDataDepGraph_(), nodeA->GetNum(),
                          nodeB->GetNum());
  }

  // Check if there is superiority involving nodes A and B. If yes, choose which
  // edge to add.
  // Returns true if a superior edge was added.
  bool TryAddingSuperiorEdge_(SchedInstruction *nodeA, SchedInstruction *nodeB);

  // Keep trying to find superior nodes until none can be found or there are no
  // more independent nodes.
  void nodeMultiPass_(
      std::list<std::pair<SchedInstruction *, SchedInstruction *>>);
};

} // namespace opt_sched
} // namespace llvm

#endif
