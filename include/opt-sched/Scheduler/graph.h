/*******************************************************************************
Description:  Defines data structures for graph representation and processing,
              with a focus on directed acyclic graphs (DAGs).
Author:       Ghassan Shobaki
Created:      Jun. 2000
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_GENERIC_GRAPH_H
#define OPTSCHED_GENERIC_GRAPH_H

#include "opt-sched/Scheduler/bit_vector.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"

namespace llvm {
namespace opt_sched {

// The type of edge labels.
typedef int UDT_GLABEL;
// The type used for numbers of nodes.
typedef int UDT_GNODES;
// The type used for numbers of edges.
typedef int UDT_GEDGES;

// Colors used to represent nodes' state in graph traversals.
enum GNODE_COLOR {
  // Not visited.
  COL_WHITE,
  // In progress.
  COL_GRAY,
  // Completed.
  COL_BLACK
};

// Traversal directions.
enum DIRECTION { DIR_FRWRD, DIR_BKWRD };

// Forward-declaring the node class to treat circular dependence.
class GraphNode;
class DirAcycGraph;

struct GraphEdge {
  // The two nodes between which the edge is.
  GraphNode *from, *to;
  // Two labels for the edge.
  UDT_GLABEL label, label2;
  // The first node's order in the second node's predecessor list.
  UDT_GEDGES predOrder;
  // The second node's order in the first node's successor list.
  UDT_GEDGES succOrder;
  // Whether or not the edge is an artificial dependency meaning it isn't
  // required to be correct
  bool IsArtificial;

  // Creates an edge between two nodes with labels label and label2.
  GraphEdge(GraphNode *from, GraphNode *to, UDT_GLABEL label,
            UDT_GLABEL label2 = 0, bool IsArtificial = false)
      : from(from), to(to), label(label), label2(label2),
        IsArtificial(IsArtificial) {}

  // Returns the node on the other side of the edge from the provided node.
  // Assumes that the argument is one of the nodes on the sides of the edge.
  GraphNode *GetOtherNode(GraphNode *node) const {
    assert(node == from || node == to);
    return node == from ? to : from;
  }
};

// TODO(max): Refactor. This has far too much stuff for a simple node.
class GraphNode {
public:
  // Creates a node with the number (label) num and with up to maxNodeCnt
  // successors or predecessors. It is assumed that a single graph never
  // contains multiple nodes with the same number.
  GraphNode(UDT_GNODES num, UDT_GNODES maxNodeCnt);
  // Destroys the node.
  ~GraphNode();
  // Clears the node's predecessor list.
  void DelPrdcsrLst();
  // Clears the node's successor list.
  void DelScsrLst();

  // Adds a new edge to the successor list.
  void ApndScsr(GraphEdge *edge);
  // Adds a new edge to the successor list and does some magic.
  // TODO(max): Elaborate on magic.
  void AddScsr(GraphEdge *edge);
  // Adds a new node as a recursive successor.
  void AddRcrsvScsr(GraphNode *node);
  // Removes the last edge from the successor list and optionally deletes
  // the edge object. scsr must be the destination node of that edge.
  void RmvLastScsr(GraphNode *scsr, bool delEdg);
  // Returns the number of edges in this node's successor list.
  UDT_GEDGES GetScsrCnt() const;

  bool RemoveSuccTo(GraphNode *succ);
  bool RemovePredFrom(GraphNode *pred);

  // Adds a new edge to the predecessor list.
  void ApndPrdcsr(GraphEdge *edge);
  // Adds a new edge to the predecessor list and does some magic.
  // TODO(max): Elaborate on magic.
  void AddPrdcsr(GraphEdge *edge);
  // Adds a new node as a recursive predecessor.
  void AddRcrsvPrdcsr(GraphNode *node);
  // Removes the last edge from the predecessor list and optionally deletes
  // the edge object. scsr must be the destination node of that edge.
  void RmvLastPrdcsr(GraphNode *prdcsr, bool delEdg);
  // Returns the number of edges in this node's predecessor list.
  UDT_GEDGES GetPrdcsrCnt() const;

  // Sets the maximum outgoing edge label value to the maximum between the
  // current value and the provided argument.
  // TODO(max): Hide calls to this back into GraphEdge.
  void UpdtMaxEdgLbl(UDT_GLABEL label);

  // Finds the successor edge from this node to the target node. Returns
  // null if not found.
  GraphEdge *FindScsr(GraphNode *trgtNode);
  // Finds the predecessor edge from this node to the target node. Returns
  // null if not found.
  GraphEdge *FindPrdcsr(GraphNode *trgtNode);
  // Fills the node's recursive predecessors or recursive successors list by
  // doing a depth first traversal either up the predecessor tree or down the
  // successor tree.
  void FindRcrsvNghbrs(DIRECTION dir, DirAcycGraph *graph);
  // Adds the specified node to this node' recursive predecessor or successor
  // list, depending on which direction is specified.
  void AddRcrsvNghbr(GraphNode *nghbr, DIRECTION dir);
  // Returns a pointer to the first successor of the node and writes the label
  // of the edge between them to the label argument. Sets the successor
  // iterator.
  GraphNode *GetFrstScsr(UDT_GLABEL &label);
  // Returns a pointer to the next successor of the node and writes the label
  // of the edge between them to the label argument. Must be called after
  // GetFrstScsr() which starts the successor iterator.
  GraphNode *GetNxtScsr(UDT_GLABEL &label);
  // Returns a pointer to the first successor of the node. Sets the successor
  // iterator.
  GraphNode *GetFrstScsr();
  // Returns a pointer to the next successor of the node. Must be called after
  // GetFrstScsr() which starts the successor iterator.
  GraphNode *GetNxtScsr();
  // Checks if a given node is successor-equivalent to this node. Two nodes
  // are successor-equivalent if they have identical successor lists.
  bool IsScsrEquvlnt(GraphNode *othrNode);
  // Returns a pointer to the first Predecesor of the node. Sets the predecesor
  // iterator.
  GraphNode *GetFrstPrdcsr(UDT_GLABEL &label);
  // Returns a pointer to the next predecessor of the node. Must be called after
  // GetFrstPrdcsr() which starts the predecessor iterator.
  GraphNode *GetNxtPrdcsr(UDT_GLABEL &label);
  // Checks if a given node is predecessor-equivalent to this node. Two nodes
  // are predecessor-equivalent if they have identical predecessor lists.
  bool IsPrdcsrEquvlnt(GraphNode *othrNode);
  // Checks if the successor list of this node is dominated by the successor
  // list of the given node. This is the case when the successor list of this
  // node is a subset of that of the given node and each edge label from this
  // node to one of its successors is less than or equal to the corresponding
  // label from the given node to the same successor
  bool IsScsrDmntd(GraphNode *cnddtDmnnt);

  // Returns the sum of the labels of edges from this node to its successors.
  // This value is not dynamically calculated and is adjusted only through
  // AddScsr().
  UDT_GLABEL GetScsrLblSum() const;
  // Returns the sum of the labels of edges from this node to its
  // predecessors. This value is not dynamically calculated and is adjusted
  // only through AddPrdcsr().
  UDT_GLABEL GetPrdcsrLblSum() const;
  // Returns the topological order of this node which was calculated by
  // DepthFirstVisit().
  UDT_GNODES GetTplgclOrdr() const;
  // Returns the maximum among the label of the edges connecting this node to
  // its successors. Calculated by AddScsr() and UpdtMaxEdgLbl().
  UDT_GLABEL GetMaxEdgeLabel() const;
  // Sets the color of this node.
  void SetColor(GNODE_COLOR color);
  // Returns the color of this node.
  GNODE_COLOR GetColor() const;
  // Returns the number (label) of the node.
  UDT_GNODES GetNum() const;

  // Returns whether this node is a root (i.e. has no predecessor).
  bool IsRoot() const;
  // Returns whether this node is a leaf (i.e. has no successor).
  bool IsLeaf() const;
  // Returns whether the given node is a recursive predecessor of this node.
  bool IsRcrsvPrdcsr(GraphNode *node) const;
  // Returns whether the given node is a recursive successor of this node.
  bool IsRcrsvScsr(GraphNode *node) const;
  // Returns whether the given node is a recursive neighbor (predecessor or
  // successor) of this node, depending on the specified direction.
  bool IsRcrsvNghbr(DIRECTION dir, GraphNode *node) const;

  // Allocates memory for the node's predecessor or successor list and bitset,
  // depending on the specified direction.
  void AllocRcrsvInfo(DIRECTION dir, UDT_GNODES nodeCnt);
  // Returns the node's recursive predecessor or successor list, depending on
  // the specified direction.
  LinkedList<GraphNode> *GetRcrsvNghbrLst(DIRECTION dir);
  // Returns the node's recursive predecessor or successor bitset, depending
  // on the specified direction. Nodes which are in the list have the bits
  // indexed by their number set.
  BitVector *GetRcrsvNghbrBitVector(DIRECTION dir);

  // Performs a depth-first visit starting from this node, which includes
  // visiting all of its successors recursively and deducing a topological
  // sort of the nodes.
  void DepthFirstVisit(GraphNode *tplgclOrdr[], UDT_GNODES &tplgclIndx);

  // Writes a comma-separated list of (direct) successor node numbers to the
  // specified file stream.
  void PrntScsrLst(FILE *outFile);
  // Writes a nicely formatted list of (direct) successor node numbers to the
  // info log.
  void LogScsrLst();

  // Returns the number of predecessors in this instruction's transitive
  // closure (i.e. total number of ancestors).
  UDT_GEDGES GetRcrsvPrdcsrCnt() const;
  // Returns the number of successors in this instruction's transitive
  // closure (i.e. total number of descendants).
  UDT_GEDGES GetRcrsvScsrCnt() const;

private:
  // The node number. Should be unique within a single graph.
  UDT_GNODES num_;
  // A list of the immediate successors of this node.
  PriorityList<GraphEdge> *scsrLst_;
  // A list of the immediate predecessors of this node.
  LinkedList<GraphEdge> *prdcsrLst_;
  // A list of all recursively successors of this node.
  LinkedList<GraphNode> *rcrsvScsrLst_;
  // A list of all recursively predecessors of this node.
  LinkedList<GraphNode> *rcrsvPrdcsrLst_;
  // A bitset indicating whether each of the other nodes in the graph is a
  // recursive successor of this node.
  BitVector *isRcrsvScsr_;
  // A bitset indicating whether each of the other nodes in the graph is a
  // recursive predecessor of this node.
  BitVector *isRcrsvPrdcsr_;
  // The index of this node in a topologically-sorted list of nodes in
  // the graph.
  UDT_GNODES tplgclOrdr_;
  // The sum of labels of the edges from this node to its successors.
  UDT_GLABEL scsrLblSum_;
  // The sum of labels of the edges to this node from its predecessors.
  UDT_GLABEL prdcsrLblSum_;
  // The maximum among the label of the edges connecting this node to
  // its successors.
  UDT_GLABEL maxEdgLbl_;
  // The color of this node, to be used during traversal.
  GNODE_COLOR color_;

protected:
  // TODO(max): Document what this is.
  bool FindScsr_(GraphNode *&crntScsr, UDT_GNODES trgtNum, UDT_GLABEL trgtLbl);
  // Actually implements the functionality of FindRcrsvNghbrs().
  void FindRcrsvNghbrs_(GraphNode *root, DIRECTION dir, DirAcycGraph *graph);

  // Returns the node's predecessor or successor list, depending on
  // the specified direction.
  LinkedList<GraphEdge> *GetNghbrLst(DIRECTION dir);

  // Returns a pointer to the edge for the first successor of the node. Sets the
  // successor iterator.
  GraphEdge *GetFrstScsrEdge();
  // Returns a pointer to the edge for the next successor of the node. Must be
  // called after GetFrstScsr() or GetFrstScsrEdge(), which starts the successor
  // iterator.
  GraphEdge *GetNxtScsrEdge();
  GraphEdge *GetLastScsrEdge();
  GraphEdge *GetPrevScsrEdge();
  // Returns a pointer to the edge for the first predecessor of the node. Sets
  // the predecessor iterator.
  GraphEdge *GetFrstPrdcsrEdge();
  // Returns a pointer to the edge for the next predecessor of the node. Must be
  // called after GetFrstPrdcsr() or GetFrstPrdcsrEdge(), which starts the
  // predecessor iterator.
  GraphEdge *GetNxtPrdcsrEdge();
  GraphEdge *GetLastPrdcsrEdge();
  GraphEdge *GetPrevPrdcsrEdge();
};

// TODO(max): Make this class actually useful by providing a way to add nodes
// and edges.
class DirAcycGraph {
public:
  // Creates an empty directed acyclic graph.
  DirAcycGraph();
  // Destroys the graph.
  virtual ~DirAcycGraph();

  // Returns the total number of nodes in the graph.
  inline UDT_GNODES GetNodeCnt() const { return nodeCnt_; }
  // Returns the total number of edges in the graph.
  inline UDT_GEDGES GetEdgeCnt() const { return edgeCnt_; }
  // Returns the maximum number of successors for each nodes in the graph.
  inline UDT_GEDGES GetMaxScsrCnt() const { return maxScsrCnt_; }
  // Returns a pointer to the root node of the graph.
  inline GraphNode *GetRoot() const { return root_; }
  // Returns a pointer to the leaf node of the graph.
  inline GraphNode *GetLeaf() const { return leaf_; }

  // Calculates the topological order of the graph's nodes by performing a
  // depth-first traversal.
  FUNC_RESULT DepthFirstSearch();
  // Fills the recursive predecessor or successor lists for each node in the
  // graph, depending on the specified direction.
  FUNC_RESULT FindRcrsvNghbrs(DIRECTION dir);

  inline void CycleDetected() { cycleDetected_ = true; }

  // Prints a nicely formatted description of the graph to the specified file.
  void Print(FILE *outFile);

  // Log formatted description of the graph.
  void LogGraph();

  // A utility function to reverse direction from forward to backward and vice
  // versa.
  static DIRECTION ReverseDirection(DIRECTION dir);

protected:
  // Pointers to the root and leaf nodes of the graph.
  GraphNode *root_, *leaf_;
  // The total number of nodes in the graph.
  UDT_GNODES nodeCnt_;
  // The total number of edges in the graph.
  UDT_GEDGES edgeCnt_;
  // An array of pointers to the graph's nodes.
  GraphNode **nodes_;
  // The maximum number of successors per node.
  UDT_GEDGES maxScsrCnt_;

  // An array holding the topological order of the graph's nodes.
  GraphNode **tplgclOrdr_;
  // Whether a depth first traversal has been performed and the topological
  // order has been calculated.
  bool dpthFrstSrchDone_;

  // Has a cycle been detected in this graph?
  bool cycleDetected_;

  // Creates a new edge between two nodes with the given numbers with the
  // given label.
  void CreateEdge_(UDT_GNODES frmNodeNum, UDT_GNODES toNodeNum, UDT_GLABEL lbl);
};

inline bool GraphNode::IsRoot() const { return prdcsrLst_->GetElmntCnt() == 0; }

inline bool GraphNode::IsLeaf() const { return scsrLst_->GetElmntCnt() == 0; }

inline void GraphNode::ApndScsr(GraphEdge *edge) {
  assert(edge->from == this);
  scsrLst_->InsrtElmnt(edge, edge->to->GetNum(), true);
}

inline void GraphNode::AddScsr(GraphEdge *edge) {
  assert(edge->from == this);
  UDT_GEDGES scsrNum = scsrLst_->GetElmntCnt();
  scsrLst_->InsrtElmnt(edge, edge->to->GetNum(), true);
  edge->succOrder = scsrNum;
  scsrLblSum_ += edge->label;

  if (edge->label > maxEdgLbl_) {
    maxEdgLbl_ = edge->label;
  }
}

inline void GraphNode::AddRcrsvPrdcsr(GraphNode *node) {
  rcrsvPrdcsrLst_->InsrtElmnt(node);
  isRcrsvPrdcsr_->SetBit(node->GetNum());
}

inline void GraphNode::AddRcrsvScsr(GraphNode *node) {
  rcrsvScsrLst_->InsrtElmnt(node);
  isRcrsvScsr_->SetBit(node->GetNum());
}

inline void GraphNode::UpdtMaxEdgLbl(UDT_GLABEL label) {
  if (label > maxEdgLbl_)
    maxEdgLbl_ = label;
}

inline void GraphNode::RmvLastScsr(GraphNode *scsr, bool delEdg) {
  assert(scsrLst_->GetElmntCnt() > 0);
  assert(scsrLst_->GetLastElmnt()->to == scsr);
  assert(scsrLst_->GetLastElmnt()->from == this);
  if (delEdg)
    delete scsrLst_->GetLastElmnt();
  scsrLst_->RmvLastElmnt();
}

inline bool GraphNode::RemoveSuccTo(GraphNode *succ) {
  auto pos = llvm::find_if(
      *scsrLst_, [succ](const GraphEdge &edge) { return edge.to == succ; });
  if (pos != scsrLst_->end()) {
    scsrLst_->RemoveAt(pos);
  }

  return pos != scsrLst_->end();
}

inline bool GraphNode::RemovePredFrom(GraphNode *pred) {
  auto pos = llvm::find_if(
      *prdcsrLst_, [pred](const GraphEdge &edge) { return edge.from == pred; });
  if (pos != prdcsrLst_->end()) {
    prdcsrLst_->RemoveAt(pos);
  }

  return pos != prdcsrLst_->end();
}

inline void GraphNode::ApndPrdcsr(GraphEdge *edge) {
  assert(edge->to == this);
  prdcsrLst_->InsrtElmnt(edge);
}

inline void GraphNode::AddPrdcsr(GraphEdge *edge) {
  assert(edge->to == this);
  UDT_GEDGES prdcsrNum = prdcsrLst_->GetElmntCnt();
  prdcsrLst_->InsrtElmnt(edge);
  edge->predOrder = prdcsrNum;
  prdcsrLblSum_ += edge->label;
}

inline void GraphNode::RmvLastPrdcsr(GraphNode *prdcsr, bool delEdg) {
  assert(prdcsrLst_->GetElmntCnt() > 0);
  assert(prdcsrLst_->GetLastElmnt()->from == prdcsr);
  assert(prdcsrLst_->GetLastElmnt()->to == this);
  if (delEdg)
    delete prdcsrLst_->GetLastElmnt();
  prdcsrLst_->RmvLastElmnt();
}

inline void GraphNode::SetColor(GNODE_COLOR color) { color_ = color; }

inline UDT_GEDGES GraphNode::GetPrdcsrCnt() const {
  return prdcsrLst_->GetElmntCnt();
}

inline UDT_GEDGES GraphNode::GetScsrCnt() const {
  return scsrLst_->GetElmntCnt();
}

inline GNODE_COLOR GraphNode::GetColor() const { return color_; }

inline UDT_GNODES GraphNode::GetNum() const { return num_; }

inline GraphNode *GraphNode::GetFrstScsr(UDT_GLABEL &label) {
  GraphEdge *edge = scsrLst_->GetFrstElmnt();
  if (edge == NULL)
    return NULL;
  label = edge->label;
  return edge->to;
}

inline GraphNode *GraphNode::GetNxtScsr(UDT_GLABEL &label) {
  GraphEdge *edge = scsrLst_->GetNxtElmnt();
  if (edge == NULL)
    return NULL;
  label = edge->label;
  return edge->to;
}

inline GraphNode *GraphNode::GetFrstPrdcsr(UDT_GLABEL &label) {
  GraphEdge *edge = prdcsrLst_->GetFrstElmnt();
  if (edge == NULL)
    return NULL;
  label = edge->label;
  return edge->to;
}

inline GraphNode *GraphNode::GetNxtPrdcsr(UDT_GLABEL &label) {
  GraphEdge *edge = prdcsrLst_->GetNxtElmnt();
  if (edge == NULL)
    return NULL;
  label = edge->label;
  return edge->to;
}

inline GraphNode *GraphNode::GetFrstScsr() {
  UDT_GLABEL label;
  return GetFrstScsr(label);
}

inline GraphNode *GraphNode::GetNxtScsr() {
  UDT_GLABEL label;
  return GetNxtScsr(label);
}

inline UDT_GLABEL GraphNode::GetScsrLblSum() const { return scsrLblSum_; }

inline UDT_GLABEL GraphNode::GetPrdcsrLblSum() const { return prdcsrLblSum_; }

inline UDT_GNODES GraphNode::GetTplgclOrdr() const { return tplgclOrdr_; }

inline UDT_GLABEL GraphNode::GetMaxEdgeLabel() const { return maxEdgLbl_; }

inline LinkedList<GraphNode> *GraphNode::GetRcrsvNghbrLst(DIRECTION dir) {
  return dir == DIR_FRWRD ? rcrsvScsrLst_ : rcrsvPrdcsrLst_;
}

inline BitVector *GraphNode::GetRcrsvNghbrBitVector(DIRECTION dir) {
  return dir == DIR_FRWRD ? isRcrsvScsr_ : isRcrsvPrdcsr_;
}

inline bool GraphNode::IsRcrsvPrdcsr(GraphNode *node) const {
  assert(node != NULL);
  if (node == this)
    return true;
  return isRcrsvPrdcsr_->GetBit(node->GetNum());
}

inline bool GraphNode::IsRcrsvScsr(GraphNode *node) const {
  assert(node != NULL);
  if (node == this)
    return true;
  return isRcrsvScsr_->GetBit(node->GetNum());
}

inline bool GraphNode::IsRcrsvNghbr(DIRECTION dir, GraphNode *node) const {
  if (dir == DIR_FRWRD) {
    return IsRcrsvScsr(node);
  } else {
    return IsRcrsvPrdcsr(node);
  }
}

inline UDT_GEDGES GraphNode::GetRcrsvPrdcsrCnt() const {
  return rcrsvPrdcsrLst_->GetElmntCnt();
}

inline UDT_GEDGES GraphNode::GetRcrsvScsrCnt() const {
  return rcrsvScsrLst_->GetElmntCnt();
}

inline LinkedList<GraphEdge> *GraphNode::GetNghbrLst(DIRECTION dir) {
  return dir == DIR_FRWRD ? prdcsrLst_ : scsrLst_;
}

inline GraphEdge *GraphNode::GetFrstScsrEdge() {
  return scsrLst_->GetFrstElmnt();
}

inline GraphEdge *GraphNode::GetNxtScsrEdge() {
  return scsrLst_->GetNxtElmnt();
}

inline GraphEdge *GraphNode::GetLastScsrEdge() {
  return scsrLst_->GetLastElmnt();
}

inline GraphEdge *GraphNode::GetPrevScsrEdge() {
  return scsrLst_->GetPrevElmnt();
}

inline GraphEdge *GraphNode::GetFrstPrdcsrEdge() {
  return prdcsrLst_->GetFrstElmnt();
}

inline GraphEdge *GraphNode::GetNxtPrdcsrEdge() {
  return prdcsrLst_->GetNxtElmnt();
}

inline GraphEdge *GraphNode::GetLastPrdcsrEdge() {
  return prdcsrLst_->GetLastElmnt();
}

inline GraphEdge *GraphNode::GetPrevPrdcsrEdge() {
  return prdcsrLst_->GetPrevElmnt();
}

inline DIRECTION DirAcycGraph::ReverseDirection(DIRECTION dir) {
  return dir == DIR_FRWRD ? DIR_BKWRD : DIR_FRWRD;
}

} // namespace opt_sched
} // namespace llvm

#endif
