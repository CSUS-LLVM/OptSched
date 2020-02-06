#include "opt-sched/Scheduler/graph.h"
#include "opt-sched/Scheduler/bit_vector.h"
#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/lnkd_lst.h"
#include "opt-sched/Scheduler/logger.h"
#include <cstdio>

using namespace llvm::opt_sched;

GraphNode::GraphNode(UDT_GNODES num, UDT_GNODES maxNodeCnt) {
  num_ = num;
  scsrLblSum_ = 0;
  prdcsrLblSum_ = 0;
  maxEdgLbl_ = 0;
  color_ = COL_WHITE;

  scsrLst_ = new PriorityList<GraphEdge>(maxNodeCnt);
  prdcsrLst_ = new LinkedList<GraphEdge>(maxNodeCnt);

  rcrsvScsrLst_ = NULL;
  rcrsvPrdcsrLst_ = NULL;
  isRcrsvScsr_ = NULL;
  isRcrsvPrdcsr_ = NULL;
}

GraphNode::~GraphNode() {
  DelScsrLst();
  delete scsrLst_;
  delete prdcsrLst_;
  if (rcrsvScsrLst_ != NULL)
    delete rcrsvScsrLst_;
  if (rcrsvPrdcsrLst_ != NULL)
    delete rcrsvPrdcsrLst_;
  if (isRcrsvScsr_ != NULL)
    delete isRcrsvScsr_;
  if (isRcrsvPrdcsr_ != NULL)
    delete isRcrsvPrdcsr_;
}

void GraphNode::DelPrdcsrLst() {
  for (GraphEdge *crntEdge = prdcsrLst_->GetFrstElmnt(); crntEdge != NULL;
       crntEdge = prdcsrLst_->GetNxtElmnt()) {
    delete crntEdge;
  }

  prdcsrLst_->Reset();
}

void GraphNode::DelScsrLst() {
  for (GraphEdge *crntEdge = scsrLst_->GetFrstElmnt(); crntEdge != NULL;
       crntEdge = scsrLst_->GetNxtElmnt()) {
    delete crntEdge;
  }

  scsrLst_->Reset();
}

void GraphNode::DepthFirstVisit(GraphNode *tplgclOrdr[],
                                UDT_GNODES &tplgclIndx) {
  color_ = COL_GRAY;

  // Iterate through the successor list of this node and recursively visit them
  // This recursion will bottom up when the exit node is reached, which then
  // gets added to the very bottom of the topological sort list.
  for (GraphEdge *crntEdge = scsrLst_->GetFrstElmnt(); crntEdge != NULL;
       crntEdge = scsrLst_->GetNxtElmnt()) {
    GraphNode *scsr = crntEdge->GetOtherNode(this);

    if (scsr->GetColor() == COL_WHITE) {
      scsr->DepthFirstVisit(tplgclOrdr, tplgclIndx);
    }
  }

  // When all the successors of this node have been recursively visited, the
  // node is finished. It is marked black and inserted in the topological list
  // on top of all of its successors.
  color_ = COL_BLACK;
  assert(tplgclIndx >= 0);
  tplgclOrdr[tplgclIndx] = this;
  tplgclOrdr_ = tplgclIndx;
  tplgclIndx--;
}

void GraphNode::FindRcrsvNghbrs(DIRECTION dir, DirAcycGraph *graph) {
  FindRcrsvNghbrs_(this, dir, graph);
}

void GraphNode::AddRcrsvNghbr(GraphNode *nghbr, DIRECTION dir) {
  LinkedList<GraphNode> *rcrsvNghbrLst = GetRcrsvNghbrLst(dir);
  BitVector *isRcrsvNghbr = GetRcrsvNghbrBitVector(dir);

  rcrsvNghbrLst->InsrtElmnt(nghbr);
  isRcrsvNghbr->SetBit(nghbr->GetNum());
}

void GraphNode::AllocRcrsvInfo(DIRECTION dir, UDT_GNODES nodeCnt) {
  if (dir == DIR_FRWRD) {
    if (rcrsvScsrLst_ != NULL) {
      delete rcrsvScsrLst_;
      rcrsvScsrLst_ = NULL;
    }
    if (isRcrsvScsr_ != NULL) {
      delete isRcrsvScsr_;
      isRcrsvScsr_ = NULL;
    }
    assert(rcrsvScsrLst_ == NULL && isRcrsvScsr_ == NULL);
    rcrsvScsrLst_ = new LinkedList<GraphNode>;
    isRcrsvScsr_ = new BitVector(nodeCnt);
  } else {
    if (rcrsvPrdcsrLst_ != NULL) {
      delete rcrsvPrdcsrLst_;
      rcrsvPrdcsrLst_ = NULL;
    }
    if (isRcrsvPrdcsr_ != NULL) {
      delete isRcrsvPrdcsr_;
      isRcrsvPrdcsr_ = NULL;
    }
    assert(rcrsvPrdcsrLst_ == NULL && isRcrsvPrdcsr_ == NULL);
    rcrsvPrdcsrLst_ = new LinkedList<GraphNode>;
    isRcrsvPrdcsr_ = new BitVector(nodeCnt);
  }
}

bool GraphNode::IsScsrDmntd(GraphNode *cnddtDmnnt) {
  if (cnddtDmnnt == this)
    return true;

  // A node dominates itself.

  UDT_GNODES thisScsrCnt = GetScsrCnt();
  UDT_GNODES cnddtScsrCnt = cnddtDmnnt->GetScsrCnt();

  if (thisScsrCnt > cnddtScsrCnt)
    return false;

  assert(thisScsrCnt > 0);

  UDT_GLABEL thisLbl;
  for (GraphNode *thisScsr = GetFrstScsr(thisLbl); thisScsr != NULL;
       thisScsr = GetNxtScsr(thisLbl)) {
    GraphNode *cnddtScsr = NULL;
    if (!cnddtDmnnt->FindScsr_(cnddtScsr, thisScsr->GetNum(), thisLbl)) {
      return false;
    }
  }

  return true;
}

bool GraphNode::FindScsr_(GraphNode *&crntScsr, UDT_GNODES trgtNum,
                          UDT_GLABEL trgtLbl) {
  UDT_GNODES crntNum = INVALID_VALUE;
  UDT_GLABEL crntLbl = 0;

  if (crntScsr == NULL) {
    crntScsr = GetFrstScsr(crntLbl);
  } else {
    crntScsr = GetNxtScsr(crntLbl);
  }

  for (; crntScsr != NULL; crntScsr = GetNxtScsr(crntLbl)) {
    // Verify the fact that the list is sorted in ascending order.
    assert(crntNum == INVALID_VALUE || crntNum >= crntScsr->GetNum());

    crntNum = crntScsr->GetNum();

    if (crntNum == trgtNum) {
      return (trgtLbl <= crntLbl);
    } else if (crntNum < trgtNum) {
      // If we reach the next successor number then we lost the chance
      // to find the target number because we have passed it (the list
      // (is sorted).
      return false;
    }
  }

  assert(crntScsr == NULL);
  return false;
}

void GraphNode::FindRcrsvNghbrs_(GraphNode *root, DIRECTION dir,
                                 DirAcycGraph *graph) {
  LinkedList<GraphEdge> *nghbrLst = (dir == DIR_FRWRD) ? scsrLst_ : prdcsrLst_;

  color_ = COL_GRAY;

  // Iterate through the neighbor list of this node and recursively visit them
  // This recursion will bottom up when the root or leaf node is reached, which
  // then gets added to the very top of the recursive neighbor list.
  for (GraphEdge *crntEdge = nghbrLst->GetFrstElmnt(); crntEdge != NULL;
       crntEdge = nghbrLst->GetNxtElmnt()) {
    GraphNode *nghbr = crntEdge->GetOtherNode(this);
    if (nghbr->GetColor() == COL_WHITE) {
      nghbr->FindRcrsvNghbrs_(root, dir, graph);
    }
  }

  // When all the neighbors of this node have been recursively visited, the
  // node is finished. It is marked black and inserted in the recursive list.
  color_ = COL_BLACK;

  if (this != root) {

    // Check if there is a cycle in the graph
    BitVector *thisBitVector = this->GetRcrsvNghbrBitVector(dir);
    if (thisBitVector && thisBitVector->GetBit(root->GetNum()) == true) {
      graph->CycleDetected();
      Logger::Info("Detected a cycle between nodes %d and %d in graph",
                   root->GetNum(), this->GetNum());
    }

    // Add this node to the recursive neighbor list of the root node of
    // this search.
    root->GetRcrsvNghbrLst(dir)->InsrtElmnt(this);
    // Set the corresponding boolean vector entry to indicate that this
    // node is a recursive neighbor of the root node of this search.
    root->GetRcrsvNghbrBitVector(dir)->SetBit(num_);
  }
}

bool GraphNode::IsScsrEquvlnt(GraphNode *othrNode) {
  UDT_GLABEL thisLbl = 0;
  UDT_GLABEL othrLbl = 0;

  if (othrNode == this)
    return true;

  if (GetScsrCnt() != othrNode->GetScsrCnt())
    return false;

  for (GraphNode *thisScsr = GetFrstScsr(thisLbl),
                 *othrScsr = othrNode->GetFrstScsr(othrLbl);
       thisScsr != NULL; thisScsr = GetNxtScsr(thisLbl),
                 othrScsr = othrNode->GetNxtScsr(othrLbl)) {
    if (thisScsr != othrScsr || thisLbl != othrLbl)
      return false;
  }

  return true;
}

bool GraphNode::IsPrdcsrEquvlnt(GraphNode *othrNode) {
  UDT_GLABEL thisLbl = 0;
  UDT_GLABEL othrLbl = 0;

  if (othrNode == this)
    return true;

  if (GetPrdcsrCnt() != othrNode->GetPrdcsrCnt())
    return false;

  // TODO(austin) Find out why the first call to GetFrstPrdcsr returns the node
  // itself
  GraphNode *thisPrdcsr = GetFrstPrdcsr(thisLbl);
  GraphNode *othrPrdcsr = othrNode->GetFrstPrdcsr(othrLbl);
  if (thisPrdcsr == NULL)
    return true;
  for (thisPrdcsr = GetNxtPrdcsr(thisLbl),
      othrPrdcsr = othrNode->GetNxtPrdcsr(othrLbl);
       thisPrdcsr != NULL; thisPrdcsr = GetNxtPrdcsr(thisLbl),
      othrPrdcsr = othrNode->GetNxtPrdcsr(othrLbl)) {
    if (thisPrdcsr != othrPrdcsr || thisLbl != othrLbl)
      return false;
  }

  return true;
}

GraphEdge *GraphNode::FindScsr(GraphNode *trgtNode) {
  GraphEdge *crntEdge;

  // Linear search for the target node in the current node's adjacency list.
  for (crntEdge = scsrLst_->GetFrstElmnt(); crntEdge != NULL;
       crntEdge = scsrLst_->GetNxtElmnt()) {
    if (crntEdge->GetOtherNode(this) == trgtNode)
      return crntEdge;
  }

  return NULL;
}

GraphEdge *GraphNode::FindPrdcsr(GraphNode *trgtNode) {
  GraphEdge *crntEdge;

  // Linear search for the target node in the current node's adjacency list
  for (crntEdge = prdcsrLst_->GetFrstElmnt(); crntEdge != NULL;
       crntEdge = prdcsrLst_->GetNxtElmnt())
    if (crntEdge->GetOtherNode(this) == trgtNode) {
      return crntEdge;
    }

  return NULL; // not found in the neighbor list
}

void GraphNode::PrntScsrLst(FILE *outFile) {
  for (GraphEdge *crnt = scsrLst_->GetFrstElmnt(); crnt != NULL;
       crnt = scsrLst_->GetNxtElmnt()) {
    UDT_GNODES othrNodeNum = crnt->GetOtherNode(this)->GetNum();
    UDT_GNODES label = crnt->label;
    fprintf(outFile, "%d,%d  ", othrNodeNum + 1, label);
  }
  fprintf(outFile, "\n");
}

void GraphNode::LogScsrLst() {
  Logger::Info("Successor List For Node #%d", num_);
  for (GraphNode *thisScsr = GetFrstScsr(); thisScsr != NULL;
       thisScsr = GetNxtScsr()) {
    Logger::Info("%d", thisScsr->GetNum());
  }
}

DirAcycGraph::DirAcycGraph() {
  nodeCnt_ = 0;
  edgeCnt_ = 0;
  nodes_ = NULL;
  maxScsrCnt_ = 0;
  root_ = leaf_ = NULL;
  tplgclOrdr_ = NULL;
  dpthFrstSrchDone_ = false;
  cycleDetected_ = false;
}

DirAcycGraph::~DirAcycGraph() {
  if (tplgclOrdr_ != NULL)
    delete[] tplgclOrdr_;
}

void DirAcycGraph::CreateEdge_(UDT_GNODES frmNodeNum, UDT_GNODES toNodeNum,
                               UDT_GLABEL label) {
  GraphEdge *newEdg;

  assert(frmNodeNum < nodeCnt_);
  GraphNode *frmNode = nodes_[frmNodeNum];
  assert(frmNode != NULL);

  assert(toNodeNum < nodeCnt_);
  GraphNode *toNode = nodes_[toNodeNum];
  assert(toNode != NULL);

  newEdg = new GraphEdge(frmNode, toNode, label);

  frmNode->AddScsr(newEdg);
  toNode->AddPrdcsr(newEdg);
}

FUNC_RESULT DirAcycGraph::DepthFirstSearch() {
  if (tplgclOrdr_ == NULL)
    tplgclOrdr_ = new GraphNode *[nodeCnt_];

  for (UDT_GNODES i = 0; i < nodeCnt_; i++) {
    nodes_[i]->SetColor(COL_WHITE);
  }

  UDT_GNODES tplgclIndx = nodeCnt_ - 1;
  root_->DepthFirstVisit(tplgclOrdr_, tplgclIndx);

  if (tplgclIndx != -1) {
    Logger::Error("Invalid DAG Format: Ureachable nodes");
    return RES_ERROR;
  }

  dpthFrstSrchDone_ = true;
  return RES_SUCCESS;
}

FUNC_RESULT DirAcycGraph::FindRcrsvNghbrs(DIRECTION dir) {
  for (UDT_GNODES i = 0; i < nodeCnt_; i++) {
    GraphNode *node = nodes_[i];

    // Set the colors of all nodes to white (not visited yet) before starting
    // each recursive search.
    for (UDT_GNODES j = 0; j < nodeCnt_; j++) {
      nodes_[j]->SetColor(COL_WHITE);
    }

    node->AllocRcrsvInfo(dir, nodeCnt_);

    node->FindRcrsvNghbrs(dir, this);

    assert((dir == DIR_FRWRD &&
            node->GetRcrsvNghbrLst(dir)->GetFrstElmnt() == leaf_) ||
           (dir == DIR_BKWRD &&
            node->GetRcrsvNghbrLst(dir)->GetFrstElmnt() == root_) ||
           node == root_ || node == leaf_);
    assert(node != root_ ||
           node->GetRcrsvNghbrLst(DIR_FRWRD)->GetElmntCnt() == nodeCnt_ - 1);
    assert(node != leaf_ ||
           node->GetRcrsvNghbrLst(DIR_FRWRD)->GetElmntCnt() == 0);
  }

  if (cycleDetected_)
    return RES_ERROR;
  else
    return RES_SUCCESS;
}

void DirAcycGraph::Print(FILE *outFile) {
  fprintf(outFile, "Number of Nodes= %d    Number of Edges= %d\n", nodeCnt_,
          edgeCnt_);

  for (UDT_GNODES i = 0; i < nodeCnt_; i++) {
    fprintf(outFile, "%d:  ", i + 1);
    nodes_[i]->PrntScsrLst(outFile);
  }
}

void DirAcycGraph::LogGraph() {
  Logger::Info("Number of Nodes= %d   Number of Edges= %d\n", nodeCnt_,
               edgeCnt_);
  for (UDT_GNODES i = 0; i < nodeCnt_; i++) {
    nodes_[i]->LogScsrLst();
  }
}
