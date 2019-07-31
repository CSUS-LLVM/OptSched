/*******************************************************************************
Description:  Defines a history table class.
Author:       Ghassan Shobaki
Created:      Unknown
Last Update:  Mar. 2011
*******************************************************************************/

#ifndef OPTSCHED_ENUM_HIST_TABLE_H
#define OPTSCHED_ENUM_HIST_TABLE_H

#include "opt-sched/Scheduler/defines.h"
#include "opt-sched/Scheduler/enumerator.h"
#include "opt-sched/Scheduler/gen_sched.h"
#include "opt-sched/Scheduler/hash_table.h"
#include "opt-sched/Scheduler/mem_mngr.h"
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace llvm {
namespace opt_sched {

class EnumTreeNode;
class Enumerator;

// The history version of a tree node to be kept in the history table
class HistEnumTreeNode {
public:
  HistEnumTreeNode();
  virtual ~HistEnumTreeNode();

  InstCount GetTime();
  void PrntPartialSched(std::ostream &out);
  bool CompPartialScheds(HistEnumTreeNode *othrHist);
  InstCount GetInstNum();
  bool IsPrdcsrViaStalls(HistEnumTreeNode *othrNode);
  HistEnumTreeNode *GetParent();
  void Clean();
  void ReplaceParent(HistEnumTreeNode *newParent);
  // Does the scheduled inst. list of this node match that of the given node
  bool DoesMatch(EnumTreeNode *node, Enumerator *enumrtr);
  // Is the sub-problem at this node dominated by the given node's?
  bool IsDominated(EnumTreeNode *node, Enumerator *enumrtr);
  // Does the sub-problem at this node dominate the given node's?
  virtual bool DoesDominate(EnumTreeNode *node, Enumerator *enumrtr);
  virtual void Construct(EnumTreeNode *node, bool isTemp);
  virtual void SetCostInfo(EnumTreeNode *node, bool isTemp,
                           Enumerator *enumrtr);
  const std::shared_ptr<std::vector<SchedInstruction *>> &GetSuffix() const;
  void
  SetSuffix(const std::shared_ptr<std::vector<SchedInstruction *>> &suffix);
  std::vector<InstCount> GetPrefix() const;

protected:
  HistEnumTreeNode *prevNode_;

  // The current time or position (or step number) in the scheduling process.
  // This is equal to the length of the path from the root node to this node.
  InstCount time_;

  SchedInstruction *inst_;

#ifdef IS_DEBUG
  bool isCnstrctd_;
#endif

  bool crntCycleBlkd_;
  ReserveSlot *rsrvSlots_;

  // (Chris)
  std::shared_ptr<std::vector<SchedInstruction *>> suffix_ = nullptr;

  InstCount SetLastInsts_(SchedInstruction *lastInsts[], InstCount thisTime,
                          InstCount minTimeToExmn);
  void SetInstsSchduld_(BitVector *instsSchduld);
  // Does this history node dominate the given node or history node?
  bool DoesDominate_(EnumTreeNode *node, HistEnumTreeNode *othrHstry,
                     ENUMTREE_NODEMODE mode, Enumerator *enumrtr,
                     InstCount shft);
  void SetLwrBounds_(InstCount lwrBounds[], SchedInstruction *lastInsts[],
                     InstCount thisTime, InstCount minTimeToExmn,
                     Enumerator *enumrtr);
  void CmputNxtAvlblCycles_(Enumerator *enumrtr, InstCount instsPerType[],
                            InstCount nxtAvlblCycles[]);

  virtual void Init_();
  void AllocLastInsts_(ArrayMemAlloc<SchedInstruction *> *lastInstsAlctr,
                       Enumerator *enumrtr);
  bool IsAbslutDmnnt_();
  InstCount GetMinTimeToExmn_(InstCount nodeTime, Enumerator *enumrtr);
  InstCount GetLwrBound_(SchedInstruction *inst, int16_t issuRate);
  void SetRsrvSlots_(EnumTreeNode *node);
};

class CostHistEnumTreeNode : public HistEnumTreeNode {
public:
  CostHistEnumTreeNode();
  virtual ~CostHistEnumTreeNode();

  void Construct(EnumTreeNode *node, bool isTemp);
  // Does the sub-problem at this node dominate the given node's?
  bool DoesDominate(EnumTreeNode *node, Enumerator *enumrtr);
  void SetCostInfo(EnumTreeNode *node, bool isTemp, Enumerator *enumrtr);

protected:
  // Why do we need to copy this data from region->tree_node->hist_node
  InstCount cost_;
  InstCount peakSpillCost_;
  InstCount spillCostSum_;

  // (Chris)
  InstCount totalCost_ = -1;
  InstCount partialCost_ = -1;
  bool totalCostIsActualCost_ = false;

  bool isLngthFsbl_;
#ifdef IS_DEBUG
  bool costInfoSet_;
#endif

  bool ChkCostDmntnForBBSpill_(EnumTreeNode *node, Enumerator *enumrtr);
  bool ChkCostDmntn_(EnumTreeNode *node, Enumerator *enumrtr,
                     InstCount &maxShft);
  virtual void Init_();
};

} // namespace opt_sched
} // namespace llvm

#endif
