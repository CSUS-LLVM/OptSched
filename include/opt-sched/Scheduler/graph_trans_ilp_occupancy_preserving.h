/*******************************************************************************
Description:  Implement graph transformations to be applied before scheduling.
Author:       Justin Bassett
Created:      Aug. 2020
Last Update:  Aug. 2020
*******************************************************************************/

#ifndef OPTSCHED_BASIC_GRAPH_TRANS_ILP_OCCUPANCY_PRESERVING_H
#define OPTSCHED_BASIC_GRAPH_TRANS_ILP_OCCUPANCY_PRESERVING_H

#include "opt-sched/Scheduler/graph_trans.h"

namespace llvm {
namespace opt_sched {

// Node superiority Occupancy preserving ILP graph transformation.
class StaticNodeSupOccupancyPreservingILPTrans : public GraphTrans {
public:
  StaticNodeSupOccupancyPreservingILPTrans(DataDepGraph *dataDepGraph);

  const char *Name() const override {
    return "occupancy-preserving-ilp.nodesup";
  }

  FUNC_RESULT ApplyTrans() override;
};

} // namespace opt_sched
} // namespace llvm

#endif
