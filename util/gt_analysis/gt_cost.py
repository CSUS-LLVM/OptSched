#!/usr/bin/env python3

import analyze
from analyze import Block


def _rp_gt_elapsed(blk: Block):
    if 'GraphTransRPNodeSuperiority' not in blk:
        return 0
    return blk.single('GraphTransRPNodeSuperiorityFinished')['time'] \
        - blk.single('GraphTransRPNodeSuperiority')['time']


def _cost_before_enum(blk: Block):
    start = blk.single('ProcessDag')['time']
    if 'Enumerating' in blk:
        return blk['Enumerating'][0]['time'] - start
    return blk.single('ScheduleVerifiedSuccessfully')['time'] - start


class RpGtCost(analyze.Analyzer):
    '''
    Computes the compile time cost of Register Pressure Graph Transformations.
    '''

    POSITIONAL = {
        'nogt': 'Logs without Graph Transformations',
        'gt': 'Logs with Graph Transformations'
    }.items()

    def run_bench(self, args):
        nogt = args[0]
        gt = args[1]

        gt_cost = sum(_rp_gt_elapsed(blk) for blk in gt.all_blocks())
        gt_cost_before_enum = sum(_cost_before_enum(blk) for blk in gt.all_blocks())
        nogt_cost_before_enum = sum(_cost_before_enum(blk) for blk in nogt.all_blocks())
        gt_total_cost = gt_cost_before_enum - nogt_cost_before_enum

        self.stat(
            gt_cost=gt_cost,
            gt_total_cost=gt_total_cost,
        )


if __name__ == '__main__':
    analyze.main(RpGtCost)
