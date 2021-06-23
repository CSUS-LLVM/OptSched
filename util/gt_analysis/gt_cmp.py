#!/usr/bin/env python3

import argparse
from typing import Tuple

import analyze
from analyze import Block, Logs, utils
from analyze.lib import block_stats, compile_times

sched_time = compile_times.sched_time


def blocks_enumerated_optimally(blocks):
    return [blk for blk in blocks if 'DagSolvedOptimally' in blk or 'HeuristicScheduleOptimal' in blk]


def rp_ilp_gt_elapsed_for_blk(blk: Block) -> int:
    if 'GraphTransOccupancyPreservingILPNodeSuperiority' not in blk:
        return 0
    return blk.single('GraphTransOccupancyPreservingILPNodeSuperiorityFinished')['time'] \
        - blk.single('GraphTransOccupancyPreservingILPNodeSuperiority')['time']


def rp_only_gt_elapsed_for_blk(blk: Block) -> int:
    if 'GraphTransRPNodeSuperiority' not in blk:
        return 0
    return blk.single('GraphTransRPNodeSuperiorityFinished')['time'] \
        - blk.single('GraphTransRPNodeSuperiority')['time']


def ilp_only_gt_elapsed_for_blk(blk: Block) -> int:
    if 'GraphTransILPNodeSuperiority' not in blk:
        return 0
    return blk.single('GraphTransILPNodeSuperiorityFinished')['time'] \
        - blk.single('GraphTransILPNodeSuperiority')['time']


def raw_gt_elapsed_for_blk(blk: Block) -> int:
    return rp_ilp_gt_elapsed_for_blk(blk) \
        + rp_only_gt_elapsed_for_blk(blk) \
        + ilp_only_gt_elapsed_for_blk(blk)


def total_gt_elapsed_for_blk(blk: Block) -> int:
    if 'GraphTransformationsStart' not in blk:
        return 0
    return blk.single('GraphTransformationsFinished')['time'] \
        - blk.single('GraphTransformationsStart')['time']


def elapsed_before_enumeration_for_blk(blk: Block) -> int:
    assert 'CostLowerBound' in blk
    return blk['CostLowerBound'][-1]['time']


def enum_time_for_blk(blk: Block) -> int:
    if 'DagSolvedOptimally' not in blk:
        return blk.single('DagTimedOut')['time'] - blk['Enumerating'][0]['time']
    return blk.single('DagSolvedOptimally')['time'] - blk['Enumerating'][0]['time']


def cost_for_blk(blk: Block) -> int:
    return blk.single('BestResult')['cost'] + blk['CostLowerBound'][-1]['cost']


def is_improved(before: Block, after: Block):
    return cost_for_blk(before) > cost_for_blk(after)


def blk_relative_cost(nogt, gt) -> Tuple[int, int]:
    no_sum = yes_sum = 0
    for no, yes in zip(nogt, gt):
        no_cost = block_stats.block_cost(no)
        yes_cost = block_stats.block_cost(yes)
        no_lb = block_stats.block_cost_lower_bound(no)
        yes_lb = block_stats.block_cost_lower_bound(yes)
        assert no_lb <= yes_lb

        # relative to the tightest LB we know
        no_rel = no_cost - yes_lb
        yes_rel = yes_cost - yes_lb

        no_sum += no_rel
        yes_sum += yes_rel

    return no_sum, yes_sum


def compute_stats(nogt: Logs, gt: Logs, *, pass_num: int, total_compile_time_seconds):
    nogt_all, gt_all = nogt, gt

    if pass_num is not None:
        nogt = nogt.keep_blocks_if(lambda b: b.single('PassFinished')['num'] == pass_num)
        gt = gt.keep_blocks_if(lambda b: b.single('PassFinished')['num'] == pass_num)

    NUM_PROVED_OPTIMAL_WITHOUT_ENUMERATING = utils.count(utils.zipped_keep_blocks_if(
        nogt, gt, pred=lambda a, b: block_stats.is_enumerated(a) and not block_stats.is_enumerated(b))[0])
    nogt, gt = utils.zipped_keep_blocks_if(
        nogt, gt, pred=block_stats.is_enumerated)

    nogt_opt, gt_opt = utils.zipped_keep_blocks_if(nogt, gt, pred=lambda b: 'DagSolvedOptimally' in b)

    nogt_rel, gt_rel = blk_relative_cost(nogt, gt)

    result = {
        'Total Blocks in Benchsuite': utils.count(nogt_all),
        'Num Blocks enumerated with & without GT': utils.count(nogt),
        'Num Blocks proved optimal just by GT': NUM_PROVED_OPTIMAL_WITHOUT_ENUMERATING,

        'Total Compile Time (s) (all benchsuite) (No GT)': total_compile_time_seconds(nogt_all),
        'Total Compile Time (s) (all benchsuite) (GT)': total_compile_time_seconds(gt_all),
        'Total Sched Time (No GT)': sched_time(nogt),
        'Total Sched Time (GT)': sched_time(gt),
        'Enum Time (No GT)': utils.sum_stat_for_all(enum_time_for_blk, nogt),
        'Enum Time (GT)': utils.sum_stat_for_all(enum_time_for_blk, gt),
        'Total GT Time': utils.sum_stat_for_all(total_gt_elapsed_for_blk, gt),

        'Total Sched Time (opt. blks only) (No GT)': sched_time(nogt_opt),
        'Total Sched Time (opt. blks only) (GT)': sched_time(gt_opt),
        'Enum Time (opt. blocks only) (No GT)': utils.sum_stat_for_all(enum_time_for_blk, nogt_opt),
        'Enum Time (opt. blocks only) (GT)': utils.sum_stat_for_all(enum_time_for_blk, gt_opt),

        'Block Cost - Relative (No GT)': nogt_rel,
        'Block Cost - Relative (GT)': gt_rel,
        'Block Cost (No GT)': utils.sum_stat_for_all(block_stats.block_cost, nogt),
        'Block Cost (GT)': utils.sum_stat_for_all(block_stats.block_cost, gt),

        'Num Nodes Examined (opt. blocks only) (No GT)': utils.sum_stat_for_all(block_stats.nodes_examined_for_blk, nogt_opt),
        'Num Nodes Examined (opt. blocks only) (GT)': utils.sum_stat_for_all(block_stats.nodes_examined_for_blk, gt_opt),

        'Num Timeout Unimproved (No GT)': utils.count(blk for blk in nogt
                                                      if block_stats.is_timed_out(blk)
                                                      and not block_stats.is_improved(blk)),
        'Num Timeout Unimproved (GT)': utils.count(blk for blk in gt
                                                   if block_stats.is_timed_out(blk)
                                                   and not block_stats.is_improved(blk)),
        'Num Timeout Improved (No GT)': utils.count(blk for blk in nogt
                                                    if block_stats.is_timed_out(blk)
                                                    and block_stats.is_improved(blk)),
        'Num Timeout Improved (GT)': utils.count(blk for blk in gt
                                                 if block_stats.is_timed_out(blk)
                                                 and block_stats.is_improved(blk)),
    }

    return result


if __name__ == "__main__":
    import sys
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument('nogt')
    parser.add_argument('gt')
    parser.add_argument('--pass-num', type=int, default=None, help='Which pass to analyze (default: all passes)')
    args = analyze.parse_args(parser, 'nogt', 'gt')

    results = utils.foreach_bench(
        compute_stats, args.nogt, args.gt,
        pass_num=args.pass_num,
        total_compile_time_seconds=compile_times.total_compile_time_seconds_f(args.benchsuite),
    )

    writer = csv.DictWriter(sys.stdout,
                            fieldnames=['Benchmark'] + list(results['Total'].keys()))
    writer.writeheader()
    for bench, bench_res in results.items():
        writer.writerow({'Benchmark': bench, **bench_res})
