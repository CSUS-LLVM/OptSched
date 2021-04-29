#!/usr/bin/env python3

import argparse

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
    return blk.single('CostLowerBound')['time']


def enum_time_for_blk(blk: Block) -> int:
    if 'DagSolvedOptimally' not in blk:
        return 0
    return blk.single('DagSolvedOptimally')['time'] - blk['Enumerating'][0]['time']


def cost_for_blk(blk: Block) -> int:
    return blk.single('BestResult')['cost'] + blk.single('CostLowerBound')['cost']


def is_improved(before: Block, after: Block):
    return cost_for_blk(before) > cost_for_blk(after)


def compute_stats(nogt: Logs, gt: Logs):
    nogt_enum, gt_enum = utils.zipped_keep_blocks_if(
        nogt, gt, pred=block_stats.is_enumerated)

    result = {
        'num regions': utils.count(nogt),
        'nogt sched time': sched_time(nogt),
        'gt sched time': sched_time(gt),
        'nogt enum time': utils.sum_stat_for_all(enum_time_for_blk, nogt_enum),
        'gt enum time': utils.sum_stat_for_all(enum_time_for_blk, gt_enum),
        'nogt nodes examined': block_stats.nodes_examined(nogt_enum),
        'gt nodes examined': block_stats.nodes_examined(gt_enum),
        'nogt num enumerated': block_stats.num_enumerated(nogt_enum),
        'gt num enumerated': block_stats.num_enumerated(gt_enum),

        'nogt timeout unimproved': utils.count(blk for blk in nogt_enum
                                               if block_stats.is_timed_out(blk)
                                               and not block_stats.is_improved(blk)),
        'gt timeout unimproved': utils.count(blk for blk in gt_enum
                                             if block_stats.is_timed_out(blk)
                                             and not block_stats.is_improved(blk)),
        'nogt timeout improved': utils.count(blk for blk in nogt_enum
                                             if block_stats.is_timed_out(blk)
                                             and block_stats.is_improved(blk)),
        'gt timeout improved': utils.count(blk for blk in gt_enum
                                           if block_stats.is_timed_out(blk)
                                           and block_stats.is_improved(blk)),

        'total gt time': utils.sum_stat_for_all(total_gt_elapsed_for_blk, gt),

        'nogt sched time (enum only)': sched_time(nogt_enum),
        'gt sched time (enum only)': sched_time(gt_enum),
    }

    return result


if __name__ == "__main__":
    import sys
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument('nogt')
    parser.add_argument('gt')
    args = analyze.parse_args(parser, 'nogt', 'gt')

    results = utils.foreach_bench(compute_stats, args.nogt, args.gt)

    writer = csv.DictWriter(sys.stdout,
                            fieldnames=['Benchmark'] + list(results['Total'].keys()))
    writer.writeheader()
    for bench, bench_res in results.items():
        writer.writerow({'Benchmark': bench, **bench_res})
