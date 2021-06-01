#!/usr/bin/env python3

from typing import *
import argparse
import analyze
from analyze import Block, Logs, utils


def is_enumerated(blk: Block) -> bool:
    return 'Enumerating' in blk


def is_optimal(blk: Block) -> bool:
    return 'DagSolvedOptimally' in blk or 'HeuristicScheduleOptimal' in blk


def is_timed_out(blk: Block) -> bool:
    return 'DagTimedOut' in blk


def block_cost_lower_bound(blk: Block) -> int:
    return blk.single('CostLowerBound')['cost']


def block_relative_cost(blk: Block) -> int:
    return blk.single('BestResult')['cost']


def block_best_length(blk: Block) -> int:
    return blk.single('BestResult')['length']


def block_cost(blk: Block) -> int:
    return block_cost_lower_bound(blk) + block_relative_cost(blk)


def cost_improvement_for_blk(blk: Block) -> int:
    if 'DagSolvedOptimally' in blk:
        return blk.single('DagSolvedOptimally')['cost_improvement']
    elif 'DagTimedOut' in blk:
        return blk.single('DagTimedOut')['cost_improvement']
    else:
        return 0


def is_improved(blk: Block) -> bool:
    return cost_improvement_for_blk(blk) > 0


def nodes_examined_for_blk(blk: Block) -> int:
    return blk.single('NodeExamineCount')['num_nodes'] if 'NodeExamineCount' in blk else 0


def num_blocks(logs: Logs) -> int:
    return sum(len(bench.blocks) for bench in logs.benchmarks)


def num_enumerated(logs: Logs) -> int:
    return sum(1 for blk in logs if is_enumerated(blk))


def nodes_examined(logs: Logs) -> int:
    return sum(nodes_examined_for_blk(blk) for blk in logs)


def compute_block_stats(logs: Logs):
    return {
        'num blocks': num_blocks(logs),
        'num blocks enumerated': num_enumerated(logs),
        'num optimal and improved': utils.count(blk for blk in logs if is_optimal(blk) and is_improved(blk) and is_enumerated(blk)),
        'num optimal and not improved': utils.count(blk for blk in logs if is_optimal(blk) and not is_improved(blk) and is_enumerated(blk)),
        'num not optimal and improved': utils.count(blk for blk in logs if not is_optimal(blk) and is_improved(blk) and is_enumerated(blk)),
        'num not optimal and not improved': utils.count(blk for blk in logs if not is_optimal(blk) and not is_improved(blk) and is_enumerated(blk)),
        'nodes examined': nodes_examined(logs),
    }


if __name__ == '__main__':
    import sys
    import csv

    parser = argparse.ArgumentParser(
        description='Computes the block stats for the logs')
    parser.add_argument('logs', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs')

    results = utils.foreach_bench(compute_block_stats, args.logs)

    writer = csv.DictWriter(sys.stdout,
                            fieldnames=['Benchmark'] + list(results['Total'].keys()))
    writer.writeheader()
    for bench, bench_res in results.items():
        writer.writerow({'Benchmark': bench, **bench_res})
