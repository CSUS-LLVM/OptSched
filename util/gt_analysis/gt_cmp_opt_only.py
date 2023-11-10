#!/usr/bin/env python3

import argparse

import analyze
from analyze import Logs, utils
from analyze.lib import block_stats
from gt_analysis import gt_cmp

is_optimal = block_stats.is_optimal


def compute_stats(nogt: Logs, gt: Logs):
    nogt, gt = utils.zipped_keep_blocks_if(nogt, gt, pred=is_optimal)
    return gt_cmp.compute_stats(nogt, gt)


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
