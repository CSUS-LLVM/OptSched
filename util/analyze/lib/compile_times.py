#!/usr/bin/env python3

import csv
import re
import argparse
import sys
import logging

import analyze
from analyze import Block, foreach_bench


def _block_time(block: Block):
    end = block.single('ScheduleVerifiedSuccessfully')['time']
    start = block.single('ProcessDag')['time']
    return end - start


def sched_time(logs):
    return sum(_block_time(blk) for blk in logs)


_CPU2017_TIME_ELAPSED = re.compile(r"Elapsed compile for '(?P<bench>[^']+)': \S+ \((?P<elapsed>\d+)\)")
_BACKUP_TIME_ELAPSED = re.compile(r'(?P<elapsed>\d+) total seconds elapsed')


def total_compile_time_seconds(logs):
    last_blk = logs.benchmarks[-1].blocks[-1]
    last_logs = last_blk.raw_log
    m = [g for g in _CPU2017_TIME_ELAPSED.finditer(last_logs)
         if last_blk.benchmark == g['bench']]

    if m:
        if len(m) != 1:
            logging.warning('Multiple CPU2017 elapsed time indicators. Using the first one out of: %s', m)
        return m[0]['elapsed']

    m = _BACKUP_TIME_ELAPSED.search(last_logs)
    assert m, \
        'Logs must contain "total seconds elapsed" output by the SPEC benchmark suite'

    return m['elapsed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('sched', 'total'),
                        help='Which timing variant to use')
    parser.add_argument('logs', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs')

    fn = total_compile_time_seconds if args.variant == 'total' else sched_time
    results = foreach_bench(fn, args.logs, combine=sum)
    writer = csv.DictWriter(sys.stdout, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)
