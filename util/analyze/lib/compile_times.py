#!/usr/bin/env python3

import re
import argparse

import analyze
from analyze import Block


def _block_time(block: Block):
    end = block.single('ScheduleVerifiedSuccessfully')['time']
    start = block.single('ProcessDag')['time']
    return end - start


def instruction_scheduling_time(logs):
    return sum(_block_time(blk) for blk in logs)


def total_compile_time_seconds(logs):
    last_logs = logs.benchmarks[-1].blocks[-1].raw_log
    m = re.search(r'(\d+) total seconds elapsed', last_logs)
    assert m, \
        'Logs must contain "total seconds elapsed" output by the SPEC benchmark suite'

    return m.group(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('sched', 'total'), help='Which timing variant to use')
    parser.add_argument('logs', type=analyze.parse_logs, help='The logs to analyze')
    args = analyze.parse_args(parser)

    if args.variant == 'total':
        print(total_compile_time_seconds(args.logs))
    else:
        print(instruction_scheduling_time(args.logs))
