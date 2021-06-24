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
_PLAIDML_TIME_ELAPSED = re.compile(
    r'Example finished, elapsed: (?P<elapsed>\S+)s \(compile\), (?P<exec>\S+)s \(execution\)')
_SHOC_TIME_ELAPSED = re.compile(r'Finished compiling; total ns = (?P<elapsed>\d+)')


def shoc_total_compile_time_seconds(logs):
    try:
        elapsed = sum(int(m['elapsed'])
                   for bench in logs.benchmarks
                   for blk in bench
                   for m in _SHOC_TIME_ELAPSED.finditer(blk.raw_log))
        return float(elapsed) * 1e-9
    except TypeError:
        raise KeyError('Logs must contain "Finished compiling; total ns = " output by the modified SHOC benchmark suite')


def plaidml_total_compile_time_seconds(logs):
    try:
        return sum(float(_PLAIDML_TIME_ELAPSED.search(bench.blocks[-1].raw_log)['elapsed']) for bench in logs.benchmarks)
    except TypeError:
        raise KeyError('Logs must contain "Example finished, elapsed:" output by the PlaidML benchmark suite')


def total_compile_time_seconds(logs):
    last_blk = logs.benchmarks[-1].blocks[-1]
    last_logs = last_blk.raw_log
    m = [g for g in _CPU2017_TIME_ELAPSED.finditer(last_logs)
         if last_blk.benchmark == g['bench']]

    if m:
        if len(m) != 1:
            logging.warning('Multiple CPU2017 elapsed time indicators. Using the first one out of: %s', m)
        return int(m[0]['elapsed'])

    m = _BACKUP_TIME_ELAPSED.search(last_logs)
    assert m, \
        'Logs must contain "total seconds elapsed" output by the SPEC benchmark suite'

    return int(m['elapsed'])


def total_compile_time_seconds_f(benchsuite):
    return {
        'spec': total_compile_time_seconds,
        'plaidml': plaidml_total_compile_time_seconds,
        'shoc': shoc_total_compile_time_seconds,
    }[benchsuite]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('sched', 'total', 'plaidml'),
                        help='Which timing variant to use')
    parser.add_argument('logs', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs')

    fn = {
        'sched': sched_time,
        'total': total_compile_time_seconds,
        'plaidml': plaidml_total_compile_time_seconds,
        'shoc': shoc_total_compile_time_seconds,
    }[args.variant]
    results = foreach_bench(fn, args.logs, combine=sum)
    writer = csv.DictWriter(sys.stdout, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)
