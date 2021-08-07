#!/usr/bin/env python3

import argparse
import re
import sys
from itertools import chain
from typing import Callable, Iterable, List,  Pattern, Tuple

import analyze
from analyze import Block, ioutils, utils

'''
Function-level stats (not Block, Logs, or Benchmark level)
'''

_RE_OCCUPANCY = re.compile(r'Final occupancy for function (?P<name>\S+):(?P<value>\d+)')
_RE_SPILLS = re.compile(r'Function: (?P<name>\S*?)\nGREEDY RA: Number of spilled live ranges: (?P<value>\d+)')
_RE_SPILLS_WEIGHTED = re.compile(r'SC in Function (?P<name>\S*?) (?P<value>-?\d+)')


def compute_avg_values(fn_info: List[Tuple[str, int]], *, fn_filter: Callable[[str, int], bool] = lambda k, v: True) -> float:
    return utils.average((v for k, v in fn_info if fn_filter(k, v)), len(fn_info))


def _fn_re_info(re: Pattern, logs: Iterable[Block], key='name', value='value') -> Iterable[Tuple[str, int]]:
    for m in chain.from_iterable(re.finditer(blk.raw_log) for blk in logs):
        yield (m[key], int(m[value]))


def fn_occupancy_info(logs: Iterable[Block]) -> List[Tuple[str, int]]:
    return list(_fn_re_info(_RE_OCCUPANCY, logs))


def avg_occupancy(logs: Iterable[Block], *, fn_filter: Callable[[str, int], bool] = lambda k, v: True) -> float:
    occ_info = fn_occupancy_info(logs)
    return compute_avg_values(occ_info, fn_filter=fn_filter)


def fn_spill_info(logs: Iterable[Block]) -> List[Tuple[str, int]]:
    return list(_fn_re_info(_RE_SPILLS, logs))


def fn_weighted_spill_info(logs: Iterable[Block]) -> List[Tuple[str, int]]:
    return list(_fn_re_info(_RE_SPILLS_WEIGHTED, logs))


def total_spills(logs: Iterable[Block], *, fn_filter: Callable[[str, int], bool] = lambda k, v: True) -> int:
    return sum(v for k, v in fn_spill_info(logs) if fn_filter(k, v))


def total_weighted_spills(logs: Iterable[Block], *, fn_filter: Callable[[str, int], bool] = lambda k, v: True) -> int:
    return sum(v for k, v in fn_weighted_spill_info(logs) if fn_filter(k, v))


def raw_main(argv: List[str] = []):
    parser = argparse.ArgumentParser(
        description='Computes the block stats for the logs')
    parser.add_argument('--stat', required=True, choices=('occ', 'spills', 'weighted-spills'),
                        help='Which stat to compute')
    parser.add_argument('--hot-only', help='A file with a space-separated list of functions to consider in the count')
    ioutils.add_output_format_arg(parser)
    parser.add_argument('logs', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs', args=argv)

    if args.hot_only:
        with open(args.hot_only, 'r') as f:
            contents = f.read()
        fns = set(contents.split())
        def fn_filter(k, v): return k in fns
    else:
        def fn_filter(k, v): return True

    STATS = {
        'occ': ('Average Occupancy', avg_occupancy),
        'spills': ('Spill Count', total_spills),
        'weighted-spills': ('Weighted Spill Count', total_weighted_spills),
    }
    label, f = STATS[args.stat]

    results = utils.foreach_bench(lambda bench: {label: f(bench, fn_filter=fn_filter)}, args.logs)

    args.format(sys.stdout, results)


if __name__ == '__main__':
    raw_main(None)  # Default to sys.argv
