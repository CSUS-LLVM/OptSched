#!/usr/bin/env python3

import argparse
import re
import sys
from itertools import chain
from typing import Iterable, List

import analyze
from analyze import Block, ioutils, utils

'''
Function-level stats (not Block, Logs, or Benchmark level)
'''

_RE_OCCUPANCY = re.compile(r'Final occupancy for function (?P<name>\S+):(?P<value>\d+)')


def _occupancy_info_in_block_log(block: Block) -> Iterable[int]:
    for m in _RE_OCCUPANCY.finditer(block.raw_log):
        yield int(m['value'])


def function_occupancy_info(logs: Iterable[Block]) -> List[int]:
    return list(chain.from_iterable(map(_occupancy_info_in_block_log, logs)))


def avg_occupancy(logs: Iterable[Block]) -> float:
    occ_info = function_occupancy_info(logs)
    return sum(occ_info) / len(occ_info) if occ_info else 0.0


def raw_main(argv: List[str] = []):
    parser = argparse.ArgumentParser(
        description='Computes the block stats for the logs')
    parser.add_argument('--stat', required=True, choices=('occ',),
                        help='Which stat to compute')
    parser.add_argument('logs', help='The logs to analyze')
    ioutils.add_output_format_arg(parser)
    args = analyze.parse_args(parser, 'logs')

    STATS = {
        'occ': ('Average Occupancy', avg_occupancy),
    }
    label, f = STATS[args.stat]

    results = utils.foreach_bench(lambda bench: {label: f(bench)}, args.logs)

    args.format(sys.stdout, results)


if __name__ == '__main__':
    raw_main(sys.argv)
