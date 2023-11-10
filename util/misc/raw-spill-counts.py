#!/usr/bin/env python3
import argparse
import enum
import re
import sys
from typing import Callable, IO, List
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def argfile(filename: str, mode: str):
    if filename == '-':
        yield sys.stdin if mode == 'r' else sys.stdout
    else:
        with open(filename, mode) as f:
            yield f


class SpillStat(enum.Enum):
    RAW = re.compile(r'Function: (?P<name>\S*?)\nGREEDY RA: Number of spilled live ranges: (?P<value>\d+)')
    WEIGHTED = re.compile(r'SC in Function (?P<name>\S*?) (?P<value>-?\d+)')


def _sum_stat(s, r: re.Match, fn_filter: Callable[[str, int], bool]) -> int:
    return sum(int(m['value']) for m in r.finditer(s) if fn_filter(m['name'], int(m['value'])))


def sum_stat(infile: IO, r: re.Match, *, fn_filter: Callable[[str, int], bool] = lambda k, v: True) -> int:
    try:
        pos = infile.tell()
        return _sum_stat(infile.read(), r, fn_filter)
    except MemoryError:
        infile.seek(pos)
        return sum(
            _sum_stat(line, r, fn_filter) for line in infile
        )


def main(infile: IO, outfile: IO, which: SpillStat = SpillStat.RAW, *, fn_filter: Callable[[str, int], bool] = lambda k, v: True):
    spill_count = sum_stat(infile, which.value, fn_filter=fn_filter)
    print(spill_count, file=outfile)


def raw_main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Extract spill counts')
    parser.add_argument('--which', default='raw', choices=('weighted', 'raw'),
                        help='Whether to extract weighted or raw spills only. Default: raw')
    parser.add_argument('--hot-only', help='A file with a space-separated list of functions to consider in the count')
    parser.add_argument('-o', '--output', default='-',
                        help='Where to output the information to, - for stdout. Defaults to stdout')
    parser.add_argument('file', help='The file to process, - for stdin.')

    args = parser.parse_args(argv)

    if args.hot_only:
        content = Path(args.hot_only).read_text()
        hot_fns = set(content.split())
        def fn_filter(k, v): return k in hot_fns
    else:
        def fn_filter(k, v): return True

    with argfile(args.file, 'r') as infile, \
            argfile(args.output, 'w') as outfile:
        main(infile, outfile, SpillStat[args.which.upper()], fn_filter=fn_filter)


if __name__ == '__main__':
    raw_main(sys.argv[1:])
