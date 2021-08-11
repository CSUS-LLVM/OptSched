#!/usr/bin/env python3
import argparse
import enum
import re
import sys
from typing import IO, List
from pathlib import Path


class argfile:
    def __init__(self, default: IO, filename: Path, mode: str = 'r'):
        self.__file = default if filename == '-' else open(filename, mode)
        self.__should_close = filename != '-'

    def __enter__(self) -> IO:
        return self.__file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        if self.__should_close:
            self.__file.close()


class SpillStat(enum.Enum):
    RAW = re.compile(r'GREEDY RA: Number of spilled live ranges: ([0-9]+)')
    WEIGHTED = re.compile(r'SC in Function \S+ ([0-9]+)')


def sum_stat(infile: IO, r: re.Match) -> int:
    return sum(
        sum(int(x) for x in r.findall(line)) for line in infile
    )


def main(infile: IO, outfile: IO, which: SpillStat = SpillStat.RAW):
    spill_count = sum_stat(infile, which.value)
    print(spill_count, file=outfile)


def raw_main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Extract spill counts')
    parser.add_argument('--which', default='raw', choices=('weighted', 'raw'),
                        help='Whether to extract weighted or raw spills only. Default: raw')
    parser.add_argument('-o', '--output', default='-',
                        help='Where to output the information to, - for stdout. Defaults to stdout')
    parser.add_argument('file', help='The file to process, - for stdin.')

    args = parser.parse_args(argv)

    with argfile(sys.stdin, args.file, 'r') as infile, \
            argfile(sys.stdout, args.output, 'w') as outfile:
        main(infile, outfile, SpillStat[args.which.upper()])


if __name__ == '__main__':
    raw_main(sys.argv[1:])
