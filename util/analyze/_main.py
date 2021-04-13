import pickle
import argparse
import json
import fnmatch
from typing import Callable

from .imports import *
from ._types import Block, Logs


def __load_file(file):
    '''
    Load imported log file (imported via one of the import scripts)
    '''
    return pickle.load(file)


def __load_filepath(filepath):
    with open(filepath, 'rb') as f:
        return __load_file(f)


def block_filter(filter: dict) -> Callable[[Block], bool]:
    def log_matches(log, pattern):
        if not isinstance(pattern, dict):
            if isinstance(pattern, str):
                return fnmatch.fnmatchcase(str(log), pattern)
            return log == pattern

        return all(
            k in log and log_matches(log[k], v)
            for k, v in pattern.items()
        )

    def blk_filter_f(blk):
        return all(
            event in blk and all(log_matches(log, matcher)
                                 for log in blk[event])
            for event, matcher in filter.items()
        )

    return blk_filter_f


def parse_args(parser: argparse.ArgumentParser, *names, args=None):
    '''
    Parses the argument parser with additional common flags.

    Use parse_args(parser) instead of parser.parse_args()

    Params:
      - *names - variadic: the strings specifying which arguments should be parsed.
                 These should be python_case, not --flag-case.
      - args - The argv to parse from. Defaults to parsing sys.argv
    '''

    parser.add_argument(
        '--benchsuite',
        required=True,
        choices=('spec', 'plaidml', 'shoc', 'pickle'),
        help='Select the benchmark suite which the input satisfies.',
    )
    parser.add_argument(
        '--keep-blocks-if',
        default='true',
        type=json.loads,
        help='Keep blocks matching (JSON format)',
    )

    args = parser.parse_args(args)

    FILE_PARSERS = {
        'pickle': __load_filepath,
        'spec': import_cpu2006.parse,
        'plaidml': import_plaidml.parse,
        'shoc': import_shoc.parse,
    }
    parser = FILE_PARSERS[args.benchsuite]
    blk_filter = block_filter(args.keep_blocks_if) if args.keep_blocks_if is not True else True

    args_dict = vars(args)

    # Go through the logs inputs and parse them.
    for name in names:
        result = parser(args_dict[name])
        if blk_filter is not True:
            result = result.keep_blocks_if(blk_filter)
        args_dict[name] = result

    return args
