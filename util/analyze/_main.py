import argparse
import fnmatch
import json
from pathlib import Path
import pickle
import sys
from typing import Callable

from ._types import Block, Logs
from . import _cpp_types
from .imports import *


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
    parser.add_argument(
        '--use-c++',
        dest='use_cpp',
        action='store_true',
        help='Use the accelerated C++ parser. The eventparser module is expected to be on the PYTHONPATH',
    )
    parser.add_argument(
        '--c++-module',
        dest='cpp_module',
        type=Path,
        default=None,
        help='The path to the accelerated C++ parser module. --use-c++ is unnecessary if this is supplied.',
    )

    args = parser.parse_args(args)

    use_cpp = bool(args.use_cpp or args.cpp_module)

    if use_cpp and args.benchsuite != 'spec':
        print(f'WARNING: Unable to use the C++-accelerated parser for {args.benchsuite}', file=sys.stderr)

    def cpp_parse_blocks_fn():
        if args.cpp_module:
            import importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location('eventanalyze', args.cpp_module)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = __import__('eventanalyze')

        _cpp_types.load_module(mod)

        def parse(file):
            return _cpp_types.parse_blocks(file, mod.SPEC_BENCH_RE)

        return parse

    FILE_PARSERS = {
        'pickle': __load_filepath,
        'spec': cpp_parse_blocks_fn() if use_cpp else import_cpu2006.parse,
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
