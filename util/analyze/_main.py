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


def parse_logs(filepath: str, blk_filter: Callable[[Block], bool] = None, parser: Callable[[str], Logs] = None) -> Logs:
    '''
    Parses the logfiles located at filepath according to the specified parser and blk_filter

    Intended to be used as the `type=analyze.parse_logs` argument to an ArgumentParser add_argument call.
    For that to work, you must use analyze.parse_args(parser) instead of parser.parse_args()
    '''

    if blk_filter is None:
        blk_filter = parse_logs._blk_filter
    if parser is None:
        parser = parse_logs._parser

    return parser(filepath).keep_blocks_if(blk_filter)


def parse_args(parser: argparse.ArgumentParser, args=None):
    '''
    Parses the argument parser with additional common flags, supporting `parse_logs`

    Use parse_args(parser) instead of parser.parse_args()
    '''

    firstparser = argparse.ArgumentParser(add_help=False)

    def add_args(parser, real):
        parser.add_argument(
            '--benchsuite',
            required=real,
            choices=('spec', 'plaidml', 'shoc', 'pickle'),
            help='Select the benchmark suite which the input satisfies.',
        )
        parser.add_argument(
            '--keep-blocks-if',
            default='{}',
            type=json.loads,
            help='Keep blocks matching (JSON format)',
        )

    add_args(firstparser, real=False)
    res, _ = firstparser.parse_known_args(args)
    if res.benchsuite: # Don't do anything if --help is passed
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
                for event, matcher in res.keep_blocks_if.items()
            )

        blk_filter = blk_filter_f

        FILE_PARSERS = {
            'pickle': __load_filepath,
            'spec': import_cpu2006.parse,
            'plaidml': import_plaidml.parse,
            'shoc': import_shoc.parse,
        }

        parse_logs._blk_filter = blk_filter
        parse_logs._parser = FILE_PARSERS[res.benchsuite]

    add_args(parser, real=True)
    return parser.parse_args(args)

    return result
