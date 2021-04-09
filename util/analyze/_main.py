import pickle
import argparse
import json
import fnmatch

from .imports import *


def __load_file(file):
    '''
    Load imported log file (imported via one of the import scripts)
    '''
    return pickle.load(file)


def __load_filepath(filepath):
    with open(filepath, 'rb') as f:
        return __load_file(f)


def parse_logs(filepath: str):
    FILE_PARSERS = {
        'pickle': __load_filepath,
        'spec': import_cpu2006.parse,
        'plaidml': import_plaidml.parse,
        'shoc': import_shoc.parse,
    }

    blk_filter = parse_logs._keep_blocks_if

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
            for event, matcher in blk_filter.items()
        )

    parser = FILE_PARSERS[parse_logs._benchsuite]

    return parser(filepath).keep_blocks_if(blk_filter_f)


def parse_args(parser: argparse.ArgumentParser, args=None):
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
    res, rest = firstparser.parse_known_args(args)
    if res.benchsuite:
        parse_logs._benchsuite = res.benchsuite
        parse_logs._keep_blocks_if = res.keep_blocks_if

    add_args(parser, real=True)
    result = parser.parse_args(args)

    return result
