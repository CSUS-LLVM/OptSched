import sys
import pickle
import argparse
import json
import fnmatch

from .analyzer import Analyzer

from . import import_cpu2006
from . import import_plaidml
from . import import_shoc
from . import import_utils


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
            event in blk and all(log_matches(log, matcher) for log in blk[event])
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
            choices=('spec', 'plaidml', 'shoc'),
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


def basemain(*, positional, options, description, action, manual_options={}):
    parser = argparse.ArgumentParser(description=description)

    for name, help in positional:
        parser.add_argument(name, help=help)

    for name, help in options.items():
        if name in manual_options:
            parser.add_argument(
                '--' + name, default=manual_options[name], help=help)
        else:
            parser.add_argument('--' + name, help=help)

    parser.add_argument(
        '--benchsuite',
        default=manual_options.get('benchsuite', None),
        choices=('spec', 'plaidml', 'shoc'),
        help='Select the benchmark suite which the input satisfies. Valid options: spec',
    )
    parser.add_argument(
        '-o', '--output',
        default=manual_options.get('output', None),
        help='Where to output the report',
    )

    args = parser.parse_args()
    option_values = {name: getattr(args, name) for name in options}
    pos = [getattr(args, name) for name, help in positional]

    FILE_PARSERS = {
        None: __load_filepath,
        'spec': import_cpu2006.parse,
        'plaidml': import_plaidml.parse,
        'shoc': import_shoc.parse,
    }
    parser = FILE_PARSERS[args.benchsuite]

    pos_data = [parser(f) for f in pos]

    if args.output is None:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    try:
        action(outfile, pos_data, option_values)
    finally:
        outfile.close()


def main(analyzer_cls, **manual_options):
    def main_action(outfile, pos_data, option_values):
        analyzer = analyzer_cls(**option_values)
        analyzer.run(pos_data)
        analyzer.print_report(outfile)

    basemain(
        positional=analyzer_cls.POSITIONAL,
        options=analyzer_cls.OPTIONS,
        description=analyzer_cls.__doc__,
        action=main_action,
        manual_options=manual_options,
    )
