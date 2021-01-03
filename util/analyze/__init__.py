from .analyzer import Analyzer
from .logs import Logs, Benchmark, Block

from . import import_cpu2006
from . import import_utils

import sys

__all__ = ['logs', 'Analyzer', 'Logs', 'Benchmark', 'Block']


def load_file(file):
    '''
    Load imported log file (imported via one of the import scripts)
    '''
    import pickle
    return pickle.load(file)


def load_filepath(filepath):
    with open(filepath, 'rb') as f:
        return load_file(f)


def main(analyzer: Analyzer):
    import argparse
    parser = argparse.ArgumentParser(description=analyzer.__doc__)

    for name, help in analyzer.POSITIONAL:
        parser.add_argument(name, help=help)

    parser.add_argument(
        '--benchsuite',
        default=None,
        choices=('spec',),
        help='Select the benchmark suite which the input satisfies. Valid options: spec',
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Where to output the report',
    )

    args = parser.parse_args()
    options = {name: getattr(args, name) for name in analyzer.OPTIONS}
    pos = [getattr(args, name) for name, help in analyzer.POSITIONAL]

    FILE_PARSERS = {
        None: load_filepath,
        'spec': import_cpu2006.parse,
    }
    parser = FILE_PARSERS[args.benchsuite]

    pos_data = [parser(f) for f in pos]

    analyzer.run(pos_data)
    if args.output is None:
        analyzer.print_report(sys.stdout, options=options)
    else:
        with open(args.output, 'w') as outfile:
            analyzer.print_report(outfile, options=options)
