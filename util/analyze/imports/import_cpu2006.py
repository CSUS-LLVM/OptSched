#!/usr/bin/env python3

import os

from . import import_utils


def parse(file):
    assert os.path.isfile(
        file), 'Only single-file CPU2006 logs supported at this time'

    with open(file, 'r') as f:
        return import_utils.parse_multi_bench_file(
            f.read(),
            benchstart=r'Building (?P<name>\S*)',
            filename=r'/[fc]lang\b.*\s(\S+\.\S+)\n')


if __name__ == '__main__':
    import_utils.import_main(
        parse,
        description='Import single-file CPU2006 logs',
    )
