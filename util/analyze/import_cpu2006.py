#!/usr/bin/env python3

import argparse
import os
import re

import analyze.logs as logs
import analyze.import_utils as import_utils


def parse(file):
    assert os.path.isfile(
        file), 'Only single-file CPU2006 logs supported at this time'

    with open(file, 'r') as f:
        return import_utils.parse_multi_bench_file(
            f.read(),
            benchstart=re.compile(r'Building (?P<name>\S*)'),
            filename=re.compile(r'/[fc]lang\b.*\s(\S+\.\S+)\n'))


if __name__ == '__main__':
    import_utils.import_main(
        parse,
        description='Import single-file CPU2006 logs',
    )
