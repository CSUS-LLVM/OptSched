#!/usr/bin/env python3

import argparse
import os
import re
import pathlib

import analyze.logs as logs
import analyze.import_utils as import_utils


def parse(path):
    assert os.path.isdir(path), \
        'Point to the SHOC output directory (not the Logs/)'

    benchmark_output_dir = pathlib.Path(path)
    benchmark_logs_dir = benchmark_output_dir / 'Logs'
    assert benchmark_logs_dir.is_dir()

    benchmarks = list(sorted(benchmark_logs_dir.glob('dev?_*.err'), key=lambda p: p.name))

    result = logs.Logs([])
    benchname_re = re.compile(r'dev._(.*)(\.err)?')

    for benchmark in benchmarks:
        with benchmark.open('r') as f:
            benchname = benchname_re.search(benchmark.stem).group(1)
            result.merge(
                import_utils.parse_single_bench_file(
                    f.read(), benchname=benchname)
            )

    return result


if __name__ == '__main__':
    import_utils.import_main(
        parse,
        description='Import SHOC directories',
    )
