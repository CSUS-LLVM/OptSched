#!/usr/bin/env python3

import os
import pathlib

from .._types import Logs, merge_logs
from . import import_utils


def parse(path):
    assert os.path.isdir(path), 'Point to the plaidbench output directory'

    benchmark_output_dir = pathlib.Path(path)
    benchmark_dirs = [x for x in benchmark_output_dir.iterdir() if x.is_dir()]
    benchmark_dirs = list(sorted(benchmark_dirs, key=lambda p: p.name))

    result = Logs([])

    for benchmark_dir in benchmark_dirs:
        logfiles = list(benchmark_dir.glob('*.log'))
        assert len(logfiles) == 1

        with logfiles[0].open('r') as f:
            benchname = benchmark_dir.stem
            merge_logs(
                result,
                import_utils.parse_single_bench_file(
                    f.read(), benchname=benchname)
            )

    return result


if __name__ == '__main__':
    import_utils.import_main(
        parse,
        description='Import plaidbench directories',
    )
