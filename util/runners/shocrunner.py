#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict


def main(shocdriver: Path, outdir: Path, shoc_args: Dict[str, str]):
    outdir = outdir.resolve()
    if not outdir.exists():
        outdir.mkdir()

    cmd = [str(shocdriver), '-opencl']
    for k, v in shoc_args.items():
        cmd.append(k)
        cmd.append(v)

    subprocess.run(cmd + ['-benchmark', 'FFT'], check=True, cwd=outdir)
    subprocess.run(cmd + ['-benchmark', 'GEMM'], check=True, cwd=outdir)
    subprocess.run(cmd + ['-benchmark', 'MD'], check=True, cwd=outdir)
    subprocess.run(cmd + ['-benchmark', 'Sort'], check=True, cwd=outdir)
    subprocess.run(cmd + ['-benchmark', 'Spmv'], check=True, cwd=outdir)
    subprocess.run(cmd + ['-benchmark', 'Stencil2D'], check=True, cwd=outdir)


if __name__ == '__main__':
    SHOCDRIVER = os.getenv('SHOCDRIVER')
    parser = argparse.ArgumentParser(description='Run the SHOC benchmarks')
    parser.add_argument('--shocdriver', default=SHOCDRIVER, required=SHOCDRIVER is None,
                        help='The path the the shocdriver executable')
    parser.add_argument('-o', '--outdir', required=True, help='The path to place the output files at')
    parser.add_argument('-s', '--shoc-problem-size', default='4',
                        help='The SHOC problem size, passed on to the shocdriver')

    args = parser.parse_args()

    main(
        shocdriver=Path(args.shocdriver),
        outdir=Path(args.outdir),
        shoc_args={'-s': args.shoc_problem_size},
    )
