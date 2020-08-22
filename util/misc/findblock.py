#!/usr/bin/env python3
import sys
import argparse

parser = argparse.ArgumentParser(description='Search spills.dat (from runspec-wrapper) to find the benchmark for a block')
parser.add_argument('spills', help='The spills.dat file to search in. - for stdin')
parser.add_argument('blocks', help='The blocks to search for. This may include the `:##` part, or it may just be the mangled function name', nargs='*')

result = parser.parse_args()

with open(result.spills, 'r') as f:
    file = f.read()

fns = (block.split(':')[0] for block in result.blocks)

fn_locs = [file.find(fn) for fn in fns]
fn_benchmarks = [file.rfind(':', 0, fnindex) for fnindex in fn_locs]
fn_benchmark_spans = [(file.rfind('\n', 0, e), e) for e in fn_benchmarks]
fn_benchmarks = [file[b + 1:e] for (b, e) in fn_benchmark_spans]

print('\n'.join(fn_benchmarks))
