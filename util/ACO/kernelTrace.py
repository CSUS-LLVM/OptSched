#!/usr/bin/python3
import subprocess
import re
import argparse
import json
import csv
import os

def load_tests(test_list : []):
    sets = {}
    for test in test_list:
        bench_set, bench_name = test
        if not bench_set in sets:
            sets[bench_set] = []
        sets[bench_set].append(bench_name)
    return sets

def parse_tests_to_run(test_path : str):
    test_list = []
    with open(test_path, "r") as file:
        for line in file:
            cols = line.split("\t")
            bench_set = cols[0]
            bench_name = cols[1].strip()
            test_list.append((bench_set, bench_name))
    return test_list

def run_test_indiv(bench_path : str,
                   benchmarks : dict
):
    kernels = {}
    d = dict(os.environ)
    d["AMD_LOG_LEVEL"] = "3"
    for bench_set in benchmarks.keys():
        for ubench in benchmarks[bench_set]:
            kernels[ubench] = []
            try:
                command = bench_path + bench_set
                regex = re.escape(ubench)
                proc = subprocess.run([
                     command,
                    f"--benchmark_repetitions=1"
                     "--benchmark_format=json",
                     "--benchmark_min_time=0",
                    f"--benchmark_filter={regex}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=d,
                    text=True)
                launched = [x.split()[-1] for x in proc.stderr.split("\n") if "ShaderName" in x]
                kernels[ubench].extend(list(set(launched)))
            except KeyboardInterrupt:
                return kernels
    return kernels

def write_json(out_path : str, results : dict):
    with open(out_path, "w") as file:
        file.write(json.dumps(results, indent=4))

def main(args):
    test_list = parse_tests_to_run(args.test_path)
    benchmarks = load_tests(test_list)
    results = run_test_indiv(args.bench_path, benchmarks)
    write_json(args.out_path, results)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(
                    description='Script to run rocPRIM tests.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument('--benchpath', '-b',
                            dest='bench_path',
                            type=str,
                            default=None,
                            required=True,
                            help='Specify the filepath of' 
                                 'rocPRIM/build/benchmark.'
        )

        parser.add_argument('--testpath', '-t',
                            dest='test_path',
                            type=str,
                            default=None,
                            required=True,
                            help='Specify the filepath of TestsToRun.txt'
        )

        parser.add_argument('--output', '-o',
                            dest='out_path',
                            type=str,
                            default=None,
                            required=True,
                            help='Path to write output csv file.'
        )

        args = parser.parse_args()
        main(args)
