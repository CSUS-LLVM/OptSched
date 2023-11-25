#!/usr/bin/python3
import subprocess
import re
import argparse
import json
import csv
import random
from tqdm import tqdm
from collections import namedtuple
from statistics import median

Run = namedtuple("Run", ["set", "bench"])

def load_tests(test_list : [],
               repetitions : int,
               shuffle : bool
):
    runs = []
    for test in test_list:
        for i in range(repetitions):
            runs.append(Run(test[0], test[1]))
    if (shuffle): random.shuffle(runs)
    return runs

def parse_tests_to_run(test_path : str):
    test_list = []
    with open(test_path, "r") as file:
        for line in file:
            cols = line.split("\t")
            bench_set = cols[0]
            bench_name = cols[1].strip()
            test_list.append((bench_set, bench_name))
    return test_list

def run_tests(bench_path : str,
                  benchmarks : [] 
):
    results = {}
    runs = iter(tqdm(benchmarks, leave=False, ascii=True))
    incomplete_runs = set()
    for run in runs:
        try:
            tqdm.write(f"Running {run.bench}...")
            regex = re.escape(run.bench)
            command = bench_path + run.set
            run_data = subprocess.run([
                command,
                "--benchmark_format=json",
                "--benchmark_min_time={}".format("1"),
                "--benchmark_filter={}".format(regex)],
                stdout=subprocess.PIPE)
            res = json.loads(run_data.stdout.decode())
            if "benchmarks" in res:
                if not run.bench in results:
                    results[run.bench] = []
                results[run.bench].append(res["benchmarks"][0]["bytes_per_second"])
        except KeyboardInterrupt:
            tqdm.write("Benchmarking run aborted by user.")
            tqdm.write("The following benchmarks were not completed:")
            incomplete_runs = list(set(runs))
            for run in incomplete_runs:
                tqdm.write(f"{run.set}\t{run.bench}")
            break
    return [bench for bench in benchmarks 
            if not bench in incomplete_runs], results

def write_csv(out_path : str, benchmarks : [], results : dict):
    with open(out_path, "w", newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(["bench_set", "name", "throughput"])
        benchmarks = set(benchmarks)
        for run in benchmarks:
            mid = median(results[run.bench])
            writer.writerow([run.set, run.bench, float(mid) * float("1e-9")]) 
    return

def main(args):
    test_list = parse_tests_to_run(args.test_path)
    benchmarks = load_tests(test_list, args.repetitions, args.shuffle)
    completed_runs, results = run_tests(args.bench_path, benchmarks)
    write_csv(args.out_path, completed_runs, results)

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

        parser.add_argument('--repetitions', '-r', 
                            dest='repetitions',
                            type=int,
                            default=1,
                            required=False,
                            help='Number of repeated trials per benchmark'
        )

        parser.add_argument('--shuffle', 
                            dest='shuffle',
                            action='store_true',
                            help='Enable random interleaving ' 
                                 'of benchmark repetitions'
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
