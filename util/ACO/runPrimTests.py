#!/usr/bin/python3
import subprocess
import re
import argparse
import json
import csv
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

def parse_tests_to_run(test_path : str):
    test_list = []
    with open(test_path, "r") as file:
        for line in file:
            cols = line.split("\t")
            bench_set = cols[0]
            bench_name = cols[1].strip()
            test_list.append((bench_set, bench_name))
    return test_list

def run_test_sets(bench_path : str,
                  benchmarks : [] 
):
    results = {}
    bench_runs = iter(tqdm(benchmarks, leave=False, ascii=True))
    for run in bench_runs:
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
                results[run.bench].append(res["benchmarks"])
        except KeyboardInterrupt:
            tqdm.write("Benchmarking run aborted by user.")
            tqdm.write("The following benchmarks were not completed:")
            incomplete_runs = list(set(bench_runs))
            for run in incomplete_runs:
                tqdm.write(f"{run.set}\t{run.bench}")
            break
    return results

def write_csv(out_path : str, results : dict):
    with open(out_path, "w", newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(["bench_set","name","throughput"])
        for bench_set in results.keys():   
            throughputs = [float(bench["bytes_per_second"]) * float("1e-9")
                           for bench in results[bench_set]
            for bench in results[bench_set]:
                name = bench["name"]
                if median_only:
                    if not "manual_time_median" in name: continue
                if "manual_time" in name:
                    throughput = float(bench["bytes_per_second"]) * float("1e-9")
                    writer.writerow([bench_set, name, throughput])

def main(args):
    test_list = parse_tests_to_run(args.test_path)
    benchmarks = load_tests(test_list)
    results = run_test_indiv(args.bench_path, benchmarks, args.repetitions)
    write_csv(args.out_path, results, args.repetitions > 1)

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

        parser.add_argument('--shuffle', '-i',
                            dest='shuffle',
                            type=bool,
                            default=False,
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
