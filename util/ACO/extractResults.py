#!/usr/bin/python3

import json
import csv
from pathlib import Path
import argparse

def main(args):
  outfile = args.output

  resultsList = {}

  wd = Path()
  for child in wd.iterdir():
      if not child.is_dir() and ".json" in child.name:
        with open(child) as file:
          results = json.loads(file.read())
          set_name = "".join(results["context"]["executable"].split("/")[-1])
          if "benchmarks" in results:
            if not set_name in resultsList:
              resultsList[set_name] = []
            resultsList[set_name].extend(results["benchmarks"])

  with open(outfile, 'w', newline='') as file:
      writer = csv.writer(file, dialect='excel')
      writer.writerow(["set", "name", "throughput"])
      for bench_set in resultsList.keys():
        for bench in resultsList[bench_set]:
          if "name" in bench and "bytes_per_second" in bench and "manual_time" in bench["name"]:
            name = bench["name"]
            throughput = float(bench["bytes_per_second"]) * float("1e-9")
            writer.writerow([bench_set, name, throughput])

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--output', '-o', dest='output', help='output .csv file path')
  args = parser.parse_args()
  main(args)
