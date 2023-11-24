#!/usr/bin/python3
from argparse import ArgumentParser
from statistics import median
import csv

def main(args):
    headers = []
    sheets = []
    merged_results = []
    for csv_file_path in args.csv_files:
        with open(csv_file_path, newline='') as csv_file:
            sheet = []
            reader = csv.reader(csv_file, dialect='excel')
            rows = [row for row in reader]
            headers = rows[0]
            data = rows[1:]
            sheets.append(data)
    merged_results.append(headers)
    for item in zip(*sheets):
        throughputs = []
        for datum in item:
            throughputs.append(datum[2])
        res = [item[0][0], item[0][1], median(throughputs)]
        merged_results.append(res)
    with open("median_results.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, dialect="excel")
        for row in merged_results:
            writer.writerow(row)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    args = parser.parse_args()
    main(args)
