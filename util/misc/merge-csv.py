#!/usr/bin/env python3
import argparse
import csv
import sys


def main(infile, outfile):
    metrics = {}
    metric_names = []

    for metric, total, bench in csv.reader(infile):
        assert total == bench or total == 'Total'
        if metric not in metrics:
            metric_names.append(metric)

        metrics.setdefault(metric, []).append(bench)

    writer = csv.writer(outfile)
    for metric in metric_names:
        try:
            writer.writerow([metric, sum(int(x) for x in metrics[metric]), *metrics[metric]])
        except ValueError:
            writer.writerow([metric, 'Total', *metrics[metric]])


if __name__ == '__main__':
    main(sys.stdin, sys.stdout)
