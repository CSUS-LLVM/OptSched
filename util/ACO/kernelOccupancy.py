#!/usr/bin/python3
import subprocess
import re
import argparse
import json
import csv
import os

def write_csv(out_path : str, results : dict):
    with open(out_path, "w", newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerow(["kernel","occupancy"])
        for kernel in results.keys():
            occupancy = results[kernel]
            writer.writerow([kernel, occupancy])

def main(args):
    kernels = {}
    with open(args.log_path, "r") as file:
        for line in file:
            if "Final occupancy" in line:
                name, occ = line.split()[-1].split(":")
                kernels[name] = int(occ)
    write_csv(args.out_path, kernels)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(
                    description='Extract per-function occupancy data'
                                'from OptSched build logs.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument('--log', '-l',
                            dest='log_path',
                            type=str,
                            default=None,
                            required=True,
                            help='Specify the filepath of' 
                                 'OptSched build log.'
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
