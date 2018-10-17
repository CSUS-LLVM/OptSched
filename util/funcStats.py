#!/bin/python3
# Find the number of functions that are compiled more than once by LLVM.

import sys
import re

RE_NEW_BENCH = re.compile(r'(\d+)\.(.*) base \.exe default')
RE_BLOCK = re.compile(r'INFO: Processing DAG (.*) with (\d+) insts')

if __name__ == "__main__":
    with open(sys.argv[1]) as logfile:
        blocks = {}
        bench = None
        totalRepeats = 0
        totalMismatches = 0
        for line in logfile.readlines():
            matchBench = RE_NEW_BENCH.findall(line)
            matchBlock = RE_BLOCK.findall(line)

            if matchBench != []:
                if bench:
                    print('In bench ' + bench + '  found ' + str(totalRepeats) + ' repeat blocks and ' + str(totalMismatches) + ' mismatches in length.')
                funcs = {}
                totalRepeats = 0
                totalMismatches = 0
                bench = matchBench[0][1]

            elif matchBlock != []:
                name = matchBlock[0][0]
                insts = matchBlock[0][1]

                if name in blocks:
                    if blocks[name] != insts:
                        totalMismatches += 1

                    totalRepeats += 1
                    continue
                else:
                    blocks[name] = insts
