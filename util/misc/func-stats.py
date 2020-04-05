#!/bin/python3
# Find the number of functions that are compiled more than once by LLVM.

import sys
import re
import json

def get_events_of_id(logs, event_id):
    event_start = 'EVENT: {"event_id": "{}"'.format(event_id)
    lines = logs.splitlines()
    event_lines = [line.split(' ', 1)[1] for line in lines if line.startswith(event_start)]
    return list(map(json.loads, event_lines))

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
            matchBlock = get_events_of_id(line)

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
