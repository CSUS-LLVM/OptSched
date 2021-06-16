#/usr/bin/python3
# TODO
# 1: Add options praser.
# 2: Make printing all mismatched dags optional and disabled by default.
# 3: Add option to print out x number of blocks with largest mismatches.
# 4: Add option to print out x number of mismatches with smallest number of instructions.

import os, sys
import itertools
from typing import List
import argparse

import analyze
from analyze import Logs

# Explain this many of the blocks missing a lower bound
MISSING_LOWER_BOUND_DUMP_COUNT = 3
MISSING_LOWER_BOUND_DUMP_LINES = 10

def dags_info(logs: Logs):
    dags = {}

    blocks = list(logs)

    no_lb = [block for block in blocks if 'CostLowerBound' not in block]

    if no_lb:
        print('WARNING: Missing a logged lower bound for {missing}/{total} blocks.'
            .format(missing=len(no_lb), total=len(blocks)), file=sys.stderr)

        trimmed = ('\n'.join(block.raw_log.splitlines()[:MISSING_LOWER_BOUND_DUMP_LINES]) for block in no_lb)

        for i, block in enumerate(itertools.islice(trimmed, MISSING_LOWER_BOUND_DUMP_COUNT)):
            print('WARNING: block {} missing lower-bound:\n{}\n...'.format(i, block),
                  file=sys.stderr)

    for block in blocks:
        lowerBound = block['CostLowerBound'][-1]['cost']
        blockInfo = block.single('BestResult')
        dagName = blockInfo['name']
        dags[dagName] = {
            'lowerBound': lowerBound,
            'cost': blockInfo['cost'] + lowerBound,
            'relativeCost': blockInfo['cost'],
            'length': blockInfo['length'],
            'isOptimal': blockInfo['optimal']
        }

    return dags


if __name__ == "__main__":
    dags1 = {}
    dags2 = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('first')
    parser.add_argument('second')
    args = analyze.parse_args(parser, 'first', 'second')

    dags1 = dags_info(args.first)
    dags2 = dags_info(args.second)

    numDagsLog1 = len(dags1)
    numDagsLog2 = len(dags2)
    # The number of blocks that are optimal in both logs.
    optimalInBoth = 0
    # The number of blocks that are only optimal in log 1.
    optimalLog1 = 0
    # The number of blocks that are only optimal in log 2.
    optimalLog2 = 0
    # Mismatches where blocks are optimal in both logs but have different costs.
    misNonEqual = 0
    # Mismatches where block is optimal in log 1 but it has a higher cost than the non-optimal block in log 2.
    misBlk1Opt = 0
    # Mismatches where block is optimal in log 2 but it has a higher cost than the non-optimal block in log 1.
    misBlk2Opt = 0
    # The quantity of blocks with the largest mismatches to print.
    numLarMisPrt = 10
    # The quantity of mismatched blocks with the shortest length to print.
    numSmlBlkPrt = 50
    # Dictionary with the sizes of the mismatches for each mismatched block and the size of the block.
    mismatches = {}



    if numDagsLog1 != numDagsLog2:
        print('Error: Different number of dags in each log file.')

    for dagName in dags1:
        if dagName not in dags2:
            print('Error: Could not find ' + dagName + ' in the second log file.')
            continue

        dag1 = dags1[dagName]
        dag2 = dags2[dagName]
        if dag1['isOptimal'] and dag2['isOptimal']:
            optimalInBoth+=1
            if dag1['cost'] != dag2['cost']:
                # There was a mismatch where blocks are optimal in both logs but have different costs
                misNonEqual += 1
                mismatches[dagName] = {}
                mismatches[dagName]['length'] = dag1['length']
                mismatches[dagName]['misSize'] = abs(dag1['cost'] - dag2['cost'])
                #print('Mismatch for dag ' + dagName + ' (Both optimal with non-equal cost)')

        elif dag1['isOptimal']:
            optimalLog1+=1
            if dag1['cost'] > dag2['cost']:
                # There was a mismatch where block is optimal in log 1 but it has a higher cost than the non-optimal block in log 2
                misBlk1Opt += 1
                mismatches[dagName] = {}
                mismatches[dagName]['length'] = dag1['length']
                mismatches[dagName]['misSize'] = dag1['cost'] - dag2['cost']
                #print('Mismatch for dag ' + dagName + ' (Only optimal in log 1 but has higher cost than the non-optimal block in log 2)')

        elif dag2['isOptimal']:
            optimalLog2+=1
            if dag2['cost'] > dag1['cost']:
                # There was a mismatch where block is optimal in log 2 but it has a higher cost than the non-optimal block in log 1
                misBlk2Opt += 1
                mismatches[dagName] = {}
                mismatches[dagName]['length'] = dag1['length']
                mismatches[dagName]['misSize'] = dag2['cost'] - dag1['cost']
                #print('Mismatch for dag ' + dagName + ' (Only optimal in log 2 but has higher cost than the non-optimal block in log 1)')

    print('Optimal Block Stats')
    print('-----------------------------------------------------------')
    print('Blocks in log file 1: ' + str(numDagsLog1))
    print('Blocks in log file 2: ' + str(numDagsLog2))
    print('Blocks that are optimal in both files: ' + str(optimalInBoth))
    print('Blocks that are optimal in log 1 but not in log 2: ' + str(optimalLog1))
    print('Blocks that are optimal in log 2 but not in log 1: ' + str(optimalLog2))
    print('----------------------------------------------------------\n')

    print('Mismatch stats')
    print('-----------------------------------------------------------')
    print('Mismatches where blocks are optimal in both logs but have different costs: ' + str(misNonEqual))
    print('Mismatches where the block is optimal in log 1 but it has a higher cost than the non-optimal block in log 2: ' + str(misBlk1Opt))
    print('Mismatches where the block is optimal in log 2 but it has a higher cost than the non-optimal block in log 1: ' + str(misBlk2Opt))
    print('Total mismatches: ' + str(misNonEqual + misBlk1Opt + misBlk2Opt))
    print('-----------------------------------------------------------\n')

    print('The ' + str(numLarMisPrt) + ' mismatched blocks with the largest difference in cost')
    print('-----------------------------------------------------------')
    sortedMaxMis =  sorted(mismatches.items(), key=lambda i: (mismatches[i[0]]['misSize'], i[0]), reverse=True)
    i = 1
    for block in sortedMaxMis[:numLarMisPrt]:
        print(str(i) + ':')
        print('Block Name: ' + block[0] + '\nLength: ' + str(block[1]['length']) + '\nDifference in cost: ' + str(block[1]['misSize']))
        i += 1
    print('-----------------------------------------------------------\n')

    print('The smallest ' + str(numSmlBlkPrt) + ' mismatched blocks')
    print('-----------------------------------------------------------')
    sortedMisSize =  sorted(mismatches.items(), key=lambda i: (mismatches[i[0]]['length'], i[0]))
    i = 1
    for block in sortedMisSize[:numSmlBlkPrt]:
        print(str(i) + ':')
        print('Block Name: ' + block[0] + '\nLength: ' + str(block[1]['length']) + '\nDifference in cost: ' + str(block[1]['misSize']))
        i += 1
    print('-----------------------------------------------------------')
