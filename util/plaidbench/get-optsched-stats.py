#!/usr/bin/python3
'''
**********************************************************************************
Description:    This script is meant to be used with the OptSched scheduler and
                the run-plaidbench.sh script or with test results from shoc.
                This script will extract stats about how our OptSched scheduler
                is doing from the log files generated plaidml or shoc.
Author:	        Vang Thao
Last Update:	September 2020
**********************************************************************************

OUTPUT:
    This script takes in data from plaidbench or shoc runs and output a single 
    spreadsheet.
        Spreadsheet 1: optsched-stats.csv

Requirements:
    - python3

HOW TO USE:
    1.) Run a plaidbench benchmarks with run-plaidbench.sh to generate a
        directory containing the results for the run.
    2.) Pass the path to the directory as an input to this script

Example:
    ./get-optsched-stats.py /home/tom/plaidbench-optsched-01/

    where plaidbench-optsched-01/ contains
        densenet121
        densenet169
        ...
        ...
'''

import argparse
import csv
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readlogs import *


# List of stats that can be initialized to 0
statsProcessed = [
    'TotalProcessed',
    'SchedRevert',
    'EnumCnt',
    'OptImpr',
    'OptNotImpr',
    'TimeoutImpr',
    'TimeoutNotImpr',
    'TimeoutCnt',
    'TotalInstr',
    'TimeoutInstrToEnum',
]

def getNewStatsDict():
    '''
    Return a dict and initialize basic stats used by this script
    '''
    stats = {}
    for stat in statsProcessed:
        stats[stat] = 0
    stats['AverageSizeToEnum'] = -1.0
    stats['LargestOptimalRegion'] = -1
    stats['LargestImprovedRegion'] = -1

    return stats


def parseStats(filePaths):
    # Get logger
    logger = logging.getLogger('parseStats')

    # Initialize pass stats collection variables
    stats = {}

    # Begin stats collection for this run
    for bench in filePaths:
        logger.debug('Verifying file {} exists'.format(filePaths[bench]))
        # first check if log file exists.
        if os.path.exists(filePaths[bench]):
            # Open log file if it exists.
            with open(filePaths[bench]) as file:
                logger.debug(
                    'File found! Processing file {}'.format(filePaths[bench]))
                
                # Contain the stats for this benchmark
                curStats = {}

                log = file.read()
                blocks = split_blocks(log)
                for block in blocks:
                    events = keep_only_first_event(parse_events(block))
                    # Get pass num, if none is found then
                    # use third as default.
                    if 'PassFinished' in events.keys():
                        passNum = events['PassFinished']['num']
                        if passNum not in curStats.keys():
                            curStats[passNum] = getNewStatsDict()
                    else:
                        passNum = 0
                        if passNum not in curStats.keys():
                            curStats[passNum] = getNewStatsDict()

                    curStats[passNum]['TotalProcessed'] += 1

                    if OPT_RE_REVERT_SCHED.search(block):
                        curStats[passNum]['SchedRevert'] += 1
                        continue

                    # If our enumerator was called then
                    # record stats for it.
                    if 'Enumerating' in events.keys():
                        curStats[passNum]['EnumCnt'] += 1
                        numOfInstr = events['ProcessDag']['num_instructions']
                        curStats[passNum]['TotalInstr'] += numOfInstr

                        if 'DagTimedOut' in events.keys():
                            cost = events['DagTimedOut']['cost_improvement']
                            # Timeout and improved
                            if cost > 0:
                                curStats[passNum]['TimeoutImpr'] += 1
                                if numOfInstr > curStats[passNum]['LargestImprovedRegion']:
                                    curStats[passNum]['LargestImprovedRegion'] = numOfInstr
                            # Timeout but not improved
                            elif cost == 0:
                                curStats[passNum]['TimeoutNotImpr'] += 1
                            # Negative Cost! Raise error
                            else:
                                raise AssertionError("Found negative cost for the block:\n" + block)
                            curStats[passNum]['TimeoutCnt'] += 1
                            curStats[passNum]['TimeoutInstrToEnum'] += numOfInstr
                        elif 'DagSolvedOptimally' in events.keys():
                            cost = events['DagSolvedOptimally']['cost_improvement']
                            # Optimal and improved
                            if cost > 0:
                                curStats[passNum]['OptImpr'] += 1
                                if numOfInstr > curStats[passNum]['LargestImprovedRegion']:
                                    curStats[passNum]['LargestImprovedRegion'] = numOfInstr
                            # Optimal but not improved
                            elif cost == 0:
                                curStats[passNum]['OptNotImpr'] += 1
                            # Negative Cost! Raise error
                            else:
                                raise AssertionError("Found negative cost for the block:\n" + block)
                            if numOfInstr > curStats[passNum]['LargestOptimalRegion']:
                                curStats[passNum]['LargestOptimalRegion'] = numOfInstr
                        else:
                            raise AssertionError("Couldn't find improvement cost for the block:\n" + block)

                for passNum in curStats:
                    if curStats[passNum]['EnumCnt'] != 0:
                        curStats[passNum]['AverageSizeToEnum'] = curStats[passNum]['TotalInstr'] / \
                            curStats[passNum]['EnumCnt']

                stats[bench] = curStats

        # If the file doesn't exist, output error log.
        else:
            print('Cannot find log file for benchmark {}.'.format(bench))

    return stats


def printStats(stats):
    for bench in stats:
        print('Printing stats for {}'.format(bench))
        for passNum in stats[bench]:            
            print('  Pass No. {}'.format(passNum))
            for stat in stats[bench][passNum]:
                print('    {}: {}'.format(stat, stats[bench][passNum][stat]))


def separatePasses(stats):
    '''
    Change mapping of stats from 'dict[bench --> passNum]' to 'dict[passNum --> bench]'
    '''
    passStats = {}
    for bench in stats:
        for passNum in stats[bench]:
            if passNum not in passStats.keys():
                passStats[passNum] = {}
            passStats[passNum][bench] = stats[bench][passNum]
    
    return passStats

def createSpreadsheets(stats, output):
    if 'csv' not in output[-3:]:
        output += '.csv'

    with open(output, 'w', newline='') as file:
        # Column header for csv file
        fieldnames = [
            'Benchmark',
            'Regions processed',
            'Sched. Reverted',
            'Passed to B&B',
            'Optimal and improved',
            'Optimal and not improved',
            'Timed out and improved',
            'Timed out and not improved',
            'Avg. Region size passed to B&B',
            'Largest optimal region',
            'Largest improved region'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        for passNum in stats:
            totalProcessed = 0
            totalReverted = 0
            totalEnumCnt = 0
            totalOptImpr = 0
            totalOptNotImpr = 0
            totalTimedoutImpr = 0
            totalTimedOutNotImpr = 0
            totalLargestOptRegion = -1
            totalLargestImprRegion = -1
            totalInstrToEnum = 0

            writer.writerow({'Benchmark': 'Pass No. {}'.format(passNum)})
            writer.writeheader()
            for bench in stats[passNum]:
                # Format output with format: # (#.##%)
                passedToEnum = getPercentageString(
                    stats[passNum][bench]['EnumCnt'], stats[passNum][bench]['TotalProcessed'])
                optImprv = getPercentageString(
                    stats[passNum][bench]['OptImpr'], stats[passNum][bench]['EnumCnt'])
                optNotImprv = getPercentageString(
                    stats[passNum][bench]['OptNotImpr'], stats[passNum][bench]['EnumCnt'])
                timedoutImpr = getPercentageString(
                    stats[passNum][bench]['TimeoutImpr'], stats[passNum][bench]['EnumCnt'])
                timedoutNotImpr = getPercentageString(
                    stats[passNum][bench]['TimeoutNotImpr'], stats[passNum][bench]['EnumCnt'])

                # Write the current bench's final formatted string to csv file
                writer.writerow({
                    'Benchmark': bench,
                    'Regions processed': stats[passNum][bench]['TotalProcessed'],
                    'Sched. Reverted': stats[passNum][bench]['SchedRevert'],
                    'Passed to B&B': passedToEnum,
                    'Optimal and improved': optImprv,
                    'Optimal and not improved': optNotImprv,
                    'Timed out and improved': timedoutImpr,
                    'Timed out and not improved': timedoutNotImpr,
                    'Avg. Region size passed to B&B': stats[passNum][bench]['AverageSizeToEnum'],
                    'Largest optimal region': stats[passNum][bench]['LargestOptimalRegion'],
                    'Largest improved region': stats[passNum][bench]['LargestImprovedRegion']
                })
                
                # Get stats for the overall run
                totalProcessed += stats[passNum][bench]['TotalProcessed']
                totalReverted += stats[passNum][bench]['SchedRevert']
                totalEnumCnt += stats[passNum][bench]['EnumCnt']
                totalOptImpr += stats[passNum][bench]['OptImpr']
                totalOptNotImpr += stats[passNum][bench]['OptNotImpr']
                totalTimedoutImpr += stats[passNum][bench]['TimeoutImpr']
                totalTimedOutNotImpr += stats[passNum][bench]['TimeoutNotImpr']
                if totalLargestOptRegion < stats[passNum][bench]['LargestOptimalRegion']:
                    totalLargestOptRegion = stats[passNum][bench]['LargestOptimalRegion']
                if totalLargestImprRegion < stats[passNum][bench]['LargestImprovedRegion']:
                    totalLargestImprRegion = stats[passNum][bench]['LargestImprovedRegion']
                totalInstrToEnum += stats[passNum][bench]['TotalInstr']

            passedToEnum_ = getPercentageString(totalEnumCnt, totalProcessed)
            optImprv_ = getPercentageString(totalOptImpr, totalEnumCnt)
            optNotImprv_ = getPercentageString(totalOptNotImpr, totalEnumCnt)
            timedoutImpr_ = getPercentageString(totalTimedoutImpr, totalEnumCnt)
            timedoutNotImpr_ = getPercentageString(
                totalTimedOutNotImpr, totalEnumCnt)
            totalCurAvgToEnum = -1
            if totalEnumCnt != 0:
                totalCurAvgToEnum = totalInstrToEnum / totalEnumCnt
            writer.writerow({
                    'Benchmark': 'Overall',
                    'Regions processed': totalProcessed,
                    'Sched. Reverted': totalReverted,
                    'Passed to B&B': passedToEnum_,
                    'Optimal and improved': optImprv_,
                    'Optimal and not improved': optNotImprv_,
                    'Timed out and improved': timedoutImpr_,
                    'Timed out and not improved': timedoutNotImpr_,
                    'Avg. Region size passed to B&B': totalCurAvgToEnum,
                    'Largest optimal region': totalLargestOptRegion,
                    'Largest improved region': totalLargestImprRegion
            })

            writer.writerow({'Benchmark': ''})


def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    filePaths = get_bench_log_paths(args.inputFolder, args.benchmark)

    # Start stats collection
    stats = parseStats(filePaths)

    if args.verbose:
        printStats(stats)

    finalStats = separatePasses(stats)

    if not args.disable:
        filename = ''
        if args.output is None:
            filename = os.path.dirname('optsched-stats-' + args.inputFolder)
        else:
            filename = args.output

        createSpreadsheets(finalStats, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to extract OptSched stats',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(dest='inputFolder',
                        help='The path to a benchmark directory')

    parser.add_argument('--verbose', '-v',
                        action='store_true', default=False,
                        dest='verbose',
                        help='Print the stats to terminal')

    parser.add_argument('--output', '-o',
                        dest='output',
                        help='Output spreadsheet filepath')

    parser.add_argument('--disable', '-d',
                        action='store_true', default=False,
                        dest='disable',
                        help='Disable spreadsheet output.')

    parser.add_argument('--benchmark', '-b',
                        default='plaid',
                        choices=['plaid', 'shoc'],
                        dest='benchmark',
                        help='Select the benchmarking suite to parse for.')

    args = parser.parse_args()

    main(args)
