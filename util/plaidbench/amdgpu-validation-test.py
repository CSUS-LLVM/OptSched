#/usr/bin/python3
'''*******************************************************************************
Description:    Validation script for OptSched with the plaidbench benchmarking
                suite. This script is meant to be used with the run-plaidbench.sh
                script.
Modified by:    Vang Thao  
Last Update:    December 17, 2019
*******************************************************************************'''

'''
HOW TO USE:
    1.) Run two plaidbench benchmarks with run-plaidbench.sh to generate
        two directories containing the results for each run.
    2.) Enter in the path to those directories as arguments to this script
'''

import sys
import re
import os
import argparse

# List of benchmark names
benchmarks = [
    'densenet121',
    'densenet169',
    'densenet201',
    'inception_resnet_v2',
    'inception_v3',
    'mobilenet',
    'nasnet_large',
    'nasnet_mobile',
    'resnet50',
    'vgg16',
    'vgg19',
    'xception',
    'imdb_lstm',
]

# Parse for DAG stats
RE_DAG_COST = re.compile(r"INFO: Best schedule for DAG (.*) has cost (\d+) and length (\d+). The schedule is (.*) \(Time")
# Parse for passthrough number
RE_PASS_NUM = re.compile(r"End of (.*) pass through")

# Store DAGs stats for each benchmark and passes
dags = []
# Data structure
# Run number
    # Benchmark Name
        # Passthrough Number
            # DAG name
                # DAG stats
# dags[Run number][Benchmark name][Passthrough number][Dag name][DAG stats]

# Store number of DAGs
numDags = []

# Comparison function, takes in a pass number
# Valid arguments: 'first' or 'second'
def compareDags(displayMismatches, displayNumLargest, displayNumSmallest, passNum):
    if numDags[0][passNum] != numDags[1][passNum]:
        print('Error: Different number of dags in each log file for pass {}.'.format(passNum))

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
    numLarMisPrt = displayNumLargest
    # The quantity of mismatched blocks with the shortest length to print.
    numSmlBlkPrt = displayNumSmallest
    # Dictionary with the sizes of the mismatches for each mismatched block and the size of the block.
    mismatches = {}
    
    for bench in benchmarks:
        for dagName in dags[0][bench][passNum]:
            if dagName not in dags[1][bench][passNum]:
                print('Error: Could not find DAG {} in benchmark {} in the second log file.'.format(\
                    dagName, bench))
                continue
            dag1 = dags[0][bench][passNum][dagName]
            dag2 = dags[1][bench][passNum][dagName]
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
    print('Optimal Block Stats for {} pass'.format(passNum))
    print('-----------------------------------------------------------')
    print('Blocks in log file 1: {}'.format(numDags[0][passNum]))
    print('Blocks in log file 2: {}'.format(numDags[1][passNum]))
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

    if displayMismatches:
        if numLarMisPrt != 0:
            print('The ' + str(numLarMisPrt) + ' mismatched blocks with the largest difference in cost')
            print('-----------------------------------------------------------')
            sortedMaxMis =  sorted(mismatches.items(), key=lambda i: (mismatches[i[0]]['misSize'], i[0]), reverse=True)
            i = 1
            for block in sortedMaxMis[:numLarMisPrt]:
                print(str(i) + ':')
                print('Block Name: ' + block[0] + '\nLength: ' + str(block[1]['length']) + '\nDifference in cost: ' + str(block[1]['misSize']))
                i += 1
            print('-----------------------------------------------------------\n')

        if numSmlBlkPrt != 0:
            print('The smallest ' + str(numSmlBlkPrt) + ' mismatched blocks')
            print('-----------------------------------------------------------')
            sortedMisSize =  sorted(mismatches.items(), key=lambda i: (mismatches[i[0]]['length'], i[0]))
            i = 1
            for block in sortedMisSize[:numSmlBlkPrt]:
                print(str(i) + ':')
                print('Block Name: ' + block[0] + '\nLength: ' + str(block[1]['length']) + '\nDifference in cost: ' + str(block[1]['misSize']))
                i += 1
            print('-----------------------------------------------------------')



def main(args):
    directories = [args.directory1, args.directory2]
    # Collect DAG info from the two run results
    for i in range(0,2):
        # Temp variable to hold the number of DAGs
        tempNumDags = {}
        tempNumDags['first'] = 0
        tempNumDags['second'] = 0

        # Temp variable to hold the benchmark's stats
        benchStats = {}
        for bench in benchmarks:
            # Declare containers for this benchmark
            benchStats[bench] = {}
            benchStats[bench]['first'] = {}
            benchStats[bench]['second'] = {}
            
            # Open log file
            currentPath = os.path.join(directories[i], bench)
            currentLogFile = os.path.join(currentPath, bench + '.log')
            with open(currentLogFile) as logfile:
                # Read and split scheduling regions
                log = logfile.read()
                blocks = log.split('********** Opt Scheduling **********')[1:]
                for block in blocks:
                    # Get pass num
                    getPass = RE_PASS_NUM.search(block)
                    passNum = getPass.group(1)
                    
                    # Get DAG stats
                    dagStats = RE_DAG_COST.search(block)
                    dag = {}
                    dagName = dagStats.group(1)
                    dag['dagName'] = dagName
                    dag['cost'] = int(dagStats.group(2))
                    dag['length'] = dagStats.group(3)
                    dag['isOptimal'] = (dagStats.group(4) == 'optimal')
                    
                    # Add this DAG's stats to temp stats container 
                    benchStats[bench][passNum][dagName] = dag
                    # Record number of DAGs
                    tempNumDags[passNum] += 1

        # Move temp stats to global vars
        dags.append(benchStats)
        numDags.append(tempNumDags)
        
    # Compare DAGs from first passthrough
    compareDags(args.displayMismatches, args.displayNumLargest, args.displayNumSmallest, 'first')

    # Compare DAGs from second passthrough
    compareDags(args.displayMismatches, args.displayNumLargest, args.displayNumSmallest, 'second')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation script for OptSched on plaidbench', \
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run 1 directory for plaidbench    
    parser.add_argument('directory1',
                        help='Directory containing a plaidbench run')
    # Run 2 directory for plaidbench
    parser.add_argument('directory2',
                        help='Directory containing a plaidbench run')

    # Option to display mismatches, defaults to off
    parser.add_argument('--mismatches', '-m', action='store_true',
                        default=False,
                        dest='displayMismatches',
                        help='Display mismatches')

    parser.add_argument('--largest', '-l', type=int,
                        default=10,
                        dest='displayNumLargest',
                        help='Print out x number of blocks with largest mismatches. Requires display mismatches.')

    parser.add_argument('--smallest', '-s', type=int,
                        default=50,
                        dest='displayNumSmallest',
                        help='Print out x number of mismatches with smallest number of instructions. Requires display mismatches.')

    args = parser.parse_args()

    main(args)

