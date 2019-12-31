'''
**********************************************************************************
Description:  Extract schedule length stats from plaidbench runs.
Author:       Vang Thao
Created:      December 30, 2019
Last Update:  December 30, 2019
**********************************************************************************

OUTPUT:
    This script takes in data from plaidbench runs and output a spreadsheet
    containing the average schedule length for each benchmark and the overall
    average schedule length.
        Spreadsheet 1: schedule-length.xlsx

Requirements:
    - python3
    - pip3
    - openpyxl (sreadsheet module, installed using pip3)

HOW TO USE:
    1.) Run a plaidbench benchmarks with run-plaidbench.sh to generate a
        directory containing the results for the run.
    2.) Move the directory into a separate folder containing only the
        directories generated by the script.
    3.) Pass the path to the folder as an input to this script with
        the -i option.

Example:
    ./get-sched-length.py -i /home/tom/plaidbench-runs
    
    where plaidbench-runs/ contains
        plaidbench-optsched-01/
        plaidbench-optsched-02/
        ...
        plaidbench-amd-01/
        ...
'''
#!/usr/bin/python3

import os
import re
import argparse
from openpyxl import Workbook
from openpyxl.styles import Font

# For AMD 
RE_DAG_NAME = re.compile('Processing DAG (.*) with')
RE_SCHED_LENGTH = re.compile('The list schedule is of length (\d+) and')

# For OptSched
RE_PASS_NUM = re.compile(r'End of (.*) pass through')
RE_DAG_INFO = re.compile(r'INFO: Best schedule for DAG (.*) has cost (\d+) and length (\d+). The schedule is (.*) \(Time')

# Contains all of the stats
benchStats = {}
# Contain cumulative stats for the run
cumulativeStats = {}

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
def parseStats(inputFolder, ignoreFolders):
    scanDirPath = os.path.abspath(inputFolder)
    
    # Get name of all directories in the specified folder
    subfolders = [f.name for f in os.scandir(path=scanDirPath) if f.is_dir() ]

    # For each folder
    for folderName in subfolders:
        if folderName in ignoreFolders:
            continue
        name = folderName.split('-')

        # Get the run number from the end
        # of the folder name
        runNumber = name[-1]

        # Get the name of the run
        # and exclude the run number
        nameOfRun = '-'.join(name[:-1])
            
        # Create an entry in the stats for the
        # name of the run
        if (nameOfRun not in benchStats):
            benchStats[nameOfRun] = {}
            cumulativeStats[nameOfRun] = {}

        cumulativeStats[nameOfRun]['average'] = 0.0
        cumulativeStats[nameOfRun]['total'] = 0.0
        cumulativeStats[nameOfRun]['numRegions'] = 0
        cumulativeStats[nameOfRun]['maxLength'] = 0

        for bench in benchmarks:
            # Get the path to the log file
            currentPath = os.path.join(inputFolder, folderName)
            currentPath = os.path.join(currentPath, bench)
            currentLogFile = os.path.join(currentPath, bench + '.log')

            stats = {}
            stats['average'] = 0.0
            stats['total'] = 0.0
            stats['numRegions'] = 0
            stats['maxLength'] = 0

            # First check if log file exists.
            if os.path.exists(currentLogFile):
                # Open log file if it exists.
                with open(currentLogFile) as file: 
                    # Read the whole log file
                    # and split the scheduling
                    # regions into a list
                    log = file.read()
                    blocks = log.split('********** Opt Scheduling **********')[1:]

                    # Iterate over each scheduling region
                    for block in blocks:
                        dagName = ''
                        schedLength = 0

                        # Skip first pass because it isn't the
                        # final schedule
                        getPassNum = RE_PASS_NUM.search(block)
                        if getPassNum:
                            passNum = getPassNum.group(1)
                            if passNum == 'first':
                                continue

                        # First check if B&B is enabled because
                        # with B&B enabled, the final output will
                        # be different.
                        # If B&B is not enabled, check for
                        # schedule from heuristic.
                        DAGInfo = RE_DAG_INFO.search(block)
                        if (DAGInfo):
                            dagName = DAGInfo.group(1)
                            schedLength = int(DAGInfo.group(3))
                        else:
                            getSchedLength = RE_SCHED_LENGTH.search(block)
                            schedLength = int(getSchedLength.group(1))

                        stats['total'] += schedLength
                        stats['numRegions'] += 1

                        if stats['maxLength'] < schedLength:
                            stats['maxLength'] = schedLength
                        
                if stats['numRegions'] != 0:
                    stats['average'] = stats['total']/stats['numRegions']
                    
                benchStats[nameOfRun][bench] = stats

                cumulativeStats[nameOfRun]['total'] += stats['total']
                cumulativeStats[nameOfRun]['numRegions'] += stats['numRegions']
                if cumulativeStats[nameOfRun]['maxLength'] < stats['maxLength']:
                    cumulativeStats[nameOfRun]['maxLength'] = stats['maxLength']

        if cumulativeStats[nameOfRun]['numRegions'] != 0:
            cumulativeStats[nameOfRun]['average'] = float(cumulativeStats[nameOfRun]['total']) \
                                                    / cumulativeStats[nameOfRun]['numRegions']

def printStats():
    for nameOfRun in benchStats:
        print('{}'.format(nameOfRun))
        for bench in benchmarks:
            print('    {} : Average: {:0.2f} Max : {}'.format(bench,
                                                              benchStats[nameOfRun][bench]['average'],
                                                              benchStats[nameOfRun][bench]['maxLength']))
        print('  Overall Average : {:0.2f} Overall Max : {}'.format(cumulativeStats[nameOfRun]['average'],
                                                     cumulativeStats[nameOfRun]['maxLength']))

def createSpreadsheets(output):
    if 'xls' not in output[-4:]:
        output += '.xlsx'
    
    # Create new excel worksheet
    wb = Workbook()

    # Grab the active worksheet
    ws = wb.active

    # Insert title and benchmark names
    ws['A1'] = 'Benchmarks'
    ws['A1'].font = Font(bold=True)

    row = 3
    for bench in benchmarks:
        ws['A' + str(row)] = bench
        row += 1
        
    ws['A' + str(row)] = 'Overall'
    ws['A' + str(row)].font = Font(bold=True)

    # Stats entry
    col = 'B'
    for nameOfRun in benchStats:
        row = 1
        ws[col + str(row)] = nameOfRun
        row = 2
        ws[col+str(row)] = 'Average Sched. Length'
        ws[chr(ord(col)+1)+str(row)] = 'Max Sched. Length'
        
        row = 3
        for bench in benchmarks:
            ws[col+str(row)] = benchStats[nameOfRun][bench]['average']
            ws[chr(ord(col)+1)+str(row)] = benchStats[nameOfRun][bench]['maxLength']
            row += 1
        ws[col+str(row)] = cumulativeStats[nameOfRun]['average']
        ws[chr(ord(col)+1)+str(row)] = cumulativeStats[nameOfRun]['maxLength']
        
        # Convert column char to ASCII value
        # then increment it and convert
        # back into char. Used to go to next
        # column for next test run.
        col = chr(ord(col)+2)

    wb.save(output)

def main(args):
    # Parse folders to ignore into a list
    ignoreFolders = args.ignoreFolders.split(',')

    # Start stats collection
    parseStats(args.inputFolder, ignoreFolders)

    if args.verbose:
        printStats()

    if not args.disable:
        createSpreadsheets(args.output)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to extract average schedule length.',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--verbose', '-v',
                        action='store_true', default=False,
                        dest='verbose',
                        help='Print average schedule lengths to terminal')

    parser.add_argument('--output', '-o',
                        default='schedule-length',
                        dest='output',
                        help='Output spreadsheet filepath')

    parser.add_argument('--disable', '-d',
                        action='store_true', default=False,
                        dest='disable',
                        help='Disable spreadsheet output.')

    parser.add_argument('--input', '-i',
                        default='.',
                        dest='inputFolder',
                        help='The path to scan for benchmark directories')

    parser.add_argument('--ignore',
                        type=str,
                        default='',
                        dest='ignoreFolders',
                        help='List of folders to ignore separated by semi-colon')

    args = parser.parse_args()

    main(args)
