#!/usr/bin/python2
# Wrapper for the runspec tool. Use with the OptSched scheduler
# and LLVM to collect data during instruction scheduling when
# compiling the CPU2006 benchmarks.

from __future__ import division
import optparse
import re
import subprocess
import sys
import os
import shutil
import pdb

## Configuration

# Dictionary for benchmark selection. Maps benchmark names and categories
# to lists of specific benchmarks.
benchDict = {}

# Add all benchmarks individually.
benchDict['perlbench'] = ['perlbench']
benchDict['bzip2'] = ['bzip2']
benchDict['gcc'] = ['gcc']
benchDict['mcf'] = ['mcf']
benchDict['gobmk'] = ['gobmk']
benchDict['hmmer'] = ['hmmer']
benchDict['sjeng'] = ['sjeng']
benchDict['libquantum'] = ['libquantum']
benchDict['h264ref'] = ['h264ref']
benchDict['omnetpp'] = ['omnetpp']
benchDict['astar'] = ['astar']
benchDict['xalancbmk'] = ['xalancbmk']
benchDict['bwaves'] = ['bwaves']
benchDict['gamess'] = ['gamess']
benchDict['milc'] = ['milc']
benchDict['zeusmp'] = ['zeusmp']
benchDict['gromacs'] = ['gromacs']
benchDict['cactus'] = ['cactus']
benchDict['leslie'] = ['leslie']
benchDict['namd'] = ['namd']
benchDict['dealII'] = ['dealII']
benchDict['soplex'] = ['soplex']
benchDict['povray'] = ['povray']
benchDict['calculix'] = ['calculix']
benchDict['Gems'] = ['Gems']
benchDict['tonto'] = ['tonto']
benchDict['lbm'] = ['lbm']
benchDict['wrf'] = ['wrf']
benchDict['sphinx'] = ['sphinx']

# Add ALL benchmark group.
benchDict['ALL'] = [
    'perlbench',
    'bzip2',
    'gcc',
    'mcf',
    'gobmk',
    'hmmer',
    'sjeng',
    'libquantum',
    'h264ref',
    'omnetpp',
    'astar',
    'xalancbmk',
    'bwaves',
    'gamess',
    'milc',
    'zeusmp',
    'gromacs',
    'cactus',
    'leslie',
    'namd',
    'dealII',
    'soplex',
    'povray',
    'calculix',
    'Gems',
    'tonto',
    'lbm',
    'wrf',
    'sphinx'
]

# Add INT benchmark group.
benchDict['INT'] = [
    'perlbench',
    'bzip2',
    'gcc',
    'mcf',
    'gobmk',
    'hmmer',
    'sjeng',
    'libquantum',
    'h264ref',
    'omnetpp',
    'astar',
    'xalancbmk'
]

# Add FP benchmark group.
benchDict['FP'] = [
    'bwaves',
    'gamess',
    'milc',
    'zeusmp',
    'gromacs',
    'cactus',
    'leslie',
    'namd',
    'dealII',
    'soplex',
    'povray',
    'calculix',
    'Gems',
    'tonto',
    'lbm',
    'wrf',
    'sphinx'
]

# Add C benchmark group.
benchDict['C'] = [
    'perlbench',
    'bzip2',
    'gcc',
    'mcf',
    'milc',
    'gobmk',
    'hmmer',
    'sjeng',
    'libquantum',
    'h264ref',
    'lbm',
    'sphinx'
]

# Add C++ benchmark group.
benchDict['C++'] = [
    'namd',
    'dealII',
    'soplex',
    'povray',
    'omnetpp',
    'astar',
    'xalancbmk'
]

# Add the FORTRAN benchmark group.
benchDict['FORTRAN'] = [
    'bwaves',
    'gamess',
    'zeusmp',
    'leslie',
    'Gems',
    'tonto'
]

# Add the MIXED language benchmark group.
benchDict['MIXED'] = [
    'gromacs',
    'cactus',
    'calculix',
    'wrf'
]

# The FP benchmarks without FORTRAN. (ie no dragonegg)
benchDict['FP_NO_F'] = list(set(benchDict['FP']) - set(benchDict['FORTRAN'] + benchDict['MIXED']))

#List of log files
logFile = {}

#Add all log files
logFile['ALL'] = [
	'adpcm',
	'epic',
	'g721',
	'gsm',
	'jpeg',
	'pegwit'
]

#Add log files individually
logFile['adpcm'] = ['adpcm']
logFile['epic'] = ['epic']
logFile['g721'] = ['g721']
logFile['gsm'] = ['gsm']
logFile['jpeg'] = ['jpeg']
logFile['pegwit'] = ['pegwit']

BUILD_COMMAND = "runspec --loose -size=ref -iterations=1 -config=%s --tune=base -r 1 -I -a build %s"
SCRUB_COMMAND = "runspec --loose -size=ref -iterations=1 -config=%s --tune=base -r 1 -I -a scrub %s"
LOG_DIR = 'logs/'

# Regular expressions.
SETTING_REGEX = re.compile(r'\bUSE_OPT_SCHED\b.*')
#SPILLS_REGEX = re.compile(r'Function: (.*?)\nEND FAST RA: Number of spills: (\d+)\n')
SPILLS_REGEX = re.compile(r'Function: (.*?)\nGREEDY RA: Number of spilled live ranges: (\d+)')
#SPILLS_REGEX = re.compile(r'Function: (.*?)\nTotal Simulated Spills: (\d+)')
TIMES_REGEX = re.compile(r'(\d+) total seconds elapsed')
BLOCK_NAME_AND_SIZE_REGEX = re.compile(r'Processing DAG (.*) with (\d+) insts')
BLOCK_NOT_ENUMERATED_REGEX = re.compile(r'The list schedule .* is optimal')
BLOCK_ZERO_TIME_LIMIT = re.compile(r'Bypassing optimal scheduling due to zero time limit')
BLOCK_ENUMERATED_OPTIMAL_REGEX = re.compile(r'DAG solved optimally')
BLOCK_COST_REGEX = re.compile(r'list schedule is of length \d+ and spill cost \d+. Tot cost = (\d+)')
BLOCK_IMPROVEMENT_REGEX = re.compile(r'cost imp=(\d+)')
BLOCK_START_TIME_REGEX = re.compile(r'-{20} \(Time = (\d+) ms\)')
BLOCK_END_TIME_REGEX = re.compile(r'verified successfully \(Time = (\d+) ms\)')
BLOCK_LIST_FAILED_REGEX = re.compile(r'List scheduling failed')
BLOCK_RP_MISMATCH = re.compile(r'RP-mismatch falling back!')
BLOCK_PEAK_REG_PRESSURE_REGEX = re.compile(r'PeakRegPresAfter Index (\d+) Name (.*) Peak (\d+) Limit (\d+)')
BLOCK_PEAK_REG_BLOCK_NAME = re.compile(r'LLVM max pressure after scheduling for BB (\S+)')
REGION_OPTSCHED_SPILLS_REGEX = re.compile(r"OPT_SCHED LOCAL RA: DAG Name: (\S+) Number of spills: (\d+) \(Time")

def writeStats(stats, spills, times, blocks, regp, trackOptSchedSpills):
    # Write times.
    if times:
        with open(times, 'w') as times_file:
            total_time = 0
            for benchName in stats:
                time = stats[benchName]['time']
                total_time += time
                times_file.write('%10s:%5d seconds\n' % (benchName, time))
            times_file.write('---------------------------\n')
            times_file.write('     Total:%5d seconds\n' % total_time)

    # Write spill stats.
    if spills:
        with open(spills, 'w') as spills_file:
            totalSpills = 0
            for benchName in stats:
                totalSpillsPerBenchmark = 0
                spills_file.write('%s:\n' % benchName)
                spills = stats[benchName]['spills']
                for functionName in spills:
                    spillCount = spills[functionName]
                    totalSpillsPerBenchmark += spillCount
                    spills_file.write('      %5d %s\n' %
                                      (spillCount, functionName))
                spills_file.write('  ---------\n')
                spills_file.write('  Sum:%5d\n\n' % totalSpillsPerBenchmark)
                totalSpills += totalSpillsPerBenchmark
            spills_file.write('------------\n')
            spills_file.write('Total:%5d\n' % totalSpills)

    # write simulated region spill stats.
    # TODO Write simulated spills for all regions.
    #if simspills:
    #    with open(simspills, 'w') as simspills_file:

    # Write block stats.
    if blocks:
        with open(blocks, 'w') as blocks_file:
            totalCount = 0
            totalSuccessful = 0
            totalEnumerated = 0
            totalOptimalImproved = 0
            totalOptimalNotImproved = 0
            totalTimedOutImproved = 0
            totalTimedOutNotImproved = 0
            totalCost = 0
            totalImprovement = 0
            totalOptSchedSpills = 0
            totalEnumeratedSizes = 0
            totalOptimalImprovedSizes = 0
            totalOptimalNotImprovedSizes = 0
            totalNonOptimalImprovedSizes = 0
            totalNonOptimalNotImprovedSizes = 0
            totalEnumeratedSpills = 0
            totalOptimalImprovedSpills = 0
            totalOptimalNotImprovedSpills = 0
            totalNonOptimalImprovedSpills = 0
            totalNonOptimalNotImprovedSpills = 0
            sizesList = []
            optimalTimesList = []
            enumeratedSizesList = []
            optimalSizesList = []
            improvedSizesList = []
            timedOutSizesList = []

            for benchName in stats:
                blocks = stats[benchName]['blocks']
                count = 0
                successful = 0
                enumerated = 0
                optimalImproved = 0
                optimalNotImproved = 0
                timedOutImproved = 0
                timedOutNotImproved = 0
                cost = improvement = 0
                optSchedSpills = 0
                enumeratedSizes = 0
                optimalImprovedSizes = 0
                optimalNotImprovedSizes = 0
                nonOptimalImprovedSizes = 0
                nonOptimalNotImprovedSizes = 0
                enumeratedSpills = 0
                optimalImprovedSpills = 0
                optimalNotImprovedSpills = 0
                nonOptimalImprovedSpills = 0
                nonOptimalNotImprovedSpills = 0


                for block in blocks:
                    count += 1
                    if block['success']:
                        successful += 1
                        cost += block['listCost']
                        size = block['size']
                        spills = block['optSchedSpills']
                        sizesList.append(size)
                        optSchedSpills += spills
                        if block['isEnumerated']:
                            enumerated += 1
                            improvement += block['improvement']
                            enumeratedSizesList.append(size)
                            enumeratedSizes += size
                            enumeratedSpills += spills
                            if block['isOptimal']:
                                optimalTimesList.append(block['time'])
                                optimalSizesList.append(size)
                                if block['improvement'] > 0:
                                    improvedSizesList.append(size)
                                    optimalImprovedSizes += size
                                    optimalImprovedSpills += spills
                                    optimalImproved += 1
                                else:
                                    optimalNotImproved += 1
                                    optimalNotImprovedSizes += size
                                    optimalNotImprovedSpills += spills
                            else:
                                timedOutSizesList.append(block['size'])
                                if block['improvement'] > 0:
                                    improvedSizesList.append(size)
                                    nonOptimalImprovedSizes += size
                                    nonOptimalImprovedSpills += spills
                                    timedOutImproved += 1
                                else:
                                    nonOptimalNotImprovedSizes += size
                                    nonOptimalNotImprovedSpills += spills
                                    timedOutNotImproved += 1

                # If the option to track simulated spills is enabled, construct the strings that display
                # information about the number of instructions and the number of spills for the categories
                # of blocks below.
                enumeratedStr = ''
                optimalImprovedStr = ''
                optimalNotImprovedStr = ''
                nonOptimalImprovedStr = ''
                nonOptimalNotImprovedStr = ''
                if trackOptSchedSpills and enumerated > 0:
                    try:
                        enumeratedStr += '  {:,} instrs, {:,} spills'.format(enumeratedSizes, enumeratedSpills)

                        optimalImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(optimalImprovedSizes, \
                                               optimalImprovedSizes / enumeratedSizes, optimalImprovedSpills, optimalImprovedSpills / enumeratedSpills)

                        optimalNotImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(optimalNotImprovedSizes, \
                                               optimalNotImprovedSizes / enumeratedSizes, optimalNotImprovedSpills, optimalNotImprovedSpills / enumeratedSpills)

                        nonOptimalImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(nonOptimalImprovedSizes, \
                                               nonOptimalImprovedSizes / enumeratedSizes, nonOptimalImprovedSpills, nonOptimalImprovedSpills / enumeratedSpills)

                        nonOptimalNotImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(nonOptimalNotImprovedSizes, \
                                               nonOptimalNotImprovedSizes / enumeratedSizes, nonOptimalNotImprovedSpills, nonOptimalNotImprovedSpills / enumeratedSpills)
                    except ZeroDivisionError:
                        print('There are 0 OptSched spills in enumerated blocks, cannot print any useful information related to them.')

                blocks_file.write('%s:\n' % benchName)
                blocks_file.write('  Blocks: %d\n' %
                                  count)
                blocks_file.write('  Successful: %d (%.2f%%)\n' %
                                  (successful, (100 * successful / count) if count else 0))
                blocks_file.write('  Enumerated: %d (%.2f%%)%s\n' %
                                  (enumerated, (100 * enumerated / successful) if successful else 0, enumeratedStr))
                blocks_file.write('  Optimal and Improved: %d (%.2f%%)%s\n' %
                                  (optimalImproved, (100 * optimalImproved / enumerated) if enumerated else 0, optimalImprovedStr))
                blocks_file.write('  Optimal but not Improved: %d (%.2f%%)%s\n' %
                                  (optimalNotImproved, (100 * optimalNotImproved / enumerated) if enumerated else 0, optimalNotImprovedStr))
                blocks_file.write('  Non-Optimal and Improved: %d (%.2f%%)%s\n' %
                                  (timedOutImproved, (100 * timedOutImproved / enumerated) if enumerated else 0, nonOptimalImprovedStr))
                blocks_file.write('  Non-Optimal and not Improved: %d (%.2f%%)%s\n' %
                                  (timedOutNotImproved, (100 * timedOutNotImproved / enumerated) if enumerated else 0, nonOptimalNotImprovedStr))
                blocks_file.write('  Heuristic cost: %d\n' %
                                  cost)
                blocks_file.write('  B&B cost: %d\n' %
                                  (cost - improvement))
                blocks_file.write('  Cost improvement: %d (%.2f%%)\n' %
                                  (improvement, (100 * improvement / cost) if cost else 0))
                if trackOptSchedSpills:
                    blocks_file.write('  Simulated Block Spills: %d\n' %
                                    optSchedSpills)

                totalCount += count
                totalSuccessful += successful
                totalEnumerated += enumerated
                totalOptimalImproved += optimalImproved
                totalOptimalNotImproved += optimalNotImproved
                totalTimedOutImproved += timedOutImproved
                totalTimedOutNotImproved += timedOutNotImproved
                totalCost += cost
                totalImprovement += improvement
                totalOptSchedSpills += optSchedSpills
                totalEnumeratedSizes += enumeratedSizes
                totalOptimalImprovedSizes += optimalImprovedSizes
                totalOptimalNotImprovedSizes += optimalNotImprovedSizes
                totalNonOptimalImprovedSizes += nonOptimalImprovedSizes
                totalNonOptimalNotImprovedSizes += nonOptimalNotImprovedSizes
                totalEnumeratedSpills += enumeratedSpills
                totalOptimalImprovedSpills += optimalImprovedSpills
                totalOptimalNotImprovedSpills += optimalNotImprovedSpills
                totalNonOptimalImprovedSpills += nonOptimalImprovedSpills
                totalNonOptimalNotImprovedSpills += nonOptimalNotImprovedSpills


            totalEnumeratedStr = ''
            totalOptimalImprovedStr = ''
            totalOptimalNotImprovedStr = ''
            totalNonOptimalImprovedStr = ''
            totalNonOptimalNotImprovedStr = ''
            if trackOptSchedSpills and enumerated > 0:
                try:
                    totalEnumeratedStr += '  {:,} instrs, {:,} spills'.format(totalEnumeratedSizes, totalEnumeratedSpills)

                    totalOptimalImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(totalOptimalImprovedSizes, \
                                           totalOptimalImprovedSizes / totalEnumeratedSizes, totalOptimalImprovedSpills, totalOptimalImprovedSpills / totalEnumeratedSpills)

                    totalOptimalNotImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(totalOptimalNotImprovedSizes, \
                                           totalOptimalNotImprovedSizes / totalEnumeratedSizes, totalOptimalNotImprovedSpills, totalOptimalNotImprovedSpills / totalEnumeratedSpills)

                    totalNonOptimalImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(totalNonOptimalImprovedSizes, \
                                           totalNonOptimalImprovedSizes / totalEnumeratedSizes, totalNonOptimalImprovedSpills, totalNonOptimalImprovedSpills / totalEnumeratedSpills)

                    totalNonOptimalNotImprovedStr += '  {:,} instrs ({:.2%}), {:,} spills ({:.2%}) '.format(totalNonOptimalNotImprovedSizes, \
                                           totalNonOptimalNotImprovedSizes / totalEnumeratedSizes, totalNonOptimalNotImprovedSpills, totalNonOptimalNotImprovedSpills / totalEnumeratedSpills)
                except ZeroDivisionError:
                    print('There are 0 OptSched spills in enumerated blocks, cannot print any useful information related to them.')


            blocks_file.write('-' * 50 + '\n')
            blocks_file.write('Total:\n')
            blocks_file.write('  Blocks: %d\n' %
                              totalCount)
            blocks_file.write('  Successful: %d (%.2f%%)\n' %
                              (totalSuccessful, (100 * totalSuccessful / totalCount) if totalCount else 0))
            blocks_file.write('  Enumerated: %d (%.2f%%)%s\n' %
                              (totalEnumerated, (100 * totalEnumerated / totalSuccessful) if totalSuccessful else 0, totalEnumeratedStr))
            blocks_file.write('  Optimal and Improved: %d (%.2f%%)%s\n' %
                              (totalOptimalImproved, (100 * totalOptimalImproved / totalEnumerated) if totalEnumerated else 0, totalOptimalImprovedStr))
            blocks_file.write('  Optimal but not Improved: %d (%.2f%%)%s\n' %
                              (totalOptimalNotImproved, (100 * totalOptimalNotImproved / totalEnumerated) if totalEnumerated else 0, totalOptimalNotImprovedStr))
            blocks_file.write('  Non-Optimal and Improved: %d (%.2f%%)%s\n' %
                              (totalTimedOutImproved, (100 * totalTimedOutImproved / totalEnumerated) if totalEnumerated else 0, totalNonOptimalImprovedStr))
            blocks_file.write('  Non-Optimal and not Improved: %d (%.2f%%)%s\n' %
                              (totalTimedOutNotImproved, (100 * totalTimedOutNotImproved / totalEnumerated) if totalEnumerated else 0, totalNonOptimalNotImprovedStr))
            blocks_file.write('  Heuristic cost: %d\n' %
                              totalCost)
            blocks_file.write('  B&B cost: %d\n' %
                              (totalCost - totalImprovement))
            blocks_file.write('  Cost improvement: %d (%.2f%%)\n' %
                              (totalImprovement, (100 * totalImprovement / totalCost) if totalCost else 0))
            if trackOptSchedSpills:
                blocks_file.write('  Total Simulated Block Spills: %d\n' %
                                totalOptSchedSpills)
            blocks_file.write('  Smallest block size: %s\n' %
                              (min(sizesList) if sizesList else 'none'))
            blocks_file.write('  Largest block size: %s\n' %
                              (max(sizesList) if sizesList else 'none'))
            blocks_file.write('  Average block size: %.1f\n' %
                              ((sum(sizesList) / len(sizesList)) if sizesList else 0))
            blocks_file.write('  Smallest enumerated block size: %s\n' %
                              (min(enumeratedSizesList) if enumeratedSizesList else 'none'))
            blocks_file.write('  Largest enumerated block size: %s\n' %
                              (max(enumeratedSizesList) if enumeratedSizesList else 'none'))
            blocks_file.write('  Average enumerated block size: %.1f\n' %
                              ((sum(enumeratedSizesList) / len(enumeratedSizesList)) if enumeratedSizesList else 0))
            blocks_file.write('  Largest optimal block size: %s\n' %
                              (max(optimalSizesList) if optimalSizesList else 'none'))
            blocks_file.write('  Largest improved block size: %s\n' %
                              (max(improvedSizesList) if improvedSizesList else 'none'))
            blocks_file.write('  Smallest timed out block size: %s\n' %
                              (min(timedOutSizesList) if timedOutSizesList else 'none'))
            blocks_file.write('  Average optimal solution time: %d ms\n' %
                              ((sum(optimalTimesList) / len(optimalTimesList)) if optimalTimesList else 0))

        # Write peak pressure stats
        if regp:
            with open(regp, 'w') as regp_file:
                for benchName in stats:
                    regp_file.write('Benchmark %s:\n' % benchName)
                    regpressure = stats[benchName]['regpressure']
                    numberOfFunctionsWithPeakExcess = 0
                    numberOfBlocksWithPeakExcess = 0
                    numberOfBlocks = 0
                    peakPressureSetSumsPerBenchmark = {}
                    for functionName in regpressure:
                        peakPressureSetSums = {}
                        regp_file.write('  Function %s:\n' % functionName)
                        listOfBlocks = regpressure[functionName]
                        if len(listOfBlocks) == 0:
                            continue
                        for blockName, listOfExcessPressureTuples in listOfBlocks:
                            numberOfBlocks += 1
                            if len(listOfExcessPressureTuples) == 0:
                                continue
                            # regp_file.write('    Block %s:\n' % blockName)
                            for setName, peakExcessPressure in listOfExcessPressureTuples:
                                # If we ever enter this loop, that means there exists a peak excess pressure
                                # regp_file.write('      %5d %s\n' % (peakExcessPressure, setName))
                                if not setName in peakPressureSetSums:
                                    peakPressureSetSums[setName] = peakExcessPressure
                                else:
                                    peakPressureSetSums[setName] += peakExcessPressure
                        regp_file.write(
                            '  Pressure Set Sums for Function %s:\n' % functionName)
                        for setName in peakPressureSetSums:
                            regp_file.write('    %5d %s\n' %
                                            (peakPressureSetSums[setName], setName))
                            if not setName in peakPressureSetSumsPerBenchmark:
                                peakPressureSetSumsPerBenchmark[setName] = peakPressureSetSums[setName]
                            else:
                                peakPressureSetSumsPerBenchmark[setName] += peakPressureSetSums[setName]
                    regp_file.write(
                        'Pressure Set Sums for Benchmark %s:\n' % benchName)
                    for setName in peakPressureSetSumsPerBenchmark:
                        regp_file.write('%5d %s\n' % (
                            peakPressureSetSumsPerBenchmark[setName], setName))
                    # regp_file.write('Number of blocks with peak excess:    %d\n' % numberOfBlocksWithPeakExcess)
                    # regp_file.write('Number of blocks total:               %d\n' % numberOfBlocks)
                    # regp_file.write('Number of functions with peak excess: %d\n' % numberOfFunctionsWithPeakExcess)
                    # regp_file.write('Number of functions total:            %d\n' % len(regpressure))
                    regp_file.write('------------\n')


def calculatePeakPressureStats(output):
    """
    Output should look like:
    Benchmark:
      Function1:
        Block1:
          PERP1 RegName1
          PERP2 RegName2
          ...
      Function1 Peak: PERPMax RegNameMax
    Number of blocks with peak excess register pressure: M
    Number of blocks total: N
    Number of functions with excess register pressure: F
    Number of functions total: G

    Then the data structure will look like:
    {
      function1: [
        (block1, [(SetName1, PERP1), ...]),
        ...
        ],
      function2: ...
    }
    """
    blocks = output.split('Scheduling **********')[1:]
    functions = {}
    for block in blocks:
        if len(BLOCK_PEAK_REG_BLOCK_NAME.findall(block)) == 0:
            continue
        dagName = BLOCK_PEAK_REG_BLOCK_NAME.findall(block)[0]
        functionName = dagName.split(':')[0]
        blockName = dagName.split(':')[1]
        pressureMatches = BLOCK_PEAK_REG_PRESSURE_REGEX.findall(block)
        peakExcessPressures = []
        for indexString, name, peakString, limitString in pressureMatches:
            peak = int(peakString)
            limit = int(limitString)
            excessPressure = peak - limit
            if excessPressure < 0:
                excessPressure = 0
            element = tuple((name, excessPressure))
            peakExcessPressures.append(element)
        if len(peakExcessPressures) > 0:
            blockStats = (blockName, peakExcessPressures)
            if not functionName in functions:
                functions[functionName] = []
            functions[functionName].append(blockStats)
    return functions


def calculateBlockStats(output, trackOptSchedSpills):
    blocks = output.split('Opt Scheduling **********')[1:]
    stats = []
    for index, block in enumerate(blocks):
        lines = [line[6:]
                 for line in block.split('\n') if line.startswith('INFO:')]
        block = '\n'.join(lines)

        try:
            name, size = BLOCK_NAME_AND_SIZE_REGEX.findall(block)[0]

            failed = BLOCK_LIST_FAILED_REGEX.findall(
                block) != [] or BLOCK_RP_MISMATCH.findall(block) != []

            if failed:
                timeTaken = 0
                isEnumerated = isOptimal = False
                listCost = improvement = 0
            else:
                start_time = int(BLOCK_START_TIME_REGEX.findall(block)[0])
                end_time = int(BLOCK_END_TIME_REGEX.findall(block)[0])
                timeTaken = end_time - start_time
                if REGION_OPTSCHED_SPILLS_REGEX.findall(block) and trackOptSchedSpills:
                    optSchedSpills = int(REGION_OPTSCHED_SPILLS_REGEX.findall(block)[0][1])
                else:
                    optSchedSpills = 0

                listCost = int(BLOCK_COST_REGEX.findall(block)[0])
                # The block is not enumerated if the list schedule is optimal or there is a zero
                # time limit for enumeration.
                isEnumerated = (BLOCK_NOT_ENUMERATED_REGEX.findall(block) == []) and (
                    BLOCK_ZERO_TIME_LIMIT.findall(block) == [])
                if isEnumerated:
                    isOptimal = bool(
                        BLOCK_ENUMERATED_OPTIMAL_REGEX.findall(block))
                    """
                    Sometimes the OptScheduler doesn't print out cost improvement.
                    This happens when the scheduler determines that the list schedule is
                    already optimal, which means no further improvements can be made.

                    The rest of this if-block ensures that this tool doesn't crash.
                    If the improvement is not found, it is assumed to be 0.
                    """
                    matches = BLOCK_IMPROVEMENT_REGEX.findall(block)
                    if matches == []:
                        improvement = 0
                    else:
                        improvement = int(matches[0])
                else:
                    isOptimal = False
                    improvement = 0

            stats.append({
                'name': name,
                'size': int(size),
                'time': timeTaken,
                'success': not failed,
                'isEnumerated': isEnumerated,
                'isOptimal': isOptimal,
                'listCost': listCost,
                'improvement': improvement,
                'optSchedSpills': optSchedSpills
            })
        except:
            print '  WARNING: Could not parse block #%d:' % (index + 1)
            print "Unexpected error:", sys.exc_info()[0]
            for line in blocks[index].split('\n')[1:-1][:10]:
                print '   ', line

    return stats


def calculateSpills(output):
    spills = {}
    for functionName, spillCountString in SPILLS_REGEX.findall(output):
        spills[functionName] = int(spillCountString)
    return spills


"""
Defining this function makes it easier to parse files.
"""


def getBenchmarkResult(output, trackOptSchedSpills):
    # Handle parsing log files that were not generated by runspec and have no time information.
    time = int(TIMES_REGEX.findall(output)[1]) if len(TIMES_REGEX.findall(output)) > 0 else -1

    return {
        'time': time,
        'spills': calculateSpills(output),
        'blocks': calculateBlockStats(output, trackOptSchedSpills),
        'regpressure': calculatePeakPressureStats(output)
    }


def runBenchmarks(benchmarks, testOutDir, shouldWriteLogs, config, trackOptSchedSpills):
    results = {}

    for bench in benchmarks:
        print 'Running', bench
        try:
            p = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
            p.stdin.write("source shrc" + "\n")
            p.stdin.write(SCRUB_COMMAND % (config, bench) + "\n")
            p.stdin.write(BUILD_COMMAND % (config, bench))
            p.stdin.close()
            output = p.stdout.read()

        except subprocess.CalledProcessError as e:
            print '  WARNING: Benchmark command failed: %s.' % e
        else:
            results[bench] = getBenchmarkResult(output, trackOptSchedSpills)

            # Optionally write log files to results directory.
            if shouldWriteLogs is True:
                writeLogs(output, testOutDir, bench)

    return results

# Write log files for a benchmark to the results directory.


def writeLogs(output, testOutDir, bench):
    with open(os.path.join(testOutDir,  LOG_DIR + bench + '.log'), 'w') as log:
        log.write(output)


def main(args):
    ## Select benchmarks.
    benchArgs = args.bench.split(',')
    # List of benchmarks to run.
    benchmarks = []
    for benchArg in benchArgs:
        if benchArg not in benchDict:
            print 'Fatal: Unknown benchmark specified: "%s".' % benchArg
            if benchArg == 'all':
                print 'Did you mean ALL?'
            sys.exit(1)

        benchmarks = list(set(benchmarks + benchDict[benchArg]))

    # Parse a log file or multiple log files instead of running benchmark
    results = {}
    if args.logfile is not None:
	logArgs = args.logfile.split(',')
	logfiles = []
	for logArg in logArgs:
		if logArg not in logFile:
			print 'Fatal: Unknown log file specified: "%s".' % logArg
			if logArg == 'all':
				print 'Did you mean ALL?'
			sys.exit(1)

		logfiles = list(set(logfiles + logFile[logArg]))

        for log in logfiles:
		with open('./' + log + '/' + log + '.log') as log_file:
			output = log_file.read()
			results[log] = getBenchmarkResult(output, args.trackOptSchedSpills)

		spills = os.path.join(args.outdir, args.spills)
	        times = os.path.join(args.outdir, args.times)
        	blocks = os.path.join(args.outdir, args.blocks)

		if args.regp:
	            	regp = os.path.join(testOutDir, args.regp)
        	else:
            		regp = None

        	# Write out the results from the logfile.
        	writeStats(results, spills, times, blocks, regp, args.trackOptSchedSpills)

	# Run the benchmarks and collect results.
    else:
        # Create a directory for the test results.
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        # Run "testruns" TODO(guess the number of tests) number of tests. Try to find a seperate ini file for each test.
        for i in range(int(args.testruns)):
            testOutDir = args.outdir

            if args.ini:
                if not args.cfg:
                    print('Fatal: No path to the OptSchedCfg directory found. Use option "-g" to specify the path.')
                    sys.exit(1)
                else:
                    iniFileName = [filename for filename in os.listdir(args.ini) if filename.split('.')[0] == str(i)]

                    # Move test ini file to OptSchedCfg directroy so the compiler uses it for this test.
                    shutil.copy(os.path.join(args.ini, iniFileName[0]), os.path.join(args.cfg, 'sched.ini'))

                    # Create a directory for this test run.
                    testOutDir = args.outdir
                    # If an ini file is known for this test, name the result directory after the
                    # ini file.
                    testOutDir = os.path.join(testOutDir, iniFileName[0].split('.')[1])

                    if not os.path.exists(testOutDir):
                        os.makedirs(testOutDir)

                    # Add a copy of the ini file to the results directory.
                    shutil.copy(os.path.join(args.ini, iniFileName[0]), testOutDir)

            # If we are writing log files create a directory for them.
            if args.writelogs:
                if not os.path.exists(os.path.join(testOutDir, LOG_DIR)):
                    os.makedirs(os.path.join(testOutDir, LOG_DIR))

            # Run the benchmarks
            results = runBenchmarks(benchmarks, testOutDir, args.writelogs, args.config, args.trackOptSchedSpills)


            spills = os.path.join(testOutDir, args.spills)
            times = os.path.join(testOutDir, args.times)
            blocks = os.path.join(testOutDir, args.blocks)
            if args.regp:
                regp = os.path.join(testOutDir, args.regp)
            else:
                regp = None

            # Write out the results for this test.
            writeStats(results, spills, times, blocks, regp, args.trackOptSchedSpills)


if __name__ == '__main__':
    parser = optparse.OptionParser(
        description='Wrapper around runspec for collecting spill counts and block statistics.')
    parser.add_option('-s', '--spills',
                      metavar='filepath',
                      default='spills.dat',
                      help='Where to write the spill counts (%default).')
    parser.add_option('-t', '--times',
                      metavar='filepath',
                      default='times.dat',
                      help='Where to write the run compile times (%default).')
    parser.add_option('-k', '--blocks',
                      metavar='filepath',
                      default='blocks.dat',
                      help='Where to write the run block stats (%default).')
    parser.add_option('-r', '--regp',
                      metavar='filepath',
                      help='Where to write the reg pressure stats (%default).')
    parser.add_option('-c', '--config',
                      metavar='filepath',
                      default='Intel_llvm_3.9.cfg',
                      help='The runspec config file (%default).')
    parser.add_option('-m', '--testruns',
                      metavar='number',
                      default='1',
                      help='The number of tests to run.')
    parser.add_option('-i', '--ini',
                      metavar='filepath',
                      default=None,
                      help='The directory with the sched.ini files for each test run (%default). They should be named 0.firsttestname.ini, 1.secondtestname.ini...')
    parser.add_option('-g', '--cfg',
                      metavar='filepath',
                      default=None,
                      help='Where to find OptSchedCfg for the test runs (%default).')
    parser.add_option('-b', '--bench',
                      metavar='ALL|INT|FP|name1,name2...',
                      default='ALL',
                      help='Which benchmarks to run.')
    parser.add_option('-l', '--logfile',
                      metavar='*.log',
                      default=None,
                      help='Parse log file(s) instead of running benchmark.')
    parser.add_option('-o', '--outdir',
                      metavar='filepath',
                      default='./',
                      help='Where to write the test results (%default).')
    parser.add_option('-w', '--writelogs',
                      action="store_true",
                      dest="writelogs",
                      help='Should the raw log files be included in the results (%default).')
    parser.add_option('-a', '--trackOptSchedSpills',
                      action="store_true",
                      dest="trackOptSchedSpills",
                      default=True,
                      help='Should the simulated number of spills per-block be tracked (%default).')

    main(parser.parse_args()[0])
