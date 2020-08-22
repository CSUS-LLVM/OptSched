#!/usr/bin/python2

from __future__ import division
import optparse
import re
import subprocess
import sys
import os

# Configuration.
INT_BENCHMARKS = [
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
FP_BENCHMARKS = [
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
ALL_BENCHMARKS = INT_BENCHMARKS + FP_BENCHMARKS
COMMAND = "runspec --loose -size=ref -iterations=1 -config=%s --tune=base -r 1 -I -a build %s"
LOG_FOLDER = "wrapperLogs/"
STATS_FOLDER = "wrapperStats/"

# Regular expressions.
SETTING_REGEX = re.compile(r'\bUSE_OPT_SCHED\b.*')
SPILLS_REGEX = re.compile(r'Function: (.*?)\nGREEDY RA: Number of spilled live ranges: (\d+)\n')
#SPILLS_REGEX = re.compile(r'GREEDY RA: Number of spilled live ranges: (\d+)')
#SPILLS_REGEX = re.compile(r'Spill Cost: (\d+)')
TIMES_REGEX = re.compile(r'(\d+) total seconds elapsed')
BLOCK_NAME_AND_SIZE_REGEX = re.compile(r'Processing DAG (.*) with (\d+) insts')
#BLOCK_NOT_ENUMERATED_REGEX = re.compile(r'The list schedule .* is optimal')
BLOCK_NOT_ENUMERATED_REGEX = re.compile(r'(Bypassing optimal scheduling due to zero time limit|The list schedule .* is optimal)')
BLOCK_ENUMERATED_OPTIMAL_REGEX = re.compile(r'DAG solved optimally')
BLOCK_COST_LOWER_BOUND_REGEX = re.compile(r'Lower bound of cost before scheduling: (\d+)')
BLOCK_COST_REGEX = re.compile(r'list schedule is of length \d+ and spill cost \d+. Tot cost = (\d+)')
BLOCK_IMPROVEMENT_REGEX = re.compile(r'cost imp=(\d+)')
BLOCK_START_TIME_REGEX = re.compile(r'-{20} \(Time = (\d+) ms\)')
BLOCK_END_TIME_REGEX = re.compile(r'verified successfully \(Time = (\d+) ms\)')
BLOCK_LIST_FAILED_REGEX = re.compile(r'List scheduling failed')
BLOCK_PEAK_REG_PRESSURE_REGEX = re.compile(r'PeakRegPresAfter Dag (.*?) Index (\d+) Name (.*) Peak (\d+) Limit (\d+)')
SLIL_HEURISTIC_REGEX = re.compile(r'SLIL after Heuristic Scheduler for dag (.*?) Type (\d+) (.*?) is (\d+)')
BLOCK_FAILED_REGEX = re.compile(r'OptSched run failed')

def writeStats(stats, args, dagSizesPerBenchmark):
    statsFolder = ""
    if not os.path.exists(STATS_FOLDER):
        os.makedirs(STATS_FOLDER)
        statsFolder = STATS_FOLDER
    elif os.path.isdir(STATS_FOLDER):
        statsFolder = STATS_FOLDER

    # Write times.
    with open(os.path.join(statsFolder, args.times), 'w') as times_file:
        total_time = 0
        for benchName in stats:
            time = stats[benchName]['time']
            total_time += time
            times_file.write('%10s:%5d seconds\n' % (benchName, time))
        times_file.write('---------------------------\n')
        times_file.write('     Total:%5d seconds\n' % total_time)

    # Write spill stats.
    with open(os.path.join(statsFolder, args.spills), 'w') as spills_file:
        totalSpills = 0
        for benchName in stats:
            totalSpillsPerBenchmark = 0
            spills_file.write('%s:\n' % benchName)
            spills = stats[benchName]['spills']
            for functionName in spills:
                spillCount = spills[functionName]
                totalSpillsPerBenchmark += spillCount
                spills_file.write('      %5d %s\n' % (spillCount, functionName))
            spills_file.write('  ---------\n')
            spills_file.write('  Sum:%5d\n\n' % totalSpillsPerBenchmark)
            totalSpills += totalSpillsPerBenchmark
        spills_file.write('------------\n')
        spills_file.write('Total:%5d\n' % totalSpills)

    # Write block stats.
    with open(os.path.join(statsFolder, args.blocks), 'w') as blocks_file:
        totalCount = 0
        totalSuccessful = 0
        totalEnumerated = 0
        totalOptimalImproved = 0
        totalOptimalNotImproved = 0
        totalTimedOutImproved = 0
        totalTimedOutNotImproved = 0
        totalCost = 0
        totalImprovement = 0
        sizes = []
        enumeratedSizes = []
        optimalSizes = []
        improvedSizes = []
        timedOutSizes = []
        optimalTimes = []

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

            for block in blocks:
                count += 1
                if block['success']:
                    successful += 1
                    cost += block['listCost']
                    sizes.append(block['size'])
                    if block['isEnumerated']:
                        enumerated += 1
                        improvement += block['improvement']
                        enumeratedSizes.append(block['size'])
                        if block['isOptimal']:
                            optimalTimes.append(block['time'])
                            optimalSizes.append(block['size'])
                            if block['improvement'] > 0:
                                improvedSizes.append(block['size'])
                                optimalImproved += 1
                            else:
                                optimalNotImproved += 1
                        else:
                            timedOutSizes.append(block['size'])
                            if block['improvement'] > 0:
                                improvedSizes.append(block['size'])
                                timedOutImproved += 1
                            else:
                                timedOutNotImproved += 1

            blocks_file.write('%s:\n' % benchName)
            blocks_file.write('  Blocks: %d\n' %
                              count)
            blocks_file.write('  Successful: %d (%.2f%%)\n' %
                              (successful, (100 * successful / count) if count else 0))
            blocks_file.write('  Enumerated: %d (%.2f%%)\n' %
                              (enumerated, (100 * enumerated / successful) if successful else 0))
            blocks_file.write('  Optimal and Improved: %d (%.2f%%)\n' %
                              (optimalImproved, (100 * optimalImproved / enumerated) if enumerated else 0))
            blocks_file.write('  Optimal but not Improved: %d (%.2f%%)\n' %
                              (optimalNotImproved, (100 * optimalNotImproved / enumerated) if enumerated else 0))
            blocks_file.write('  Non-Optimal and Improved: %d (%.2f%%)\n' %
                              (timedOutImproved, (100 * timedOutImproved / enumerated) if enumerated else 0))
            blocks_file.write('  Non-Optimal and not Improved: %d (%.2f%%)\n' %
                              (timedOutNotImproved, (100 * timedOutNotImproved / enumerated) if enumerated else 0))
            blocks_file.write('  Heuristic cost: %d\n' %
                              cost)
            blocks_file.write('  B&B cost: %d\n' %
                              (cost - improvement))
            blocks_file.write('  Cost improvement: %d (%.2f%%)\n' %
                              (improvement, (100 * improvement / cost) if cost else 0))

            totalCount += count
            totalSuccessful += successful
            totalEnumerated += enumerated
            totalOptimalImproved += optimalImproved
            totalOptimalNotImproved += optimalNotImproved
            totalTimedOutImproved += timedOutImproved
            totalTimedOutNotImproved += timedOutNotImproved
            totalCost += cost
            totalImprovement += improvement

        blocks_file.write('-' * 50 + '\n')
        blocks_file.write('Total:\n')
        blocks_file.write('  Blocks: %d\n' %
                          totalCount)
        blocks_file.write('  Successful: %d (%.2f%%)\n' %
                          (totalSuccessful, (100 * totalSuccessful / totalCount) if totalCount else 0))
        blocks_file.write('  Enumerated: %d (%.2f%%)\n' %
                          (totalEnumerated, (100 * totalEnumerated / totalSuccessful) if totalSuccessful else 0))
        blocks_file.write('  Optimal and Improved: %d (%.2f%%)\n' %
                          (totalOptimalImproved, (100 * totalOptimalImproved / totalEnumerated) if totalEnumerated else 0))
        blocks_file.write('  Optimal but not Improved: %d (%.2f%%)\n' %
                          (totalOptimalNotImproved, (100 * totalOptimalNotImproved / totalEnumerated) if totalEnumerated else 0))
        blocks_file.write('  Non-Optimal and Improved: %d (%.2f%%)\n' %
                          (totalTimedOutImproved, (100 * totalTimedOutImproved / totalEnumerated) if totalEnumerated else 0))
        blocks_file.write('  Non-Optimal and not Improved: %d (%.2f%%)\n' %
                          (totalTimedOutNotImproved, (100 * totalTimedOutNotImproved / totalEnumerated) if totalEnumerated else 0))
        blocks_file.write('  Heuristic cost: %d\n' %
                          totalCost)
        blocks_file.write('  B&B cost: %d\n' %
                          (totalCost - totalImprovement))
        blocks_file.write('  Cost improvement: %d (%.2f%%)\n' %
                          (totalImprovement, (100 * totalImprovement / totalCost) if totalCost else 0))

        blocks_file.write('  Smallest block size: %s\n' %
                          (min(sizes) if sizes else 'none'))
        blocks_file.write('  Largest block size: %s\n' %
                          (max(sizes) if sizes else 'none'))
        blocks_file.write('  Average block size: %.1f\n' %
                          ((sum(sizes) / len(sizes)) if sizes else 0))
        blocks_file.write('  Smallest enumerated block size: %s\n' %
                          (min(enumeratedSizes) if enumeratedSizes else 'none'))
        blocks_file.write('  Largest enumerated block size: %s\n' %
                          (max(enumeratedSizes) if enumeratedSizes else 'none'))
        blocks_file.write('  Average enumerated block size: %.1f\n' %
                          ((sum(enumeratedSizes) / len(enumeratedSizes)) if enumeratedSizes else 0))
        blocks_file.write('  Largest optimal block size: %s\n' %
                          (max(optimalSizes) if optimalSizes else 'none'))
        blocks_file.write('  Largest improved block size: %s\n' %
                          (max(improvedSizes) if improvedSizes else 'none'))
        blocks_file.write('  Smallest timed out block size: %s\n' %
                          (min(timedOutSizes) if timedOutSizes else 'none'))
        blocks_file.write('  Average optimal solution time: %d ms\n' %
                          ((sum(optimalTimes) / len(optimalTimes)) if optimalTimes else 0))

    # # Write peak pressure stats
    # with open(os.path.join(statsFolder, args.regp), 'w') as regp_file:
    #   for benchName in stats:
    #     regp_file.write('Benchmark %s:\n' % benchName)
    #     regpressure = stats[benchName]['regpressure']
    #     benchRegp = {}
    #     for functionName in regpressure:
    #       functionRegp = {}
    #       regp_file.write('  Function %s:\n' % functionName)
    #       for blockName in regpressure[functionName]:
    #         # Note: In here you could write the BB name...
    #         # regp_file.write('    Basic Block %s:\n' % blockName)
    #         for setIndex, perp in regpressure[functionName][blockName]:
    #           # NOTE: In here you could write the pressure per BB...
    #           # regp_file.write('      %2d %5d\n' % (setIndex, perp))
    #           if not setIndex in functionRegp: functionRegp[setIndex] = 0
    #           functionRegp[setIndex] += perp
    #       for setIndex in sorted(functionRegp.keys()):
    #         regp_file.write('    %2d %5d\n' % (setIndex, functionRegp[setIndex]))
    #         if not setIndex in benchRegp: benchRegp[setIndex] = 0
    #         benchRegp[setIndex] += functionRegp[setIndex]
    #     regp_file.write('Summary for Benchmark %s:\n' % benchName)
    #     for setIndex in sorted(benchRegp.keys()):
    #       regp_file.write('  %2d %5d\n' % (setIndex, benchRegp[setIndex]))


    # write regp csv file
    with open(os.path.join(statsFolder, args.regp + ".mf.csv"), 'w') as regp_file:
        with open(os.path.join(statsFolder, args.regp + ".bb.csv"), 'w') as regpBBFile:
            regp_file.write('Benchmark,Function\n')
            regpBBFile.write('Benchmark,Function,BB#\n')
            for benchName in stats:
                spills = stats[benchName]['spills']
                regp_file.write('\n%s,' % benchName)
                regpBBFile.write('\n%s,' % benchName)
                regpressure = stats[benchName]['regpressure']
                benchRegp = {}
                counter = 0
                for functionName in regpressure:
                    if counter > 0:
                        regp_file.write('\n,')
                        regpBBFile.write('\n,')
                    counter += 1
                    functionRegp = {}
                    regp_file.write('%s,' % functionName)
                    regpBBFile.write('%s,' % functionName)
                    bbCounter = 0
                    for blockName in sorted(regpressure[functionName].keys()):
                        # Check to see that the block satisfies the constraints. If not, move on to the next iteration.
                        maxSetPerp = 0
                        for setIndex, perp in regpressure[functionName][blockName]:
                            maxSetPerp = max(maxSetPerp, perp)
                        if maxSetPerp < args.minperp:
                            # print("DAG %s:%d MaxSetPerp %d does not meet minimum PERP requirement %d. Skipping." %
                            #   (functionName, blockName, maxSetPerp, args.minperp))
                            continue
                        dagName = "%s:%d" % (functionName, blockName)
                        if not dagSizesPerBenchmark[benchName] is None and dagSizesPerBenchmark[benchName][dagName] < args.mininstcnt:
                            # print("DAG %s:%d InstCnt %d does not meet minimum InstCnt requirement %d. Skipping."
                            #   % (functionName, blockName,
                            #      dagSizesPerBenchmark[benchName]["%s:%d" % (functionName, blockName)],
                            #      args.mininstcnt))
                            continue
                        # print("DAG %s InstCnt %d MaxPERP %d meets contraints MinInstCnt %d MinPERP %d." %
                        #   (dagName,
                        #   dagSizesPerBenchmark[benchName][dagName],
                        #   maxSetPerp, args.mininstcnt, args.minperp))
                        if bbCounter > 0: regpBBFile.write('\n,,')
                        bbCounter += 1
                        regpBBFile.write('%s,' % blockName)
                        for setIndex, perp in regpressure[functionName][blockName]:
                            regpBBFile.write('%d,' % perp)
                            if not setIndex in functionRegp: functionRegp[setIndex] = 0
                            functionRegp[setIndex] += perp
                        regpBBFile.write('%d' % spills[functionName])
                    for setIndex in sorted(functionRegp.keys()):
                        regp_file.write('%d,' % functionRegp[setIndex])
                        if not setIndex in benchRegp: benchRegp[setIndex] = 0
                        benchRegp[setIndex] += functionRegp[setIndex]
                    regp_file.write('%d' % spills[functionName])

    # # write SLIL
    # with open(os.path.join(statsFolder, args.slil), 'w') as slilFile:
    #   for benchName in stats:
    #     benchmarkSlil = {}
    #     slilStats = stats[benchName]['slil']
    #     slilFile.write('Benchmark %s:\n' % benchName)
    #     for functionName in slilStats:
    #       functionSlil = {}
    #       functionStats = slilStats[functionName]
    #       slilFile.write('  Function %s:\n' % functionName)
    #       for blockNum in sorted(functionStats.keys()):
    #         for registerType in functionStats[blockNum]:
    #           if not registerType in functionSlil:
    #             functionSlil[registerType] = 0
    #           functionSlil[registerType] += functionStats[blockNum][registerType]
    #       for registerType in functionSlil:
    #         slilFile.write('    %2d %5d\n' % (registerType, functionSlil[registerType]))
    #         if not registerType in benchmarkSlil:
    #           benchmarkSlil[registerType] = 0
    #         benchmarkSlil[registerType] += functionSlil[registerType]
    #     slilFile.write('Summary for Benchmark %s:\n' % benchName)
    #     for registerType in sorted(benchmarkSlil.keys()):
    #       slilFile.write('  %2d %5d\n' % (registerType, functionSlil[registerType]))
    #     # slilFile.write('Total for benchmark %s: %d\n' % (benchName, benchmarkSlil))

    with open(os.path.join(statsFolder, args.slil + '.mf.csv'), 'w') as slilCsvFile:
        with open(os.path.join(statsFolder, args.slil + '.bb.csv'), 'w') as slilBBCsvFile:
            slilCsvFile.write('Benchmark,Function\n')
            slilBBCsvFile.write('Benchmark,Function,BB#\n')
            for benchName in stats:
                spills = stats[benchName]['spills']
                slilCsvFile.write('\n%s,' % benchName)
                slilBBCsvFile.write('\n%s,' % benchName)
                slilStats = stats[benchName]['slil']
                counter = 0
                for functionName in slilStats:
                    functionStats = {}
                    if counter > 0:
                        slilCsvFile.write('\n,')
                        slilBBCsvFile.write('\n,')
                    counter += 1
                    slilCsvFile.write('%s,' % functionName)
                    slilBBCsvFile.write('%s,' % functionName)
                    bbCounter = 0
                    for blockNum in sorted(slilStats[functionName].keys()):
                        # Check to see that the block satisfies the constraints. If not, move on to the next iteration.
                        maxSetPerp = 0
                        regpressure = stats[benchName]['regpressure']
                        for setIndex, perp in regpressure[functionName][blockNum]:
                            maxSetPerp = max(maxSetPerp, perp)
                        if maxSetPerp < args.minperp:
                            # print("DAG %s:%d MaxSetPerp %d does not meet minimum PERP requirement %d. Skipping." %
                            #   (functionName, blockNum, maxSetPerp, args.minperp))
                            continue
                        dagName = "%s:%d" % (functionName, blockNum)
                        if not dagSizesPerBenchmark[benchName] is None and dagSizesPerBenchmark[benchName][dagName] < args.mininstcnt:
                            # print("DAG %s:%d InstCnt %d does not meet minimum InstCnt requirement %d. Skipping."
                            #   % (functionName, blockNum,
                            #      dagSizesPerBenchmark[benchName]["%s:%d" % (functionName, blockNum)],
                            #      args.mininstcnt))
                            continue
                        # print("DAG %s InstCnt %d MaxPERP %d meets contraints MinInstCnt %d MinPERP %d." %
                        #   (dagName,
                        #   dagSizesPerBenchmark[benchName][dagName],
                        #   maxSetPerp, args.mininstcnt, args.minperp))
                        if bbCounter > 0:
                            slilBBCsvFile.write('\n,,')
                        bbCounter += 1
                        slilBBCsvFile.write('%d,' % blockNum)
                        for registerType in sorted(slilStats[functionName][blockNum].keys()):
                            slilBBCsvFile.write('%d,' % slilStats[functionName][blockNum][registerType])
                            if not registerType in functionStats:
                                functionStats[registerType] = 0
                            functionStats[registerType] += slilStats[functionName][blockNum][registerType]
                        slilBBCsvFile.write('%d' % spills[functionName])
                    if len(functionStats) > 0:
                        for registerType in functionStats:
                            slilCsvFile.write('%d,' % functionStats[registerType])
                        slilCsvFile.write('%d' % spills[functionName])

def calculateDagSizes(output):
    dagSizes = None
    matches = BLOCK_NAME_AND_SIZE_REGEX.findall(output)
    if len(matches) > 0:
        dagSizes = {}
        for dagName, instCountString in matches:
            dagSizes[dagName] = int(instCountString)
    return dagSizes

def calculateSLIL(output):
    stats = {}
    for dagName, registerType, setName, sumOfLiveIntervalLengths in SLIL_HEURISTIC_REGEX.findall(output):
        functionName = dagName.split(':')[0]
        blockNum = int(dagName.split(':')[1])
        if not functionName in stats:
            stats[functionName] = {}
        if not blockNum in stats[functionName]:
            stats[functionName][blockNum] = {}
        registerTypeAsInt = int(registerType)
        stats[functionName][blockNum][registerTypeAsInt] = int(sumOfLiveIntervalLengths)
    return stats

def calculatePeakPressureStats(output):
    stats = {}
    for dagName, setIndexString, setName, peakSetPressureString, limitString in BLOCK_PEAK_REG_PRESSURE_REGEX.findall(output):
        dagSplit = dagName.split(':')
        functionName = dagSplit[0]
        blockName = int(dagSplit[1])
        if not functionName in stats: stats[functionName] = {}
        # Each function gets a dict from blockName to tuple(setIndex, PERP)
        if not blockName in stats[functionName]: stats[functionName][blockName] = []
        peakExcessPressure = int(peakSetPressureString) - int(limitString)
        if peakExcessPressure < 0: peakExcessPressure = 0
        stats[functionName][blockName].append((int(setIndexString), peakExcessPressure))
    return stats

def calculateBlockStats(output):
    blocks = output.split('Opt Scheduling **********')[1:]
    stats = []
    for index, block in enumerate(blocks):
        lines = [line[6:] for line in block.split('\n') if line.startswith('INFO:')]
        block = '\n'.join(lines)

        try:
            name, size = BLOCK_NAME_AND_SIZE_REGEX.findall(block)[0]

            failed = (BLOCK_LIST_FAILED_REGEX.findall(block) != []) or (BLOCK_FAILED_REGEX.findall(block) != [])
            if failed:
                timeTaken = 0
                isEnumerated = isOptimal = False
                listCost = improvement = 0
            else:
                start_time = int(BLOCK_START_TIME_REGEX.findall(block)[0])
                end_time = int(BLOCK_END_TIME_REGEX.findall(block)[0])
                timeTaken = end_time - start_time

                listCost = int(BLOCK_COST_REGEX.findall(block)[0]) + int(BLOCK_COST_LOWER_BOUND_REGEX.search(block).group(1))
                isEnumerated = BLOCK_NOT_ENUMERATED_REGEX.findall(block) == []
                if isEnumerated:
                    isOptimal = bool(BLOCK_ENUMERATED_OPTIMAL_REGEX.findall(block))
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
              'improvement': improvement
            })
        except:
            print '  WARNING: Could not parse block #%d:' % (index + 1)
            print "Unexpected error:", sys.exc_info()[0]
            for line in blocks[index].split('\n')[1:-1][:10]:
                print '   ', line
            # raise

    return stats

def calculateSpills(output):
    spills = {}
    for functionName, spillCountString in SPILLS_REGEX.findall(output):
        spills[functionName] = int(spillCountString)
    return spills

"""
Defining this function makes it easier to parse files.
"""
def getBenchmarkResult(output):
    return {
      'time': int(TIMES_REGEX.findall(output)[0]),
      'spills': calculateSpills(output),
      'blocks': calculateBlockStats(output),
      'regpressure': calculatePeakPressureStats(output),
      'slil': calculateSLIL(output)
    }

def runBenchmarks(benchmarks, config):
    results = {}
    dagSizesPerBenchmark = {}
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    for fileName in os.listdir(LOG_FOLDER):
        os.remove(os.path.join(LOG_FOLDER, fileName))

    for bench in benchmarks:
        print 'Running', bench
        try:
            p = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE)
            p.stdin.write("source shrc" + "\n");
            p.stdin.write(COMMAND % (config, bench) + "\n")
            p.stdin.write("runspec --loose -size=ref -iterations=1 -config=%s --tune=base -r 1 -I -a scrub %s" % (config,bench))
            p.stdin.close()
            output = p.stdout.read()
            logFilePath = os.path.join(LOG_FOLDER, bench + ".log")
            with open(logFilePath, 'w') as logFile:
                logFile.write(output)
            print("Wrote log file of benchmark %s to log file %s." % (bench, logFilePath))

        except subprocess.CalledProcessError as e:
            print '  WARNING: Benchmark command failed: %s.' % e
        else:
            results[bench] = getBenchmarkResult(output)
            dagSizesPerBenchmark[bench] = calculateDagSizes(output)

    return results, dagSizesPerBenchmark


def main(args):
    # Select benchmarks.
    if args.bench == 'ALL':
        benchmarks = ALL_BENCHMARKS
    elif args.bench == 'INT':
        benchmarks = INT_BENCHMARKS
    elif args.bench == 'FP':
        benchmarks = FP_BENCHMARKS
    else:
        benchmarks = args.bench.split(',')
        for bench in benchmarks:
            if bench not in ALL_BENCHMARKS:
                print 'WARNING: Unknown benchmark specified: "%s".' % bench

    print('Using config file: %s' % args.config)
    print('Specified benchmarks: %s' % args.bench)

    results = {}
    dagSizesPerBenchmark = {}
    # Parse previous run instead of rebuilding.
    if args.readlogs is not None:
        print("Reading log files of previous run instead of building benchmarks.")
        if not os.path.isdir(args.readlogs):
            print("ERROR: --readlogs=%s is not a valid folder! Quitting runspec wrapper." % args.readlogs)
        else:
            if args.nodirwalk:
                for benchName in benchmarks:
                    logFilePath = os.path.join(args.readlogs, benchName + ".log")
                    if not os.path.isfile(logFilePath): continue
                    print("Parsing log file %s" % logFilePath)
                    output = None
                    with open(logFilePath) as logFile:
                        output = logFile.read()
                    results[benchName] = getBenchmarkResult(output)
                    dagSizesPerBenchmark[benchName] = calculateDagSizes(output)
            else:
                for filename in os.listdir(args.readlogs):
                    print("Parsing log file %s" % filename)
                    benchName = filename.split(".")[0]
                    logFilePath = os.path.join(args.readlogs, filename)
                    output = None
                    with open(logFilePath) as logFile:
                        output = logFile.read()
                    results[benchName] = getBenchmarkResult(output)
                    dagSizesPerBenchmark[benchName] = calculateDagSizes(output)

    # Run the benchmarks and collect results.
    elif args.opt is not None:

        # Temporarily adjust the INI file.
        with open(args.ini) as ini_file: ini = ini_file.read()
        new_ini = SETTING_REGEX.sub('USE_OPT_SCHED ' + args.opt, ini)
        with open(args.ini, 'w') as ini_file: ini_file.write(new_ini)
        try:
            results, dagSizesPerBenchmark = runBenchmarks(banchmarks, args.config)
        finally:
            with open(args.ini, 'w') as ini_file: ini_file.write(ini)
    else:
        results, dagSizesPerBenchmark = runBenchmarks(benchmarks, args.config)

    # Write out the results.
    writeStats(results, args, dagSizesPerBenchmark)


if __name__ == '__main__':
    parser = optparse.OptionParser(
        description='Wrapper around runspec for collecting spill counts.')
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
    parser.add_option('-i', '--ini',
                      metavar='filepath',
                      default='../OptSchedCfg/sched.ini',
                      help='Where to find sched.ini (%default).')
    parser.add_option('-o', '--opt',
                      metavar='YES|NO|HOTONLY',
                      default=None,
                      choices=['YES', 'NO', 'HOTONLY'],
                      help='Override the USE_OPT_SCHED setting in sched.ini.')
    parser.add_option('-b', '--bench',
                      metavar='ALL|INT|FP|name1,name2...',
                      default='ALL',
                      help='Which benchmarks to run.')
    parser.add_option('-r', '--regp',
                      metavar='filepath',
                      default='regp.dat',
                      help='Where to write the reg pressure stats (%default).')
    parser.add_option('-S', '--slil',
                      metavar='filepath',
                      default='slil.dat',
                      help='Where to write the SLIL stats (%default).')
    parser.add_option('-c', '--config',
                     metavar='spec_config_file.cfg',
                     default='Intel_llvm_3.3.cfg',
                     help='Choose SPEC config file (%default).')
    parser.add_option('--readlogs',
                     metavar='log_file_folder/',
                     default=None,
                     help='Specify folder of logs from previous run to gather statistics from log files instead of rebuilding.')
    parser.add_option('--mininstcnt',
                     metavar='number',
                     default=0,
                     type=int,
                     help='Minimum basic block instruction count to write PERP and SLIL stats for.')
    parser.add_option('--minperp',
                     metavar='number',
                     default=0,
                     type=int,
                     help='Minimum PERP count to write PERP and SLIL stats for.')
    parser.add_option('--nodirwalk', action='store_true')
    main(parser.parse_args()[0])
