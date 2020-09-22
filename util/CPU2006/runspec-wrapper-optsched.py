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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readlogs import *

## Configuration

# Dictionary for benchmark selection. Maps benchmark names and categories
# to lists of specific benchmarks.
specVersions = {
    'CPU2006' : {
        'BUILD_COMMAND' : 'runspec --loose -size=ref -iterations=1 -config=%s --tune=base -r 1 -I -a build %s',
        'SCRUB_COMMAND' : 'runspec --loose -size=ref -iterations=1 -config=%s --tune=base -r 1 -I -a scrub %s',
        'benchDict' : {
            # Add all benchmarks individually.
            'perlbench': ['perlbench'],
            'bzip2' : ['bzip2'],
            'gcc' : ['gcc'],
            'mcf' : ['mcf'],
            'gobmk' : ['gobmk'],
            'hmmer' : ['hmmer'],
            'sjeng' : ['sjeng'],
            'libquantum' : ['libquantum'],
            'h264ref' : ['h264ref'],
            'omnetpp' : ['omnetpp'],
            'astar' : ['astar'],
            'xalancbmk' : ['xalancbmk'],
            'bwaves' : ['bwaves'],
            'gamess' : ['gamess'],
            'milc' : ['milc'],
            'zeusmp' : ['zeusmp'],
            'gromacs' : ['gromacs'],
            'cactus' : ['cactus'],
            'leslie' : ['leslie'],
            'namd' : ['namd'],
            'dealII' : ['dealII'],
            'soplex' : ['soplex'],
            'povray' : ['povray'],
            'calculix' : ['calculix'],
            'Gems' : ['Gems'],
            'tonto' : ['tonto'],
            'lbm' : ['lbm'],
            'wrf' : ['wrf'],

            # Add ALL benchmark group.
            'ALL' : [
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
            ],

            # Add INT benchmark group.
            'INT' : [
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
            ],

            # Add FP benchmark group.
            'FP' : [
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
            ],

            # Add C benchmark group.
            'C' : [
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
            ],

            # Add C++ benchmark group.
            'C++' : [
                'namd',
                'dealII',
                'soplex',
                'povray',
                'omnetpp',
                'astar',
                'xalancbmk'
            ],

            # Add the FORTRAN benchmark group.
            'FORTRAN' : [
                'bwaves',
                'gamess',
                'zeusmp',
                'leslie',
                'Gems',
                'tonto'
            ],

            # Add the MIXED language benchmark group.
            'MIXED' : [
                'gromacs',
                'cactus',
                'calculix',
                'wrf'
            ],

            # The FP benchmarks without FORTRAN. (ie no dragonegg/flang)
            'FP_NO_F' : [
                'milc',
                'namd',
                'dealII',
                'soplex',
                'povray',
                'lbm',
                'sphinx'
            ]
        }
    },

    'CPU2017' : {
        'BUILD_COMMAND' : 'runcpu --tune=base --config=%s -a build %s',
        'SCRUB_COMMAND' : 'runcpu --tune=base --config=%s -a scrub %s',
        'benchDict' : {
            # Add all benchmarks individually.
            '500.perlbench_r' : ['500.perlbench_r'],
            '502.gcc_r' : ['502.gcc_r'],
            '503.bwaves_r' : ['503.bwaves_r'],
            '505.mcf_r' : ['505.mcf_r'],
            '507.cactuBSSN_r' : ['507.cactuBSSN_r'],
            '508.namd_r' : ['508.namd_r'],
            '510.parest_r' : ['510.parest_r'],
            '511.povray_r' : ['511.povray_r'],
            '519.lbm_r' : ['519.lbm_r'],
            '520.omnetpp_r' : ['520.omnetpp_r'],
            '523.xalancbmk_r' : ['523.xalancbmk_r'],
            '525.x264_r' : ['525.x264_r'],
            '531.deepsjeng_r' : ['531.deepsjeng_r'],
            '541.leela_r' : ['541.leela_r'],
            '548.exchange2_r' : ['548.exchange2_r'],
            '557.xz_r' : ['557.xz_r'],
            '600.perlbench_s' : ['600.perlbench_s'],
            '602.gcc_s' : ['602.gcc_s'],
            '603.bwaves_s' : ['603.bwaves_s'],
            '605.mcf_s' : ['605.mcf_s'],
            '607.cactuBSSN_s' : ['607.cactuBSSN_s'],
            '619.lbm_s' : ['619.lbm_s'],
            '620.omnetpp_s' : ['620.omnetpp_s'],
            '621.wrf_s' : ['621.wrf_s'],
            '623.xalancbmk_s' : ['623.xalancbmk_s'],
            '627.cam4_s' : ['627.cam4_s'],
            '625.x264_s' : ['625.x264_s'],
            '628.pop2_s' : ['628.pop2_s'],
            '631.deepsjeng_s' : ['631.deepsjeng_s'],
            '638.imagick_s' : ['638.imagick_s'],
            '641.leela_s' : ['641.leela_s'],
            '644.nab_s' : ['644.nab_s'],
            '648.exchange2_s' : ['648.exchange2_s'],
            '649.fotonik3d_s' : ['649.fotonik3d_s'],
            '654.roms_s' : ['654.roms_s'],
            '657.xz_s' : ['657.xz_s'],

            # Add TEST benchmark group.
            'TEST' : [
                '603.bwaves_s',
                '619.lbm_s',
                '607.cactuBSSN_s'
            ],

            # Add ALL benchmark group.
            'ALL' : [
                '500.perlbench_r',
                '600.perlbench_s',
                '502.gcc_r',
                '602.gcc_s',
                '505.mcf_r',
                '605.mcf_s',
                '520.omnetpp_r',
                '620.omnetpp_s',
                '523.xalancbmk_r',
                '623.xalancbmk_s',
                '525.x264_r',
                '625.x264_s',
                '531.deepsjeng_r',
                '631.deepsjeng_s',
                '541.leela_r',
                '641.leela_s',
                '548.exchange2_r',
                '648.exchange2_s',
                '557.xz_r',
                '657.xz_s',
                '503.bwaves_r',
                '603.bwaves_s',
                '507.cactuBSSN_r',
                '607.cactuBSSN_s',
                '508.namd_r',
                '510.parest_r',
                '511.povray_r',
                '519.lbm_r',
                '619.lbm_s',
                '521.wrf_r',
                '621.wrf_s',
                '526.blender_r',
                '527.cam4_r',
                '627.cam4_s',
                '628.pop2_s',
                '538.imagick_r',
                '638.imagick_s',
                '544.nab_r',
                '644.nab_s',
                '549.fotonik3d_r',
                '649.fotonik3d_s',
                '554.roms_r',
                '654.roms_s'
            ],

            # Add INT benchmark group.
            'INT' : [
                '500.perlbench_r',
                '600.perlbench_s',
                '502.gcc_r',
                '602.gcc_s',
                '505.mcf_r',
                '605.mcf_s',
                '520.omnetpp_r',
                '620.omnetpp_s',
                '523.xalancbmk_r',
                '623.xalancbmk_s',
                '525.x264_r',
                '625.x264_s',
                '531.deepsjeng_r',
                '631.deepsjeng_s',
                '541.leela_r',
                '641.leela_s',
                '548.exchange2_r',
                '648.exchange2_s',
                '557.xz_r',
                '657.xz_s',
            ],

            #Add INT SPECrate
            'INT_RATE' : [
                '500.perlbench_r',
                '502.gcc_r',
                '505.mcf_r',
                '520.omnetpp_r',
                '523.xalancbmk_r',
                '525.x264_r',
                '531.deepsjeng_r',
                '541.leela_r',
                '548.exchange2_r',
                '557.xz_r',
            ],

            #Add INT SPECspeed
            'INT_SPEED' : [
                '600.perlbench_s',
                '602.gcc_s',
                '605.mcf_s',
                '620.omnetpp_s',
                '623.xalancbmk_s',
                '625.x264_s',
                '631.deepsjeng_s',
                '641.leela_s',
                '648.exchange2_s',
                '657.xz_s',
            ],

            # Add FP benchmark group.
            'FP' : [
                '503.bwaves_r',
                '603.bwaves_s',
                '507.cactuBSSN_r',
                '607.cactuBSSN_s',
                '508.namd_r',
                '510.parest_r',
                '511.povray_r',
                '519.lbm_r',
                '619.lbm_s',
                '521.wrf_r',
                '621.wrf_s',
                '526.blender_r',
                '527.cam4_r',
                '627.cam4_s',
                '628.pop2_s',
                '538.imagick_r',
                '638.imagick_s',
                '544.nab_r',
                '644.nab_s',
                '549.fotonik3d_r',
                '649.fotonik3d_s',
                '554.roms_r',
                '654.roms_s'
            ],

            # Add FP SPECrate
            'FP_RATE' : [
                '503.bwaves_r',
                '507.cactuBSSN_r',
                '508.namd_r',
                '510.parest_r',
                '511.povray_r',
                '519.lbm_r',
                '521.wrf_r',
                '526.blender_r',
                '527.cam4_r',
                '538.imagick_r',
                '544.nab_r',
                '549.fotonik3d_r',
                '554.roms_r',
            ],

            # Add FP SPECrate w/o the non-working wrf
            'FP_RATE_NWRF' : [
                '503.bwaves_r',
                '507.cactuBSSN_r',
                '508.namd_r',
                '510.parest_r',
                '511.povray_r',
                '519.lbm_r',
                '526.blender_r',
                '527.cam4_r',
                '538.imagick_r',
                '544.nab_r',
                '549.fotonik3d_r',
                '554.roms_r',
            ],

            # Add FP SPECspeed
            'FP_SPEED' : [
                '603.bwaves_s',
                '607.cactuBSSN_s',
                '619.lbm_s',
                '621.wrf_s',
                '627.cam4_s',
                '628.pop2_s',
                '638.imagick_s',
                '644.nab_s',
                '649.fotonik3d_s',
                '654.roms_s'
            ],

            # Add C benchmark group.
            'C' : [
                '500.perlbench_r',
                '600.perlbench_s',
                '502.gcc_r',
                '602.gcc_s',
                '505.mcf_r',
                '605.mcf_s',
                '525.x264_r',
                '625.x264_s',
                '557.xz_r',
                '657.xz_s',
                '519.lbm_r',
                '619.lbm_s',
                '538.imagick_r',
                '638.imagick_s',
                '544.nab_r',
                '644.nab_s',
            ],

            # Add C++ benchmark group.
            'C++' : [
                '520.omnetpp_r',
                '620.omnetpp_s',
                '523.xalancbmk_r',
                '623.xalancbmk_s',
                '531.deepsjeng_r',
                '631.deepsjeng_s',
                '541.leela_r',
                '641.leela_s',
                '508.namd_r',
                '510.parest_r',
            ],

            # Add the FORTRAN benchmark group.
            'FORTRAN' : [
                '548.exchange2_r',
                '648.exchange2_s',
                '503.bwaves_r',
                '603.bwaves_s',
                '549.fotonik3d_r',
                '649.fotonik3d_s',
                '554.roms_r',
                '654.roms_s'
            ],

            # Add the MIXED language benchmark group.
            'MIXED' : [
                '507.cactuBSSN_r',
                '607.cactuBSSN_s',
                '511.povray_r',
                '521.wrf_r',
                '621.wrf_s',
                '526.blender_r',
                '527.cam4_r',
                '627.cam4_s',
                '628.pop2_s'
            ],

            # The FP benchmarks without FORTRAN. (ie no dragonegg/flang)
            'FP_NO_F' : [
                '508.namd_r',
                '510.parest_r',
                '511.povray_r',
                '519.lbm_r',
                '619.lbm_s',
                '526.blender_r',
                '538.imagick_r',
                '638.imagick_s',
                '544.nab_r',
                '644.nab_s'
            ]
        }
    }
}


DETECT_COMMAND = 'if ! . shrc &> /dev/null; then  echo "NX_INSTALL";  elif runspec -h &> /dev/null; then echo "CPU2006"; elif runcpu -h &> /dev/null; then echo "CPU2017"; else echo "SPEC_AUTODETECT_ERROR"; fi'
LOG_DIR = 'logs/'

# Regular expressions.
SETTING_REGEX = re.compile(r'\bUSE_OPT_SCHED\b.*')
#SPILLS_REGEX = re.compile(r'Function: (.*?)\nEND FAST RA: Number of spills: (\d+)\n')
SPILLS_REGEX = re.compile(r'Function: (.*?)\nGREEDY RA: Number of spilled live ranges: (\d+)')
#SPILLS_REGEX = re.compile(r'Function: (.*?)\nTotal Simulated Spills: (\d+)')
SPILLS_WEIGHTED_REGEX = re.compile(r'SC in Function (.*?) (-?\d+)')
TIMES_REGEX = re.compile(r'(\d+) total seconds elapsed')

def writeStats(stats, spills, weighted, times, blocks, trackOptSchedSpills):
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
                spills = stats[benchName]['spills']['spills']
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

    # Write the weighted spill stats
    if weighted:
        with open(weighted, 'w') as weighted_file:
            totalWeightedSpills = 0
            for benchName in stats:
                totalWeightedSpillsPerBenchmark = 0
                weighted_file.write('%s:\n' % benchName)
                weightedSpills = stats[benchName]['spills']['weightedSpills']
                for functionName in weightedSpills:
                    weightedSpillCount = weightedSpills[functionName]
                    totalWeightedSpillsPerBenchmark += weightedSpillCount
                    weighted_file.write('      %5d %s\n' %
                                      (weightedSpillCount, functionName))
                weighted_file.write('  ---------\n')
                weighted_file.write('  Sum:%5d\n\n' % totalWeightedSpillsPerBenchmark)
                totalWeightedSpills += totalWeightedSpillsPerBenchmark
            weighted_file.write('------------\n')
            weighted_file.write('Total:%5d\n' % totalWeightedSpills)

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
            totalAcoImprovement = 0
            totalAcoPostImprovement = 0
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
                cost = 0
                acoImprovement = 0
                acoPostImprovement = 0
                improvement = 0
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
                        acoImprovement += block['acoImprovement']
                        acoPostImprovement += block['acoPostImprovement']
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
                blocks_file.write('  Aco cost: %d\n' %
                                  (cost - acoImprovement))
                blocks_file.write('  B&B cost: %d\n' %
                                  (cost - acoImprovement - improvement))
                fullImprovement = acoImprovement + improvement + acoPostImprovement
                blocks_file.write('  AcoPost cost: %d\n' %
                                  (cost - fullImprovement))
                blocks_file.write('  Cost improvement: %d (%.2f%%)\n' %
                                  (fullImprovement, (100 * fullImprovement / cost) if cost else 0))
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
                totalAcoImprovement += acoImprovement
                totalAcoPostImprovement += acoPostImprovement
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
            if trackOptSchedSpills and totalEnumerated > 0:
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
            blocks_file.write('  Aco cost: %d\n' %
                              (totalCost - totalAcoImprovement))
            blocks_file.write('  B&B cost: %d\n' %
                              (totalCost - totalAcoImprovement - totalImprovement))
            totalFullImprovement = totalAcoImprovement + totalImprovement + totalAcoPostImprovement
            blocks_file.write('  AcoPost cost: %d\n' %
                              (totalCost -totalFullImprovement))
            blocks_file.write('  Cost improvement: %d (%.2f%%)\n' %
                              (totalFullImprovement, (100 * totalFullImprovement / totalCost) if totalCost else 0))
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


def calculateBlockStats(output, trackOptSchedSpills, normalized):
    blocks = split_blocks(output)
    stats = []
    for index, block in enumerate(blocks):
        events = keep_only_first_event(parse_events(block))

        try:
            process_dag = events['ProcessDag']
            name = process_dag['name']
            size = process_dag['num_instructions']

            heuristic_failed = 'HeuristicSchedulerFailed' in events
            failed = heuristic_failed

            if failed:
                timeTaken = 0
                isEnumerated = isOptimal = False
                listCost = improvement = 0
            else:
                start_time = process_dag['time']
                end_time = events['ScheduleVerifiedSuccessfully']['time']
                timeTaken = end_time - start_time
                if 'LocalRegAllocSimulationChoice' in events and trackOptSchedSpills:
                    optSchedSpills = events['LocalRegAllocSimulationChoice']['num_spills']
                else:
                    optSchedSpills = 0

                costLowerBound = events['CostLowerBound']['cost'] if not normalized else 0
                listCost = costLowerBound + events['HeuristicResult']['cost']
                # The block is not enumerated if the list schedule is optimal or there is a zero
                # time limit for enumeration.
                isEnumerated = 'Enumerating' in events
                if isEnumerated:
                    isOptimal = 'DagSolvedOptimally' in events
                    """
                    Sometimes the OptScheduler doesn't print out cost improvement.
                    This happens when the scheduler determines that the list schedule is
                    already optimal, which means no further improvements can be made.

                    The rest of this if-block ensures that this tool doesn't crash.
                    If the improvement is not found, it is assumed to be 0.
                    """
                    if isOptimal:
                        improvement = events['DagSolvedOptimally']['cost_improvement']
                    elif 'DagTimedOut' in events:
                        improvement = events['DagTimedOut']['cost_improvement']
                    else:
                        improvement = 0
                else:
                    isOptimal = False
                    improvement = 0

            if 'ACO_sched_complete' in events:
                acoImprovement = events['AcoSchedComplete']['improvement']
            else:
                acoImprovement = 0
            if 'ACOpost_sched_complete' in events:
                acoPostImprovement = events['AcoPostSchedComplete']['improvement']
            else:
                acoPostImprovement = 0

            stats.append({
                'name': name,
                'size': int(size),
                'time': timeTaken,
                'success': not failed,
                'isEnumerated': isEnumerated,
                'isOptimal': isOptimal,
                'listCost': listCost,
                'improvement': improvement,
                'acoImprovement': acoImprovement,
                'acoPostImprovement': acoPostImprovement,
                'optSchedSpills': optSchedSpills
            })
        except Exception as e:
            print e
            print '  WARNING: Could not parse block #%d:' % (index + 1)
            print "Unexpected error:", sys.exc_info()[0]
            for line in blocks[index].split('\n')[1:-1][:10]:
                print '   ', line



    return stats


"""
calculateSpills(output)

Get the number of spills and weighted spills from an input

Input:
Log file or input from terminal that contains spilling
information from CPU2006.

Output:
Dictionary Variable
{
'spills', {'functionName', spills}
'weightedSpills, {'functionName', weightedSpillCount}
}
"""
def calculateSpills(output):
    spills = {} # This dictionary variable will be return as the result
    numOfSpills = {}
    weightedSpills = {}

    # Get and record the number of spills using a regular expression
    for functionName, spillCountString in SPILLS_REGEX.findall(output):
        numOfSpills[functionName] = int(spillCountString)

    # Get and record the number of weighted spills using a regular expression
    for functionName, weightedSpillCount in SPILLS_WEIGHTED_REGEX.findall(output):
        weightedSpills[functionName] = int(weightedSpillCount)

    # Insert the spills and weighted spills into the spills dictionary variable
    spills['spills'] = numOfSpills
    spills['weightedSpills'] = weightedSpills


    return spills


"""
Defining this function makes it easier to parse files.
"""


def getBenchmarkResult(output, trackOptSchedSpills, normalized):
    # Handle parsing log files that were not generated by runspec and have no time information.
    time = int(TIMES_REGEX.findall(output)[1]) if len(TIMES_REGEX.findall(output)) > 0 else -1
    return {
        'time': time,
        'spills': calculateSpills(output),
        'blocks': calculateBlockStats(output, trackOptSchedSpills, normalized),
    }

def detectSPECInstall():
    try:
        p = subprocess.Popen(['/bin/bash', '-c', DETECT_COMMAND], stdout=subprocess.PIPE)
        output = p.stdout.read().strip()
    except subprocess.CalledProcessError as e:
        print "FATAL ERROR in runspec wrapper.  Could not auto detect SPEC installation"
        sys.exit()

    if output == "SPEC_AUTODETECT_ERROR":
        print "Runspec-wrapper found the shrc script, but was unable to find either runspec or runcpu"
        sys.exit()

    if output == "NX_INSTALL":
        print "SPEC does not appear to be installed in this directory"
        sys.exit()

    return output

def runBenchmarks(benchmarks, testOutDir, shouldWriteLogs, config, trackOptSchedSpills, normalized):
    # Detect Install
    version = detectSPECInstall()
    BUILD_COMMAND = specVersions[version]['BUILD_COMMAND']
    SCRUB_COMMAND = specVersions[version]['SCRUB_COMMAND']

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
            results[bench] = getBenchmarkResult(output, trackOptSchedSpills, normalized)

            # Optionally write log files to results directory.
            if shouldWriteLogs is True:
                writeLogs(output, testOutDir, bench)

    return results

# Write log files for a benchmark to the results directory.


def writeLogs(output, testOutDir, bench):
    with open(os.path.join(testOutDir,  LOG_DIR + bench + '.log'), 'w') as log:
        log.write(output)


def main(args):
    # Parse a log file or multiple log files instead of running benchmark
    results = {}
    if args.logfile is not None:
        logfiles = [f for f in os.listdir(args.logfile) if os.path.isfile(os.path.join(args.logfile, f)) and f[-4:]=='.log']

        for log in logfiles:
            with open(os.path.join(args.logfile, f)) as log_file:
                output = log_file.read()
                results[log] = getBenchmarkResult(output, args.trackOptSchedSpills, args.normalized)

                spills = os.path.join(args.outdir, args.spills)
                weighted = os.path.join(args.outdir, args.weighted)
                times = os.path.join(args.outdir, args.times)
                blocks = os.path.join(args.outdir, args.blocks)

                # Write out the results from the logfile.
                writeStats(results, spills, weighted, times, blocks, args.trackOptSchedSpills)

        # Run the benchmarks and collect results.
    else:

        # Detect Install
        version = detectSPECInstall()
        benchDict = specVersions[version]['benchDict']
        print 'Using benchmark suite: "%s".' % version

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
            results = runBenchmarks(benchmarks, testOutDir, args.writelogs, args.config, args.trackOptSchedSpills, args.normalized)

            spills = os.path.join(testOutDir, args.spills)
            weighted = os.path.join(testOutDir, args.weighted)
            times = os.path.join(testOutDir, args.times)
            blocks = os.path.join(testOutDir, args.blocks)

            # Write out the results for this test.
            writeStats(results, spills, weighted, times, blocks, args.trackOptSchedSpills)


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
    parser.add_option('-u', '--weighted',
                      metavar='filepath',
                      default='weighted-spills.dat',
                      help='Where to write the weighted spill counts (%default).')
    parser.add_option('-k', '--blocks',
                      metavar='filepath',
                      default='blocks.dat',
                      help='Where to write the run block stats (%default).')
    parser.add_option('-c', '--config',
                      metavar='filepath',
                      default='default.cfg',
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
                      metavar='ALL|INT_SPEED|FP_SPEED|C|C++|FORTRAN|...',
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
    parser.add_option('-n', '--normalized',
                      action="store_true",
                      dest="normalized",
                      default=False,
                      help='Output normalized/relative costs to blocks.dat instead of absolute costs (%default).')

    main(parser.parse_args()[0])
