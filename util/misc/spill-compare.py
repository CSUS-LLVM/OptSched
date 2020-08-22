#!/usr/bin/python3
# Compare two log files using the OptSched scheduler with simulate register
# allocation enabled. Find instances where a reduction in cost does not
# correspond with a reduction in spills.

import sys
import re

RE_REGION_DELIMITER = re.compile(r'INFO: \*{4,}? Opt Scheduling \*{4,}?')

RE_REGION_COST_LOWER_BOUND = re.compile(r'INFO: Lower bound of cost before scheduling: (\d+)')
RE_REGION_COST_BEST = re.compile(r"INFO: Best schedule for DAG (.*) has cost (\d+) and length (\d+). The schedule is (.*) \(Time")
RE_REGION_COST_HEURISTIC = re.compile(r"INFO: The list schedule is of length (\d+) and spill cost (\d+). Tot cost = (\d+) \(Time")
RE_REGION_SPILLS_BEST = re.compile(r"INFO: OPT_SCHED LOCAL RA: DAG Name: (\S+) Number of spills: (\d+) \(Time")
RE_REGION_SPILLS_HEURISTIC= re.compile(r"INFO: OPT_SCHED LOCAL RA: DAG Name: (.*) \*\*\*heuristic_schedule\*\*\* Number of spills: (\d+) \(Time")

regions = {}
totalBlocks = 0
totalMismatches = 0
lowestLength = sys.maxsize
smallestFoundRegion = ''
foundRegion = False


with open(str(sys.argv[1])) as logfile:
    log1 = logfile.read()
    blocks = [block for block in RE_REGION_DELIMITER.split(log1) if RE_REGION_COST_BEST.search(block)]
    for block in blocks:
        if not RE_REGION_COST_LOWER_BOUND.search(block):
            print("WARNING: Block does not have a logged lower bound.", out=sys.stderr)

        totalBlocks += 1

        lowerBound = int(RE_REGION_COST_LOWER_BOUND.search(block).group(1))
        regionCostMatchB = RE_REGION_COST_BEST.findall(block)
        regionName = regionCostMatchB[0][0]
        regionCostBest = int(regionCostMatchB[0][1])
        regionLengthBest = int(regionCostMatchB[0][2])

        if (len(RE_REGION_SPILLS_BEST.findall(block)) == 0):
            print(regionName)

        regionCostMatchH = RE_REGION_COST_HEURISTIC.findall(block)
        regionCostHeuristic = int(regionCostMatchH[0][2])

        regionSpillsMatchB = RE_REGION_SPILLS_BEST.findall(block)
        regionSpillsBest = int(regionSpillsMatchB[0][1])

        regionSpillsMatchH = RE_REGION_SPILLS_HEURISTIC.findall(block)
        regionSpillsHeuristic = int(regionSpillsMatchH[0][1])

        if (regionCostBest < regionCostHeuristic and regionSpillsBest > regionSpillsHeuristic):
            totalMismatches+=1
            print("Found Region: "  + regionName + " With Length: " + str(regionLengthBest))
            print("Best Cost: " + str(regionCostBest) + " Heuristic Cost: " + str(regionCostHeuristic))
            print("Best Cost (Absolute): " + (lowerBound + regionCostBest))
            print("Best Spills: " + str(regionSpillsBest) + " Heurisitc Spills: " + str(regionSpillsHeuristic))
            if (regionLengthBest < lowestLength):
                foundRegion = True
                smallestFoundRegion = regionName
                lowestLength = regionLengthBest

    if (foundRegion):
        print("Smallest region with mismatch is: " + str(smallestFoundRegion) + " with length " + str(lowestLength))

    print("Processed " + str(totalBlocks) + " blocks")
    print("Found " + str(totalMismatches) + " mismatches")
