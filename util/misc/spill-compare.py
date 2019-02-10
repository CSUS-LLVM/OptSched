#!/usr/bin/python3
# Compare two log files using the OptSched scheduler with simulate register
# allocation enabled. Find instances where a reduction in cost does not
# correspond with a reduction in spills.

import sys
import re

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
    blocks = log1.split("INFO: ********** Opt Scheduling **********")
    for block in blocks:
        # Assume that if this matches the block is valid.
        if (len(RE_REGION_COST_BEST.findall(block)) == 0):
            continue;

        totalBlocks+=1

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
            print("Best Spills: " + str(regionSpillsBest) + " Heurisitc Spills: " + str(regionSpillsHeuristic))
            if (regionLengthBest < lowestLength):
                foundRegion = True
                smallestFoundRegion = regionName
                lowestLength = regionLengthBest

    if (foundRegion):
        print("Smallest region with mismatch is: " + str(smallestFoundRegion) + " with length " + str(lowestLength))

    print("Processed " + str(totalBlocks) + " blocks")
    print("Found " + str(totalMismatches) + " mismatches")
