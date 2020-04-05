#!/usr/bin/python3
# Compare two log files using the OptSched scheduler with simulate register
# allocation enabled. Find instances where a reduction in cost does not
# correspond with a reduction in spills.

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readlogs import *

regions = {}
totalBlocks = 0
totalMismatches = 0
lowestLength = sys.maxsize
smallestFoundRegion = ''
foundRegion = False


with open(str(sys.argv[1])) as logfile:
    log1 = logfile.read()
    blocks = [block for block in parse_blocks(log1) if 'BestResult' in block]
    for block in blocks:
        if not 'CostLowerBound' not in block:
            print("WARNING: Block does not have a logged lower bound. Skipping block: " + block,
                out=sys.stderr)
            continue

        totalBlocks += 1

        lowerBound = block['CostLowerBound']['cost']
        bestCostInfo = block['BestResult']
        regionName = bestCostInfo['name']
        regionCostBest = bestCostInfo['cost']
        regionLengthBest = bestCostInfo['length']

        if 'BestLocalRegAllocSimulation' not in block:
            print(regionName)

        regionCostHeuristic = block['HeuristicResult']['spill_cost']
        regionSpillsBest = block['BestLocalRegAllocSimulation']['num_spills']
        regionSpillsHeuristic = block['HeuristicLocalRegAllocSimulation']['num_spills']

        if regionCostBest < regionCostHeuristic and regionSpillsBest > regionSpillsHeuristic:
            totalMismatches+=1
            print("Found Region: "  + regionName + " With Length: " + str(regionLengthBest))
            print("Best Cost: " + str(regionCostBest) + " Heuristic Cost: " + str(regionCostHeuristic))
            print("Best Cost (Absolute): " + (lowerBound + regionCostBest))
            print("Best Spills: " + str(regionSpillsBest) + " Heurisitc Spills: " + str(regionSpillsHeuristic))
            if regionLengthBest < lowestLength:
                foundRegion = True
                smallestFoundRegion = regionName
                lowestLength = regionLengthBest

    if (foundRegion):
        print("Smallest region with mismatch is: " + str(smallestFoundRegion) + " with length " + str(lowestLength))

    print("Processed " + str(totalBlocks) + " blocks")
    print("Found " + str(totalMismatches) + " mismatches")
