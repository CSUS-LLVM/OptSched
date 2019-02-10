# /bin/python3
# Run this script with a CPU2006 logfile as the only argument.
# When using RegAllocFast, find the total number of spills and the proportion of
# those spills that are added at region and block boundaries.

import re
import sys

RE_FUNC = re.compile('Function: (.*?)\n')
RE_TOTAL_SPILLS = re.compile('END FAST RA: Number of spills: (\d+)\n')
RE_CALL_BOUNDARY_STORES = re.compile('Call Boundary Stores in function: (\d+)\n')
RE_BLOCK_BOUNDARY_STORES = re.compile('Block Boundary Stores in function: (\d+)\n')
RE_LIVE_IN_LOADS = re.compile('Live-In Loads in function: (\d+)\n')

totalSpills = 0
totalCallBoundaryStores = 0
totalBlockBoundaryStores = 0
totalLiveInLoads = 0
totalFuncs = 0
#funcs = {}

if __name__ == '__main__':
    with open(sys.argv[1]) as inputLog:
        for line in inputLog.readlines():
            searchTotalSpills = RE_TOTAL_SPILLS.findall(line)
            searchCallBoundaryStores = RE_CALL_BOUNDARY_STORES.findall(line)
            searchBlockBoundaryStores = RE_BLOCK_BOUNDARY_STORES.findall(line)
            searchLiveInLoads = RE_LIVE_IN_LOADS.findall(line)
            # TDOD remove
            #searchFunc = RE_FUNC.findall(line)
            #if searchFunc != []:
            #    if searchFunc[0] in funcs:
            #        print(searchFunc[0] + 'Is a copy')
            #    else:
            #        funcs[searchFunc[0]] = 0
            if searchTotalSpills != []:
                totalSpills += int(searchTotalSpills[0])
                totalFuncs+=1
            elif searchCallBoundaryStores != []:
                totalCallBoundaryStores += int(searchCallBoundaryStores[0])
            elif searchBlockBoundaryStores != []:
                totalBlockBoundaryStores += int(searchBlockBoundaryStores[0])
            elif searchLiveInLoads != []:
                totalLiveInLoads += int(searchLiveInLoads[0])

    print("Total Spills: " + str(totalSpills))
    print("Total Call Boundary Stores: " + str(totalCallBoundaryStores))
    print("Total Block Boundary Stores: " + str(totalBlockBoundaryStores))
    print("Total Live-In Loads: " + str(totalLiveInLoads))
    print("Total funcs: " + str(totalFuncs))
