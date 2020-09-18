import re
import mmap
import optparse
import os
import json

parser = optparse.OptionParser(
    description='Wrapper around runspec for collecting spill counts.')
parser.add_option('-b', '--bruteforce',
                  metavar='filepath',
                  default=None,
                  help='Log file of brute force compiler.')
parser.add_option('-d', '--dynamic',
                  metavar='filepath',
                  default=None,
                  help='Log file of dynamic compiler.')

args = parser.parse_args()[0]

bruteForceFile =  args.bruteforce
bbFile =          args.dynamic

if not os.path.isfile(bruteForceFile):
    raise Error("Please specify a valid brute force log file.")
if not os.path.isfile(bbFile):
    raise Error("Please specify a valid dynamic log file.")

results = {}

errorCount = 0
equalCount = 0
improvementCount = 0

# Gather results from log files (assumed to be just 1 log file per build)
with open(bruteForceFile) as bff:
    bffm = mmap.mmap(bff.fileno(), 0, access=mmap.ACCESS_READ)
    dagResults = {}
    for match in re.finditer(r'EVENT: ({"event_id": "StaticLowerBoundDebugInfo".*)', bffm):
        info = json.loads(match.group(1))
        dagResults[info['name']] = int(info['spill_cost_lb'])
    bffm.close()
    results['bf'] = dagResults

with open(bbFile) as bbf:
    bbfm = mmap.mmap(bbf.fileno(), 0, access=mmap.ACCESS_READ)
    dagResults = {}
    for match in re.finditer(r'EVENT: ({"event_id": "StaticLowerBoundDebugInfo".*)', bffm):
        info = json.loads(match.group(1))
        dagResults[info['name']] = int(info['spill_cost_lb'])
    bbfm.close()
    results['bb'] = dagResults

#analyze results
#
for dagName in results['bf']:
    bfLowerBound = results['bf'][dagName]
    if not dagName in results['bb']: continue
    bbLowerBound = results['bb'][dagName]
    if bfLowerBound < bbLowerBound:
        print("Improvement: oldLB %d newLB %d dag %s" % (bfLowerBound, bbLowerBound, dagName))
        improvementCount += 1
    elif bfLowerBound == bbLowerBound:
        print("Equal: oldLB %d newLB %d dag %s" % (bfLowerBound, bbLowerBound, dagName))
        equalCount += 1
    else:
        print("Error: oldLB %d newLB %d dag %s" % (bfLowerBound, bbLowerBound, dagName))
        errorCount += 1
print("Improved blocks: %d"% improvementCount)
print("Equal blocks:    %d"% equalCount)
print("Errors:          %d"% errorCount)
