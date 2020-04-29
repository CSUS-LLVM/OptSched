import re
import mmap
import optparse
import os

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

regex = re.compile(r'Dag (.*?) (.*?) absolute cost (\d+?) time (\d+)')

results = {}

SUCCESS = "optimal"
TIMEOUT = "timeout"
FAILED = "failed"

staticErrorCount = 0
dynamicErrorCount = 0
goodCount = 0
# Gather results from log files (assumed to be just 1 log file per build)
with open(bruteForceFile) as bff:
    bffm = mmap.mmap(bff.fileno(), 0, access=mmap.ACCESS_READ)

    for match in regex.finditer(bffm):
        dagResult = {}
        dagResult['bf'] = {}
        dagResult['bf']['result'] = match.group(2)
        dagResult['bf']['cost'] = int(match.group(3))
        dagResult['bf']['time'] = int(match.group(4))
        results[match.group(1)] = dagResult

    bffm.close()

with open(bbFile) as bbf:
    bbfm = mmap.mmap(bbf.fileno(), 0, access=mmap.ACCESS_READ)
    for match in regex.finditer(bbfm):
        if not match.group(1) in results:
            results[match.group(1)] = {}
        results[match.group(1)]['bb'] = {}
        results[match.group(1)]['bb']['result'] = match.group(2)
        results[match.group(1)]['bb']['cost'] = int(match.group(3))
        results[match.group(1)]['bb']['time'] = int(match.group(4))
    bbfm.close()


#analyze results
for dagName in results:
    if not "bf" in results[dagName] or not "bb" in results[dagName]:
        if len(results[dagName]) > 0:
            staticErrorCount += 1
            print("StaticLBError: Found B&B results for one file but not the other")
            for key in results[dagName]:
                print("  %s: Dag %s %s cost %d time %d" % (key, dagName, results[dagName][key]['result'], results[dagName][key]['cost'], results[dagName][key]['time']))
        continue
    bfCost = results[dagName]['bf']['cost']
    bbCost = results[dagName]['bb']['cost']
    bfResult = results[dagName]['bf']['result']
    bbResult = results[dagName]['bb']['result']
    # Case 1: both success -> must be same cost
    if bfResult == SUCCESS and bbResult == SUCCESS:
        if bbCost != bfCost:
            dynamicErrorCount += 1
            print("DynamicLBError: Dag %s: both implementations optimal, but brute force cost (%d) is different from dynamic cost (%d)" %(dagName, bfCost, bbCost))
        else:
            goodCount += 1
            print("Good: Dag %s: both implementations solved optimally, and both costs match" % dagName)
    # Case 2: one timeout and other success -> timeout cost shouldn't be better
    elif bfResult == SUCCESS and bbResult == TIMEOUT:
        if bbCost < bfCost:
            dynamicErrorCount += 1
            print("DynamicLBError: Dag %s: brute force optimal and dynamic timed out, but brute force cost (%d) is worse than dynamic cost (%d)" % (dagName, bfCost, bbCost))
        else:
            goodCount += 1
            print("Good: Dag %s: brute force optimal and dynamic timed out, and brute force cost (%d) is not worse than dynamic cost (%d)" % (dagName, bfCost, bbCost))
    elif bfResult == TIMEOUT and bbResult == SUCCESS:
        if bbCost > bfCost:
            dynamicErrorCount += 1
            print("DynamicLBError: Dag %s: brute force timed out and dynamic optimal, but brute force cost (%d) is better than dynamic cost (%d)" % (dagName, bfCost, bbCost))
        else:
            goodCount += 1
            print("Good: Dag %s: brute force timed out and dynamic optimal, and brute force cost (%d) is not better than dynamic cost (%d)" % (dagName, bfCost, bbCost))


print("Good: %d" % goodCount)
print("Static LB Error: %d" % staticErrorCount)
print("Dynamic LB Error: %d" % dynamicErrorCount)
