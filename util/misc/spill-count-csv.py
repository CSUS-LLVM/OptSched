import sys
import re
import os

BenchmarkRegex = re.compile(r'(.*?):$')
FunctionRegex = re.compile(r' +(\d+) (.*?)$')
DEBUG = False

def debugPrint(str):
    if DEBUG: print(str)

if len(sys.argv) != 2:
    raise Exception("Invalid number of arguments. Expected 1")

if not os.path.isfile(sys.argv[1]):
    raise Exception("%s is not a file!" % sys.argv[1])

with open(sys.argv[1]) as f:
    benchName = ""
    for line in f:
        match = BenchmarkRegex.match(line)
        if not match is None:
            benchName = match.group(1)
            debugPrint("Found benchmark %s" % benchName)
            continue
        match = FunctionRegex.match(line)
        if not match is None:
            debugPrint("Found function %s with %d spills" % (match.group(2), int(match.group(1))))
            sys.stdout.write("%s,%s,%d\n" % (benchName, match.group(2), int(match.group(1))))
        else:
            debugPrint("Not a match: %s" % line)
