#!/usr/bin/python3

import os
import sys
import re
import csv
import pandas as pd
from decimal import Decimal

benchName = re.compile("build    ([0-9]+).([a-zA-Z_]+)")
benchTime = re.compile("([0-9]+) total seconds elapsed")
functionName = re.compile("Function: ([a-zA-Z_:0-9]+)")
spilledLiveRangesRE = re.compile("Number of spilled live ranges: (\d+)")
weightedSpillsRE = re.compile(" SC in Function [a-zA-Z_:0-9]+ (\d+)")
fnOccRE      = re.compile("Final occupancy for function ([a-zA-Z_0-9]+):([0-9]+)")

def parseLog(log):
    # get name and total time before we cut off top and bottom
    name = "prim"
    time = "0"
    
    res = {}

    timeInfo = {}
    timeStuff = {}
    timeStuff["bench_name"] = str(name)
    timeStuff["compile_time"] = int(time)
    timeInfo[0] = timeStuff
    functionNum = 0
    
    occupancyList = re.findall("Final occupancy for function ([a-zA-Z_0-9]+):([0-9]+)", log)
    print("Function count: ", len(occupancyList))
    for x in occupancyList:
        functionInfo = {}
        functionInfo["function_name"] = x[0]
        functionInfo["Occupancy"] = int(x[1])

        res[functionNum] = functionInfo
        functionNum += 1

    return [res, timeInfo]

def parseLogSpills(log):
    # get name and total time before we cut off top and bottom
    name = "prim"
    time = "0"

    res = {}

    timeInfo = {}
    timeStuff = {}
    timeStuff["bench_name"] = str(name)
    timeStuff["compile_time"] = int(time)
    timeInfo[0] = timeStuff
    functionNum = 0

    spillsList = re.findall("Number of spilled live ranges: (\d+)", log)
    print("Function count: ", len(spillsList))
    for x in spillsList:
        functionInfo = {}
        spilledLiveRanges = int(x[0])
        functionInfo["spilled_live_ranges"] = int(spilledLiveRanges.group(1))

        res[functionNum] = functionInfo
        functionNum += 1

    return [res, timeInfo]

def getLogData(f):
    rgns = parseLog(f.read())
    dataFrame = pd.DataFrame(rgns[0])
    dataFrameTime = pd.DataFrame(rgns[1])
    return [dataFrame, dataFrameTime]

def getLogDataSpills(f):
    rgns = parseLogSpills(f.read())
    dataFrame = pd.DataFrame(rgns[0])
    dataFrameTime = pd.DataFrame(rgns[1])
    return [dataFrame, dataFrameTime]

def getLogDataCompileTime(f):
    rgns = parseLogCompileTime(f.read())
    dataFrame = pd.DataFrame(rgns)
    return dataFrame

# prints results of log file




def iterationsSaved(series):
    return


# main
if len(sys.argv) == 3:
  dataFrame = None
  dataFrameCompileTime = None
  #Bruce, change this
  fpath =  sys.argv[1]
  if os.path.isfile(fpath):
    print("Checking log: " + fpath)
    with open(fpath) as f:
        theData = getLogData(f)
        logData = theData[0]
        logDataCompileTime = theData[1]
        # data is transposed from what I want
        logDataTransposed = logData.T
        logDataCompileTimeTransposed = logDataCompileTime.T
        # add benchmark to the dataFrame
        dataFrame = pd.concat([dataFrame, logDataTransposed])
        dataFrameCompileTime = pd.concat(
            [dataFrameCompileTime, logDataCompileTimeTransposed])
  #dataFrame = dataFrame.loc[dataFrame['pass_no'] == 2]
  print(dataFrame)

  with pd.ExcelWriter(sys.argv[2] + ".xlsx", engine='xlsxwriter') as writer:
    dataFrame.to_excel(writer, sheet_name='AllRegions')

else:
  print ("USAGE: simpleExtractDataPandas.py <name of log> <name of output file>")
