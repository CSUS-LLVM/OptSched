#!/usr/bin/python3

import os
import sys
import re
import csv
import pandas as pd
from decimal import Decimal

#benchName  = re.compile("build    ([0-9]+).([a-zA-Z_]+)")
benchName  = re.compile("Building CXX object .*\/([a-zA-Z_]+)\.cpp\.o")
benchTime  = re.compile("([0-9]+) total seconds elapsed")
rgnName    = re.compile("Processing DAG ([a-zA-Z_:0-9]+) with (\d+) insts")
iCostRE    = re.compile("The list schedule is of length (\d+) and spill cost (\d+)\. Tot cost = (\d+)")
bbCost     = re.compile("Best schedule for DAG (.+) has cost (\d+) and length (\d+)")
bbOptimal  = re.compile("DAG solved optimally")
acoImprove = re.compile("ACO found schedule cost:(\d+), rp cost:(\d+), exec cost: (\d+), and iteration:(\d+) \(sched length: (\d+), abs rp cost: (\d+), rplb: (\d+)\)")
bestSched  = re.compile("Best schedule: Absolute RP Cost: (\d+), Length: (\d+), Cost: (\d+)")
#acoImprove = re.compile("ACO found schedule cost:(\d+), rp cost:(\d+), exec cost: (\d+), \(sched length: (\d+), abs rp cost: (\d+), rplb: (\d+)\)")
acoTime    = re.compile("ACO_Time (\d+)")
copyTime   = re.compile("Launching Dev_ACO with (\d+) blocks of (\d+) threads \(Time = (\d+) ms\)")
totalIter  = re.compile("ACO finished after (\d+) iterations")
fPassInd   = re.compile("End of first pass through");
sPassInd   = re.compile("End of second pass through");
fnOcc      = re.compile("Final occupancy for function ([a-zA-Z_0-9]+):([0-9]+)")
rlSize     = re.compile("Ready List Size is: (\d+),")
lowerBound = re.compile("Sched Lower Bound: (\d+)")
cpDist     = re.compile("CP Distance: (\d+)")
memRatio   = re.compile("Memory Instructions: (\d+), Memory Ratio: (\d)\.(\d+), SUnits\.size\(\): (\d+)")
rlSize     = re.compile("Absolute difference in RL size per cycle: (\d)\.(\d+), percent difference in RL size: (\d)\.(\d+)")
diffScheds = re.compile("(\d+) different schedules for (\d+) total ants \(Time = (\d+) ms\)")
revertedOcc = re.compile("Reverting Scheduling because of a decrease in occupancy from (\d+) to (\d+). \(Time = (\d+) ms\)");
revertedCycleThresh = re.compile("Skipping ACO \(No Mem ops\) \(Time = (\d+) ms\)");
revertedNoMemOps = re.compile("Skipping ACO \(list sched within (\d+) cycles from optimal\) \(Time = (\d+) ms\)");


def parseLog(log):
    #get name and total time before we cut off top and bottom
    name = benchName.search(log).group(1)
    time = 0
    #first we cut off the end where the results are reported
    log = log.split("Example finished")[0]
    #now we break the log into scheduling regions
    regions = log.split("INFO: ********** Opt Scheduling **********")
    #throw out first "region" is actually info regarding what benchmark is running
    regions = regions[1:]

    #iterate through regions extracting infoi
    res={}

    rgnNum = 0
    for region in regions:
        rgnInf={}
        mName = rgnName.search(region)
        rgnInf["instruction_count"]  =int(mName.group(2)) if mName else -1
        #ignore all regions with less than 50 instructions
        if rgnInf["instruction_count"] < 1:
            pass
        rgnInf["region_name"]        =mName.group(1) if mName else "err_no_name"
        bName = benchName.search(region)
        if bName:
            name = bName.group(1)
        rgnInf["bench"]              =str(name)
        rgnInf["instruction_count"]  =int(mName.group(2)) if mName else -1
        mICost = iCostRE.search(region)
        rgnInf["initial_length"]     =int(mICost.group(1)) if mICost else sys.maxsize
        rgnInf["initial_spill_cost"] =int(mICost.group(2)) if mICost else -1
        rgnInf["initial_total_cost"] =int(mICost.group(3)) if mICost else -1

        #figure out if we got an optimal list schedule
        rgnInf["lst_opt"]    = True if rgnInf["initial_total_cost"] == 0 else False

        #get bb cost and optimality
        #mbbImp = bbCost.search(region)
        #rgnInf["bb_cost"]    = int(mbbImp.group(2)) if mbbImp else -1
        #rgnInf["bb_length"]  = int(mbbImp.group(3)) if mbbImp else sys.maxsize
        #rgnInf["bb_optimal"] = 1 if bbOptimal.search(region) else 0

        rOcc = revertedOcc.search(region)
        rCT = revertedCycleThresh.search(region)
        rMem = revertedNoMemOps.search(region)

        rgnInf["revertedOcc"] = 1 if rOcc else 0
        rgnInf["revertedCycleThresh"] = 1 if rCT else 0
        rgnInf["revertedNoMemOps"] = 1 if rMem else 0

        #get aco cost and iterations
        mTotIter = totalIter.search(region)
        rgnInf["total_iterations"]   =int(mTotIter.group(1)) if mTotIter else -1
        ACOTime = acoTime.search(region)
        if mTotIter:
          rgnInf["ACOTime"] = int(ACOTime.group(1)) if ACOTime else 0
        else:
          rgnInf["ACOTime"] = -1
        acoImps = acoImprove.findall(region)
        bestSchedVals = bestSched.findall(region)
        #rgnInf["aco_cost"] = int(acoImps[-1][0]) if acoImps!=[] else -1
        #rgnInf["aco_length"] = int(acoImps[-1][4]) if acoImps!=[] else rgnInf["initial_length"]
        #rgnInf["aco_abs_rp_cost"] = int(acoImps[-1][5]) if acoImps!=[] else rgnInf["initial_spill_cost"]
        rgnInf["aco_cost"] = int(bestSchedVals[-1][2]) if bestSchedVals!=[] else -1
        rgnInf["aco_length"] = int(bestSchedVals[-1][1]) if bestSchedVals!=[] else rgnInf["initial_length"]
        rgnInf["aco_abs_rp_cost"] = int(bestSchedVals[-1][0]) if bestSchedVals!=[] else rgnInf["initial_spill_cost"]
        if rgnInf["total_iterations"] >= 0 and rgnInf["aco_cost"] == -1:
            rgnInf["aco_cost"] = rgnInf["initial_total_cost"]
            
        #get copyToDevice time if running on GPU
        copyInf = copyTime.search(region)
        rgnInf["copyTime"] = int(copyInf.group(3)) if copyInf else 0

        diffSchedInf = diffScheds.search(region)
        rgnInf["different_scheds"] = int(diffSchedInf.group(1)) if diffSchedInf else 0
        rgnInf["total_scheds"] = int(diffSchedInf.group(2)) if diffSchedInf else 0

        #get if we are first pass, second pass, or neither(0 means no pass system)
        if fPassInd.search(region):
            rgnInf["pass_no"] = 1
        elif sPassInd.search(region):
            rgnInf["pass_no"] = 2
        else:
            rgnInf["pass_no"] = 0
        
        mReadySize = rlSize.search(region)
        rgnInf["ready_size"] = int(mReadySize.group(1)) if mReadySize else -1
        
        #rgnInf["final_length"] = min(rgnInf["initial_length"], rgnInf["bb_length"], rgnInf["aco_length"])
        rgnInf["final_length"] = min(rgnInf["initial_length"], rgnInf["aco_length"])
        
        #mFnOcc = fnOcc.search(region)
        #rgnInf["fn_occ"] = int(mFnOcc.group(2)) if mFnOcc else -1

        #append rgnInf to end of results
        #if rgnInf["total_iterations"] >= 50:
        lowerBoundFinding = lowerBound.search(region)
        rgnInf["lower_bound"] = int(lowerBoundFinding.group(1)) if lowerBoundFinding else -1

        cpDistFinding = cpDist.search(region)
        rgnInf["CP_length"] = int(cpDistFinding.group(1)) if cpDistFinding else -1

        memRatioFinding = memRatio.search(region)
        rgnInf["memory_operations"] = float(memRatioFinding.group(1))

        rlSizeFinding = rlSize.search(region)
        #rgnInf["abs_ready_list_diff"] = float(rlSizeFinding.group(1)) if rlSizeFinding else -1
        rgnInf["abs_ready_list_diff"] = float(rlSizeFinding.group(1) + "." + rlSizeFinding.group(2)) if rlSizeFinding else -1
        rgnInf["percent_ready_list_diff"] = float(rlSizeFinding.group(3) + "." + rlSizeFinding.group(4)) if rlSizeFinding else -1

        res[rgnNum]=rgnInf
        rgnNum += 1
    return res

globalStats={
    'bench': "Global",
    'ACOTime':0,
    'ACOTime1':0,
    'ACOTime2':0,
    'numberOfAcoCalls':0,
    'numberOfAcoCalls1':0,
    'numberOfAcoCalls2':0,
    'avgRP':0,
    'RPSum':0,
    'avgLen':0,
    'lenSum':0,
    'rgnCnt':0,
    'copyTimeSum':0,
    'avgCopyTime':0
}

#calculates geomean of second pass costs where ACO is invoked
def getRgnRPandLen(regions):
  stats={
    'bench': "tbd",
    'ACOTime':0,
    'ACOTime1':0,
    'ACOTime2':0,
    'numberOfAcoCalls':0,
    'numberOfAcoCalls1':0,
    'numberOfAcoCalls2':0,
    'avgRP':0,
    'RPSum':0,
    'avgLen':0,
    'lenSum':0,
    'rgnCnt':0,
    'copyTimeSum':0,
    'avgCopyTime':0
  }
  listOfRegionDicts = []
  for _, rgn in regions.items():
    regionDict={
      'bench': "tbd",
      'region_name':0,
      'pass_no':0,
      'total_iterations':0,
      'ACOTime':0,
      'RP':0,
      'len':0,
      'copyTime':0,
    }
    if rgn["pass_no"] == 1 and rgn["instruction_count"] >= 50:
      regionDict["region_name"] = rgn["region_name"]
      regionDict["bench"] = rgn["bench"]
      regionDict["pass_no"] = rgn["pass_no"]
      regionDict["ACOTime"] = rgn["ACOTime"]
      regionDict["total_iterations"] = rgn["total_iterations"]
      if rgn["ACOTime"] > 0:
        regionDict["copyTime"] = rgn["copyTime"]
      listOfRegionDicts.append(regionDict)
    elif rgn["pass_no"] == 2 and rgn["instruction_count"] >= 50:
      regionDict["region_name"] = rgn["region_name"]
      regionDict["bench"] = rgn["bench"]
      regionDict["pass_no"] = rgn["pass_no"]
      regionDict["ACOTime"] = rgn["ACOTime"]
      regionDict["total_iterations"] = rgn["total_iterations"]
      if rgn["ACOTime"] > 0:
        regionDict["copyTime"] = rgn["copyTime"]
      regionDict["RP"] = rgn["initial_spill_cost"]
      regionDict["len"] = rgn["aco_length"]
      listOfRegionDicts.append(regionDict)

  # save global stats for benchmark and calculate averages
  if stats["rgnCnt"] > 0:
    globalStats["ACOTime"] += stats["ACOTime"];
    globalStats["ACOTime1"] += stats["ACOTime1"];
    globalStats["ACOTime2"] += stats["ACOTime2"];
    globalStats["numberOfAcoCalls"] += stats["numberOfAcoCalls"]
    globalStats["numberOfAcoCalls1"] += stats["numberOfAcoCalls1"]
    globalStats["numberOfAcoCalls2"] += stats["numberOfAcoCalls2"]
    globalStats["copyTimeSum"] += stats["copyTimeSum"]
    globalStats["RPSum"] += stats["RPSum"]
    globalStats["lenSum"] += stats["lenSum"]
    globalStats["rgnCnt"] += stats["rgnCnt"]
    stats["avgRP"] = stats["RPSum"]/stats["rgnCnt"]
    stats["avgLen"] = stats["lenSum"]/stats["rgnCnt"]
    stats["avgCopyTime"] = stats["copyTimeSum"]/stats["numberOfAcoCalls"]
  return listOfRegionDicts

def getLogData(f):
  rgns = parseLog(f.read())
  dataFrame = pd.DataFrame(rgns)
  return dataFrame

#prints results of log file
def printLogStats(logData):
  print(logData["bench"].ljust(13), '{:.2f}'.format(logData["avgRP"]).ljust(17), '{:.2f}'.format(logData["avgLen"]).ljust(12), str(logData["ACOTime"]).ljust(12), str(logData["ACOTime1"]).ljust(11), str(logData["ACOTime2"]).ljust(11), str(logData["copyTimeSum"]).ljust(12), str(logData["rgnCnt"]).ljust(16), str(logData["numberOfAcoCalls"]).ljust(15), str(logData["numberOfAcoCalls1"]).ljust(14), str(logData["numberOfAcoCalls2"]).ljust(14))

def writeCSV(logData):
  try:
    with open('test.csv', 'w') as file:
      #csv_columns = ["Benchmark name", "Average RP", "Average Length", "ACO TIME", "ACO Time in pass 1", "ACO Time in pass 2", "Sum of time spent copying", "Region count", "Calls to ACO", "Calls to ACO in pass 1", "Calls to ACO in pass 2"]
      #csv_columns = ['bench', 'ACOTime', 'ACOTime1', 'ACOTime2', 'numberOfAcoCalls', 'numberOfAcoCalls1', 'numberOfAcoCalls2', 'avgRP', 'RPSum', 'avgLen', 'lenSum', 'rgnCnt', 'copyTimeSum', 'avgCopyTime']
      csv_columns = ['bench', 'region_name', 'pass_no', "total_iterations", 'ACOTime', 'RP', 'len', 'copyTime']
      writer = csv.DictWriter(file, fieldnames=csv_columns)
      writer.writeheader()
      for data in logData:
        writer.writerow(data)
  except IOError:
    print("I/O Error")

def iterationsSaved(series):
  return

#main

if len(sys.argv) == 3:
  dataFrame = None
  fpath = sys.argv[1]
  if os.path.isfile(fpath):
    print("Checking log: " + fpath)
    with open(fpath) as f:
      logData = getLogData(f)
      # data is transposed from what I want
      logDataTransposed = logData.T
      # add benchmark to the dataFrame
      dataFrame = pd.concat([dataFrame,logDataTransposed])
  #dataFrame = dataFrame.loc[dataFrame['pass_no'] == 2]
  print(dataFrame)
  
  with pd.ExcelWriter(sys.argv[2] + ".xlsx", engine='xlsxwriter') as writer: 
    dataFrame.to_excel(writer, sheet_name='AllRegions')
    
    # formatting column widths
    # worksheet = writer.sheets['BenchInfo']
    # for i, col in enumerate(dataFrameTimeLaunches.columns):
      # column_len = max(dataFrameTimeLaunches[col].astype(str).str.len().max(), len(col) + 2)
      # if column_len > 50:
        # column_len = 50
      # worksheet.set_column(i, i, column_len)
      
    # worksheet = writer.sheets['BenchInfoIterations']
    # for i, col in enumerate(combinedDF.columns):
      # column_len = max(combinedDF[col].astype(str).str.len().max(), len(col) + 2)
      # if column_len > 50:
        # column_len = 50
      # worksheet.set_column(i, i, column_len)
      
    # worksheet = writer.sheets['AffectedRegions']
    # for i, col in enumerate(dataFrameGroups.columns):
      # column_len = max(dataFrameGroups[col].astype(str).str.len().max(), len(col) + 2)
      # if column_len > 50:
        # column_len = 50
      # worksheet.set_column(i, i, column_len)
    
    # worksheet = writer.sheets['AllRegions']
    # for i, col in enumerate(dataFrame.columns):
      # column_len = max(dataFrame[col].astype(str).str.len().max(), len(col) + 2)
      # if column_len > 50:
        # column_len = 50
      # worksheet.set_column(i, i, column_len)
    
  if globalStats["rgnCnt"] > 0:
    globalStats["avgRP"] = globalStats["RPSum"]/globalStats["rgnCnt"]
    globalStats["avgLen"] = globalStats["lenSum"]/globalStats["rgnCnt"]
    globalStats["avgCopyTime"] = globalStats["copyTimeSum"]/globalStats["numberOfAcoCalls"]
  #printLogStats(globalStats)

else:
  print ("USAGE: simpleExtractDataPandas.py <name of log> <name of output file>")
