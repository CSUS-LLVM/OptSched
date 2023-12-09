#!/usr/bin/python3

import os
import sys
import re
import csv
import pandas as pd
from decimal import Decimal

#def countIfGreaterThan5(series):
#    return lambda x, y:

#main
if len(sys.argv) == 4:
  xlsx1 = pd.ExcelFile(sys.argv[1])
  xlsx2 = pd.ExcelFile(sys.argv[2])
  dfRun1 = pd.read_excel(xlsx1, 'AllRegions')
  dfRun2 = pd.read_excel(xlsx2, 'AllRegions')
  
  dfRegionInfo1 = dfRun1
  dfRegionInfo1['RP Cost Percent Improvement'] = (dfRegionInfo1['initial_spill_cost'] - dfRegionInfo1['aco_abs_rp_cost'])/dfRegionInfo1['initial_spill_cost']
  dfRegionInfo1['Sched Length Percent Improvement'] = (dfRegionInfo1['initial_length'] - dfRegionInfo1['aco_length'])/dfRegionInfo1['initial_length']
  dfRegionInfo1['Regions with >= 5% RP Cost Improvement'] = dfRegionInfo1['RP Cost Percent Improvement'] > 0.05
  dfRegionInfo1['Regions with >= 5% Sched Length Improvement'] = dfRegionInfo1['Sched Length Percent Improvement'] > 0.05
  dfRegionInfoFinal1 = dfRegionInfo1.groupby(['bench', 'pass_no'], sort=True).agg({'RP Cost Percent Improvement':['max', 'min'], 'Sched Length Percent Improvement':['max', 'min'], 
  'Regions with >= 5% RP Cost Improvement':['sum'], 'Regions with >= 5% Sched Length Improvement':['sum']})
  dfRegionInfoFinal1.columns = ['Max RP Cost Improvement', 'Min RP Cost Improvement', 'Max Sched Length Improvement', 'Min Sched Length Improvement', 'Regions with >= 5% RP Cost Improvement', 'Regions with >= 5% Sched Length Improvement']
  

  dfRegionInfo2 = dfRun2
  dfRegionInfo2['RP Cost Percent Improvement'] = (dfRegionInfo2['initial_spill_cost'] - dfRegionInfo2['aco_abs_rp_cost'])/dfRegionInfo2['initial_spill_cost']
  dfRegionInfo2['Sched Length Percent Improvement'] = (dfRegionInfo2['initial_length'] - dfRegionInfo2['aco_length'])/dfRegionInfo2['initial_length']
  dfRegionInfo2['Regions with >= 5% RP Cost Improvement'] = dfRegionInfo2['RP Cost Percent Improvement'] > 0.05
  dfRegionInfo2['Regions with >= 5% Sched Length Improvement'] = dfRegionInfo2['Sched Length Percent Improvement'] > 0.05
  dfRegionInfoFinal2 = dfRegionInfo2.groupby(['bench', 'pass_no'], sort=True).agg({'RP Cost Percent Improvement':['max', 'min'], 'Sched Length Percent Improvement':['max', 'min'], 
  'Regions with >= 5% RP Cost Improvement':['sum'], 'Regions with >= 5% Sched Length Improvement':['sum']})
  dfRegionInfoFinal2.columns = ['Max RP Cost Improvement', 'Min RP Cost Improvement', 'Max Sched Length Improvement', 'Min Sched Length Improvement', 'Regions with >= 5% RP Cost Improvement', 'Regions with >= 5% Sched Length Improvement']


  dfCompareRegion = pd.DataFrame()
  dfCompareRegion['Run 1 Max RP Improvement'] = dfRegionInfoFinal1['Max RP Cost Improvement']
  dfCompareRegion['Run 2 Max RP Improvement'] = dfRegionInfoFinal2['Max RP Cost Improvement']
  dfCompareRegion['Run 1 Min RP Improvement'] = dfRegionInfoFinal1['Min RP Cost Improvement']
  dfCompareRegion['Run 2 Min RP Improvement'] = dfRegionInfoFinal2['Min RP Cost Improvement']
  dfCompareRegion['Run 1 Max Sched Length Improvement'] = dfRegionInfoFinal1['Max Sched Length Improvement']
  dfCompareRegion['Run 2 Max Sched Length Improvement'] = dfRegionInfoFinal2['Max Sched Length Improvement']
  dfCompareRegion['Run 1 Min Sched Length Improvement'] = dfRegionInfoFinal1['Min Sched Length Improvement']
  dfCompareRegion['Run 2 Min Sched Length Improvement'] = dfRegionInfoFinal2['Min Sched Length Improvement']
  
  dfCompareRegion['Run 1 Regions with >= 5% RP Cost Improvement'] = dfRegionInfoFinal1['Regions with >= 5% RP Cost Improvement']
  dfCompareRegion['Run 2 Regions with >= 5% RP Cost Improvement'] = dfRegionInfoFinal2['Regions with >= 5% RP Cost Improvement']
  dfCompareRegion['Run 1 Regions with >= 5% Sched Length Improvement'] = dfRegionInfoFinal1['Regions with >= 5% Sched Length Improvement']
  dfCompareRegion['Run 2 Regions with >= 5% Sched Length Improvement'] = dfRegionInfoFinal2['Regions with >= 5% Sched Length Improvement']

  
  # get aco Time and launches per benchmark per pass for run 1
  dfTimeLaunches1 = dfRun1[dfRun1['total_iterations'] > -1]
  dfTimeLaunches1 = dfTimeLaunches1.groupby(['bench', 'pass_no'], sort=True).agg({'ACOTime':['sum', 'count'], 'copyTime':['sum'], 'aco_length':['mean'], 'aco_abs_rp_cost':['mean']})
  dfTimeLaunches1.columns = ['ACO time', 'ACO Launches', 'Copy Time', 'Avg Schedule Length', 'Avg RP Cost']
  
  
  # get aco Time and launches per benchmark per pass for run 2
  dfTimeLaunches2 = dfRun2[dfRun2['total_iterations'] > -1] 
  dfTimeLaunches2 = dfTimeLaunches2.groupby(['bench', 'pass_no'], sort=True).agg({'ACOTime':['sum', 'count'], 'copyTime':['sum'], 'aco_length':['mean'], 'aco_abs_rp_cost':['mean']})
  dfTimeLaunches2.columns = ['ACO time', 'ACO Launches', 'Copy Time', 'Avg Schedule Length', 'Avg RP Cost']
  
  
  
  # make comparison dfs
  dfCompareTime = pd.DataFrame()
  dfCompareQuality = pd.DataFrame()
  
  # get data from relevant frames
  dfCompareTime['Run 1 ACO Time'] = dfTimeLaunches1['ACO time']
  dfCompareTime['Run 2 ACO Time'] = dfTimeLaunches2['ACO time']
  dfCompareTime['Run 1 ACO Launches'] = dfTimeLaunches1['ACO Launches']
  dfCompareTime['Run 2 ACO Launches'] = dfTimeLaunches2['ACO Launches']
  dfCompareTime['Run 1 Copy Time'] = dfTimeLaunches1['Copy Time']
  dfCompareTime['Run 2 Copy Time'] = dfTimeLaunches2['Copy Time']
  
  dfCompareQuality['Run 1 Avg Schedule Length'] = dfTimeLaunches1['Avg Schedule Length']
  dfCompareQuality['Run 2 Avg Schedule Length'] = dfTimeLaunches2['Avg Schedule Length']
  dfCompareQuality['Run 1 Avg RP Cost'] = dfTimeLaunches1['Avg RP Cost']
  dfCompareQuality['Run 2 Avg RP Cost'] = dfTimeLaunches2['Avg RP Cost']
  
  # create global row grouped by pass_no
  dfGlobalTime = dfCompareTime.groupby('pass_no').sum()
  dfGlobalQuality = dfCompareQuality.groupby('pass_no').sum()
  # set the indices to be the same
  dfGlobalTime['bench'] = 'Global'
  dfGlobalTime['pass_no'] = dfGlobalTime.index
  dfGlobalTime.set_index(['bench', 'pass_no'],inplace=True)
  dfCompareTime = pd.concat([dfCompareTime, dfGlobalTime])
  
  dfGlobalQuality['bench'] = 'Global'
  dfGlobalQuality['pass_no'] = dfGlobalQuality.index
  dfGlobalQuality.set_index(['bench', 'pass_no'],inplace=True)
  dfCompareQuality = pd.concat([dfCompareQuality, dfGlobalQuality])
  
  compareSameProcessor = True
  if compareSameProcessor:
    # now that global row is added, do the needed calculations and comparisons
    dfCompareTime['Percent ACO Time Improvement'] = (dfCompareTime['Run 1 ACO Time'] - dfCompareTime['Run 2 ACO Time'])/dfCompareTime['Run 1 ACO Time']
    dfCompareTime['Absolute ACO Time Improvement'] = dfCompareTime['Run 1 ACO Time'] - dfCompareTime['Run 2 ACO Time']
    dfCompareTime['Percent Copy Time Improvement'] = (dfCompareTime['Run 1 Copy Time'] - dfCompareTime['Run 2 Copy Time'])/dfCompareTime['Run 1 Copy Time']
    dfCompareTime['Absolute Copy Time Improvement'] = dfCompareTime['Run 1 Copy Time'] - dfCompareTime['Run 2 Copy Time']
    
    dfCompareQuality['Percent Avg Schedule Length Improvement'] = (dfCompareQuality['Run 1 Avg Schedule Length'] - dfCompareQuality['Run 2 Avg Schedule Length'])/dfCompareQuality['Run 1 Avg Schedule Length']
    dfCompareQuality['Absolute Avg Schedule Length Improvement'] = dfCompareQuality['Run 1 Avg Schedule Length'] - dfCompareQuality['Run 2 Avg Schedule Length']
    dfCompareQuality['Percent Avg RP Cost Improvement'] = (dfCompareQuality['Run 1 Avg RP Cost'] - dfCompareQuality['Run 2 Avg RP Cost'])/dfCompareQuality['Run 1 Avg RP Cost']
    dfCompareQuality['Absolute Avg RP Cost Improvement'] = dfCompareQuality['Run 1 Avg RP Cost'] - dfCompareQuality['Run 2 Avg RP Cost']
    
    # reorder columns
    dfCompareTime = dfCompareTime[['Run 1 ACO Time', 'Run 2 ACO Time', 'Percent ACO Time Improvement', 'Absolute ACO Time Improvement', 'Run 1 Copy Time', 'Run 2 Copy Time', 'Percent Copy Time Improvement', 'Absolute Copy Time Improvement', 'Run 1 ACO Launches', 'Run 2 ACO Launches']]
    #dfCompareTime.rename(columns={0:'Run 1\nACO Time', 1:'Run 2\nACO Time', 2:'Percent ACO\nTime Improvement', 3:'Absolute ACO\nTime Improvement', 4:'Run 1\nCopy Time', 5:'Run 2\nCopy Time', 6:'Percent Copy\nTime Improvement', 7:'Absolute Copy\nTime Improvement', 8:'Run 1 ACO Launches', 9:'Run 2 ACO Launches'})
    print(dfCompareTime)
    
    dfCompareQuality = dfCompareQuality[['Run 1 Avg Schedule Length', 'Run 2 Avg Schedule Length', 'Percent Avg Schedule Length Improvement', 'Absolute Avg Schedule Length Improvement', 'Run 1 Avg RP Cost', 'Run 2 Avg RP Cost', 'Percent Avg RP Cost Improvement', 'Absolute Avg RP Cost Improvement']]
    #dfCompareQuality.rename(columns={0:'Run 1 Avg\nSchedule Length', 1:'Run 2 Avg\nSchedule Length', 2:'Percent Avg Schedule\nLength Improvement', 3:'Absolute Avg Schedule\nLength Improvement', 4:'Run 1 Avg\nRP Cost', 5:'Run 2 Avg\nRP Cost', 6:'Percent Avg Spill\nCost Improvement', 7:'Absolute Avg Spill\nCost Improvement'})

  else:
    # now that global row is added, do the needed calculations and comparisons
    dfCompareTime['Speedup ratio of Device compared to Host'] = dfCompare['Run 1 ACO Time']/dfCompare['Run 2 ACO Time']
    
    # reorder columns
    dfCompareTime = dfCompareTime[['Run 1 ACO Time', 'Run 2 ACO Time', 'Speedup ratio of Device compared to Host', 'Run 1 ACO Launches', 'Run 2 ACO Launches']]
    print(dfCompareTime)
    
  #dfCompareTime.style.set_table_styles([dict(selector="th",props=[('max-width', '50px')])])
  #dfCompareQuality.style.set_table_styles([dict(selector="th",props=[('max-width', '50px')])])
  with pd.ExcelWriter(sys.argv[3] + ".xlsx", engine='xlsxwriter') as writer: 
    dfCompareTime.to_excel(writer, sheet_name='Time Comparison', float_format = "%0.4f")
    dfCompareQuality.to_excel(writer, sheet_name='Schedule Quality Comparison', float_format = "%0.4f")
    dfRun1.to_excel(writer, sheet_name='Run 1 All Regions')
    dfRun2.to_excel(writer, sheet_name='Run 2 All Regions')
    dfCompareRegion.to_excel(writer, sheet_name='Region Analysis')
    
    

else:
  print("Usage: Name of first .xlsx file (the one that you expect to perform worse), name of second .xlsx file, name of .xlsx file to store results")
  print("Example: ./compareTwoRuns.py run1 run2 compareRun1and2")
