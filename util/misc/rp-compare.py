#/usr/bin/python3
# Calculate how often OptSched's register pressure estimates match LLVM's
# You must compile OptSched with IS_DEBUG_PEAK_PRESSURE flag enabled.

import sys
import re

# The number of register types.
MAX_REG_TYPES = 30

RP_OPT_INFO = re.compile(r'INFO: OptSchPeakRegPres Index (\d+) Name (.+) Peak (\d+) Limit (\d+)')
RP_AFT_INFO = re.compile(r'INFO: PeakRegPresAfter  Index (\d+) Name (.+) Peak (\d+) Limit (\d+)')
RP_DAG_NAME = re.compile(r'INFO: Processing DAG (.+) with')

totalBlocks = 0
totalMismatches = 0
majorMismatches = 0

with open(str(sys.argv[1])) as logfile:
    log = logfile.read()
    blocks = log.split("INFO: ********** Opt Scheduling **********")

for block in blocks:
    optSchedPressures = [None]*MAX_REG_TYPES
    llvmPressures = [None]*MAX_REG_TYPES
    if (len(RP_DAG_NAME.findall(block)) == 0):
        continue;

    totalBlocks+=1
    blockName = RP_DAG_NAME.findall(block)[0]

    for matchOpt in RP_OPT_INFO.finditer(block):
        index = int(matchOpt.group(1))
        name = matchOpt.group(2)
        peak = matchOpt.group(3)
        limit = matchOpt.group(4)
        optSchedPressures[index] = {}
        optSchedPressures[index]['name'] = name
        optSchedPressures[index]['peak'] = peak
        optSchedPressures[index]['limit'] = limit

    for matchLLVM in RP_AFT_INFO.finditer(block):
        index = int(matchLLVM.group(1))
        name = matchLLVM.group(2)
        peak = matchLLVM.group(3)
        limit = matchLLVM.group(4)
        llvmPressures[index] = {}
        llvmPressures[index]['name'] = name
        llvmPressures[index]['peak'] = peak
        llvmPressures[index]['limit'] = limit

    for i in range(MAX_REG_TYPES):
        optP = optSchedPressures[i]
        llvmP = llvmPressures[i]

        if (optP['peak'] != llvmP['peak']):
            print('Mismatch in block ' + blockName + '.')
            print('Reg type with mismatch ' + optP['name'] + \
                  ' Limit ' + optP['limit'] + ' Peak OptSched ' + optP['peak'] + \
                  ' Peak LLVM ' + llvmP['peak'] + '.')
            totalMismatches+=1
            # A major mismatch occurs when peak pressure is over physical limit.
            if (max(int(optP['peak']), int(llvmP['peak'])) > int(optP['limit'])):
                print('Major mismatch!')
                majorMismatches+=1

print('Total blocks processed ' + str(totalBlocks) + '.')
print('Total mismatches ' + str(totalMismatches) + '.')
print('Total major mismatches ' + str(majorMismatches) + '.')
