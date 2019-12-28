import os
import re
import argparse
from openpyxl import Workbook

# For AMD 
RE_DAG_NAME = re.compile('Processing DAG (.*) with')
RE_SCHED_LENGTH = re.compile('The list schedule is of length (\d+) and')

# For OptSched
RE_PASS_NUM = re.compile(r"End of (.*) pass through")
RE_DAG_INFO = re.compile(r"INFO: Best schedule for DAG (.*) has cost (\d+) and length (\d+). The schedule is (.*) \(Time")

# Contains all of the stats
benchStats = {}

# List of benchmark names
benchmarks = [
    "densenet121",
    "densenet169",
    "densenet201",
    "inception_resnet_v2",
    "inception_v3",
    "mobilenet",
    "nasnet_large",
    "nasnet_mobile",
    "resnet50",
    "vgg16",
    "vgg19",
    "xception",
    "imdb_lstm",
]
def start(isAMD):
    for bench in benchmarks:
        currentLogFile = os.path.join(bench, bench + '.log')

        stats = {}
        stats['average'] = 0.0
        stats['total'] = 0.0
        stats['numRegions'] = 0
        stats['maxLength'] = 0

        # First check if log file exists.
        if os.path.exists(currentLogFile):
            # Open log file if it exists.
            with open(currentLogFile) as file: 
                # Read the whole log file
                # and split the scheduling
                # regions into a list
                log = file.read()
                blocks = log.split("********** Opt Scheduling **********")[1:]

                # Iterate over each scheduling region
                for block in blocks:
                    dagName = ''
                    schedLength = 0

                    # Skip first pass because it isn't the
                    # final schedule
                    getPassNum = RE_PASS_NUM.search(block)
                    passNum = getPassNum.group(1)
                    if passNum == 'first':
                        continue

                    # First check if B&B is enabled because
                    # with B&B enabled, the final output will
                    # be different.
                    # If B&B is not enabled, check for
                    # schedule from heuristic.
                    DAGInfo = RE_DAG_INFO.search(block)
                    if (DAGInfo):
                        dagName = DAGInfo.group(1)
                        schedLength = int(DAGInfo.group(3))
                    else:
                        getSchedLength = RE_SCHED_LENGTH.search(block)
                        schedLength = int(getSchedLength.group(1))

                    stats['total'] += schedLength
                    stats['numRegions'] += 1

                    if stats['maxLength'] < schedLength:
                        stats['maxLength'] = schedLength
                    
            if stats['numRegions'] != 0:
                stats['average'] = stats['total']/stats['numRegions']
                
            benchStats[bench] = stats

    total = 0.0
    numRegions = 0
    maxLength = 0
    test = 0
    for bench in benchmarks:
        print('{} : Average: {:0.2f} Max : {}'.format(bench,
                                                     benchStats[bench]['average'],
                                                     benchStats[bench]['maxLength']))
        total += benchStats[bench]['total']
        numRegions += benchStats[bench]['numRegions']
        if maxLength < benchStats[bench]['maxLength']:
            maxLength = benchStats[bench]['maxLength']


    print('Average : {:0.2f} Max : {}'.format(total/numRegions, maxLength))

def main(args):
    start(args.isAMD)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to get schedule length', \
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--amd', '-a', action='store_true',
                        default=False,
                        dest='isAMD',
                        help='Parse AMD scheduler logs for schedule length')

    args = parser.parse_args()

    main(args)
