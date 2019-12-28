import os
import re

REGEX_OCCUPANCY = re.compile('Final occupancy for function (.*):(\d+)')

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

# Ignore these functions
# They are outputted before scheduling
ignore = [
    'copyBufferRect',
    'copyBufferRectAligned',
    'copyBuffer',
    'copyBufferAligned',
    'fillBuffer',
    'copyBufferToImage',
    'copyImageToBuffer',
    'copyImage',
    'copyImage1DA',
    'fillImage',
    'scheduler'
]

# Get name of all directories in current folder
subfolders = [f.name for f in os.scandir(".") if f.is_dir() ]

# For each folder
for folderName in subfolders:
    name = folderName.split("-")

    # Get the run number from the end
    # of the folder name
    runNumber = name[-1]

    # Get the name of the run
    # and exclude the run number
    nameOfRun = "-".join(name[:-1])
        
    # Create an entry in the stats for the
    # name of the run
    if (nameOfRun not in benchStats):
        benchStats[nameOfRun] = {}

    for bench in benchmarks:
        currentPath = os.path.join(folderName, bench)
        currentLogFile = os.path.join(currentPath, bench + ".log")
        stats = {}
        stats['average'] = 0.0
        stats['total'] = 0.0
        stats['numKernel'] = 0

        # First check if log file exists.
        if (os.path.exists(currentLogFile)):
            # Open log file if it exists.
            with open(currentLogFile) as file:
                for line in file:
                    # Match the line that contain occupancy stats
                    getOccupancyStats = REGEX_OCCUPANCY.match(line)
                    if (getOccupancyStats):
                        # Get the kernel name
                        kernelName = getOccupancyStats.group(1)

                        # Ignore these function
                        if (kernelName in ignore):
                            continue

                        # Get occupancy
                        occupancy = int(getOccupancyStats.group(2))
                        
                        # Used for averaging
                        stats['total'] += occupancy
                        stats['numKernel'] += 1
        else:
            print("Cannot find log file for {} run {} benchmark {}.".format(nameOfRun, runNumber, bench))
        
        if stats['numKernel'] != 0:
            stats['average'] = stats['total'] / stats['numKernel']

        # Save stats
        benchStats[nameOfRun][bench] = stats

for nameOfRun in benchStats:
    print('{}'.format(nameOfRun))
    total = 0.0
    kernel = 0
    for bench in benchStats[nameOfRun]:
        print('    {} : {:.2f}'.format(bench, benchStats[nameOfRun][bench]['average']))
        total += benchStats[nameOfRun][bench]['total']
        kernel += benchStats[nameOfRun][bench]['numKernel']
    if kernel != 0:
        print('  Average: {:.2f}'.format(total/kernel))
