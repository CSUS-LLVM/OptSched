import re
import mmap
import optparse
import os

regex = re.compile("Examined (\d+) nodes")

def getNodeCount(fileName):
    count = 0
    with open(fileName) as bff:
        bffm = mmap.mmap(bff.fileno(), 0, access=mmap.ACCESS_READ)

        for match in regex.finditer(bffm):
            count += int(match.group(1))

        bffm.close()

    return count

parser = optparse.OptionParser(
    description='Wrapper around runspec for collecting spill counts.')
parser.add_option('-p', '--path',
                  metavar='path',
                  default=None,
                  help='Log file.')
parser.add_option('--isfolder',
                  action='store_true',
                  help='Specify if parsing a foldere.')

args = parser.parse_args()[0]

total = 0

if args.isfolder:
    if not os.path.isdir(args.path):
        raise Error("Please specify a valid folder.")
    for filename in os.listdir(args.path):
        total += getNodeCount(os.path.join(args.path, filename))
else:
    if not os.path.isfile(args.path):
        raise Error("Please specify a valid log file.")
    total += getNodeCount(args.path)

print(total)
