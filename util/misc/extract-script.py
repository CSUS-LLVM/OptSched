#!/usr/bin/env python3
import argparse
import os, sys
import shutil
import re

parser = argparse.ArgumentParser(
    description='Extract a standalone version of an OptSched script')
parser.add_argument(
    'script', help='The path to script to extract a standalone version of')
parser.add_argument(
    'output', help='The output file to write the extracted script to')
parser.add_argument('--optsched', help='The path to the OptSched directory, '
                    'if this extract-script.py is not in its original location')

args = parser.parse_args()

OPTSCHED_ROOT = args.optsched if args.optsched else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON_FNS = os.path.join(OPTSCHED_ROOT, 'util', 'readlogs', '__init__.py')

with open(args.script, 'r') as f:
    script = f.read()

with open(COMMON_FNS, 'r') as f:
    readlogs = f.read()


def replace_module(modulename, modulecontent, script):
    return re.sub(
        r'^(?:(?:\s*from\s+{0}\s+import.*)|(?:\s*import\s+{0}.*))$'.format(re.escape(modulename)),
        modulecontent, script, flags=re.MULTILINE)

script = script.replace('sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n', '')
script = replace_module('readlogs', readlogs, script)

if 'readlogs' in script:
    sys.exit('Failed to make {} standalone. The "readlogs" library couldn\'t be'
        ' replaced.'.format(args.script))

if os.path.isdir(args.output):
    # Allow cp-like behavior of "copy to this directory" rather than requiring a
    # name for the script.
    output = os.path.join(args.output, os.path.basename(args.script))
else:
    output = args.output
    # Allow placing in a non-existent directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

with open(output, 'w') as f:
    f.write(script)
# Try to keep all permissions
shutil.copystat(args.script, output)
