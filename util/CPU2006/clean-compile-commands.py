#!/usr/bin/env python3
import argparse
import re

parser = argparse.ArgumentParser(description='Cleans CPU2006 logs, moving compilation commands to the appropriate location if necessary')
parser.add_argument('files', nargs='+', help='The logs to clean, in place')

files = parser.parse_args().files

RE_AFTER_FUNCTION = re.compile(r'\*{20,}\nFunction:.*?\*{20,}\n', re.DOTALL)
RE_BUGGED_COMPILE_COMMAND = re.compile(
    r'''
    # Compilation commands will always appear at the beginning of a line if things happened correctly.
    # We're trying to fix it when it doesn't happen correctly.
    ^(?: # (Non-capturing)
        E       # Inside an EVENT: log, but possible happening mid-word (EV/usr/bin/clang++ ...)
        | I     # Inside an INFO: log
    )
    .*?
    (
        # When we see any of the bugged compilation commands,
        (/.*?/[cf]lang.*\n) # clang, clang++, flang
        | (specperl\ /.*\n) # specperl commands
    )   # then we want to match the command and move it to the end.
    ''',
    re.VERBOSE | re.MULTILINE)

for file in files:
    with open(file, 'r') as f:
        text = f.read()

    # Keep the file content we wish to write back as a list of strings.
    # We will do a join at the end.
    result = []

    cur = 0
    # Iterate over the locations that we will place the bugged commands (after next fn)
    for next_fn_m in RE_AFTER_FUNCTION.finditer(text):
        # The strings we will be placing after the fn.
        after_fn = []

        # Gather all the bugged compile commands from `cur` to the location of this next_fn_m.
        while True:
            bugged = RE_BUGGED_COMPILE_COMMAND.search(text, cur, next_fn_m.start())

            if bugged:
                result.append(text[cur:bugged.start(1)])
                after_fn.append(bugged.group(1))
                cur = bugged.end(1)
            else:
                result.append(text[cur:next_fn_m.end()])
                cur = next_fn_m.end()
                break
        result += after_fn

    # Include any remnant
    result.append(text[cur:])

    resultstr = ''.join(result)
    with open(file, 'w') as f:
        f.write(resultstr)
