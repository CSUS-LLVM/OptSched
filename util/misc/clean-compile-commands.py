#!/usr/bin/env python3
import argparse
import re

parser = argparse.ArgumentParser(description='Cleans CPU2006 logs, moving compilation commands to the appropriate location if necessary')
parser.add_argument('files', nargs='+', help='The logs to clean, in place')

files = parser.parse_args().files

RE_AFTER_FUNCTION = re.compile(r'\*{20,}\nFunction:.*?\*{20,}\n', re.DOTALL)
RE_BUGGED_COMPILE_COMMAND = re.compile(r'EVENT: \{.*?(/.*?/[cf]lang.*\n)')

for file in files:
    with open(file, 'r') as f:
        text = f.read()

    result = []
    lastpos = 0

    while True:
        bugged = RE_BUGGED_COMPILE_COMMAND.search(text, lastpos)

        if bugged:
            result.append(text[lastpos:bugged.start(1)])

            next_function = RE_AFTER_FUNCTION.search(text, bugged.end())

            result.append(text[bugged.end(1):next_function.end()])
            result.append(bugged.group(1))
            lastpos = next_function.end()
        else:
            result.append(text[lastpos:])
            break

    resultstr = ''.join(result)
    with open(file, 'w') as f:
        f.write(resultstr)
