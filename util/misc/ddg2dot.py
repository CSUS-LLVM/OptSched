#!/usr/bin/env python3
import argparse
import sys
import re

parser = argparse.ArgumentParser(description='Convert data_dep WriteToFile format to a .dot file')
parser.add_argument('input', help='The WriteToFile format file to convert. Input a single hyphen (-) to read from stdin')
parser.add_argument('-o', '--output', help='The destination to write to. Defaults to stdout')
parser.add_argument('--filter-weights', nargs='*', default=[], help='filter out weights with the respective values')

args = parser.parse_args()

if args.input == '-':
    infile = sys.stdin
else:
    infile = open(args.input, 'r')

filtered_weights = set(int(x) for x in args.filter_weights)

text = infile.read()
infile.close()

NODE_RE = re.compile(r'node (?P<number>\d+) "(?P<name>.*?)"(\s*"(?P<other_name>.*?)")?')
EDGE_RE = re.compile(r'dep (?P<from>\d+) (?P<to>\d+) "(?P<type>.*?)" (?P<weight>\d+)')

# Holds the resulting strings as a list of the lines.
result = ['digraph G {\n']

# Create the nodes in the graph
for match in NODE_RE.finditer(text):
    num = match['number']
    name = match['name']
    if name == 'artificial': # Prettify entry/exit names
        name = ['exit', 'entry'][match['other_name'] == '__optsched_entry']

    # Add the node to the graph. Include a node to make it clear what this is
    result.append(f'    n{num} [label="{name}:n{num}"];\n')

result.append('\n')

# Create the edges in the graph
for match in EDGE_RE.finditer(text):
    from_ = match['from']
    to = match['to']
    type_ = match['type']
    weight = match['weight']

    # The additional label text if we want to display the weight
    # (that is, if the weight is not filtered out)
    weight_label = '' if int(weight) in filtered_weights else ':' + weight
    # The actual label text
    label = weight_label if type_ == 'data' else f'{type_}{weight_label}'
    # The dot syntax to produce the actual label
    label_section = f'label="{label}"'

    attr_section = f' [{label_section}]' if label else ''
    # Add the edge
    result.append(f'    n{from_} -> n{to}{attr_section};\n')

# Graph is now finished:
result.append('}\n')

output = sys.stdout
if args.output: output = open(args.output, 'w')

print(''.join(result), file=output)
