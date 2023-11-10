#!/usr/bin/env python3

from typing import *
import argparse
import analyze
from analyze import Block, Logs


def nodes_examined(block: Block) -> int:
    return block.single('NodeExamineCount')['num_nodes'] if 'NodeExamineCount' in block else 0


def is_negative_nodes_examined(first: Block, second: Block) -> bool:
    return nodes_examined(first) < nodes_examined(second)


def find_negative_nodes_examined(first: Logs, second: Logs, percent_threshold: float = 0, absolute_threshold: float = 0) -> List[Tuple[Block, Block, int, int]]:
    return [
        (f, s, nodes_examined(f), nodes_examined(s)) for f, s in zip(first, second)
        if is_negative_nodes_examined(f, s)
        and nodes_examined(s) - nodes_examined(f) < absolute_threshold
        and (nodes_examined(s) - nodes_examined(f)) / nodes_examined(f) * 100 < percent_threshold
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finds all blocks for which nodes_examined(first_logs) < nodes_examined(second_logs)')
    parser.add_argument('first', help='The first logs')
    parser.add_argument('second', help='The second logs')
    parser.add_argument('-%', '--percent-threshold', type=float, default=0,
                        help='Ignore any blocks with a %%-difference < threshold')
    parser.add_argument('-$', '--absolute-threshold', type=float, default=0,
                        help='Ignore any blocks with a difference < threshold')
    args = analyze.parse_args(parser, 'first', 'second')

    negatives = find_negative_nodes_examined(
        args.first, args.second, percent_threshold=args.percent_threshold, absolute_threshold=args.absolute_threshold)
    negatives = sorted(negatives, key=lambda x: x[3] - x[2])
    for fblock, sblock, num_f, num_s in negatives:
        print(
            f"{fblock.info['benchmark']} {fblock.name} : {num_f} - {num_s} = {num_f - num_s}")
