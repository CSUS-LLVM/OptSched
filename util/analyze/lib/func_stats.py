#!/usr/bin/env python3

import re
from itertools import chain
from typing import Iterable, List

from analyze import Block

'''
Function-level stats (not Block, Logs, or Benchmark level)
'''

_RE_OCCUPANCY = re.compile(r'Final occupancy for function (?P<name>\S+):(?P<value>\d+)')


def _occupancy_info_in_block_log(block: Block) -> Iterable[int]:
    for m in _RE_OCCUPANCY.finditer(block.raw_log):
        yield int(m['value'])


def function_occupancy_info(logs: Iterable[Block]) -> List[int]:
    return list(chain.from_iterable(map(_occupancy_info_in_block_log, logs)))


def avg_occupancy(logs: Iterable[Block]) -> float:
    occ_info = function_occupancy_info(logs)
    return sum(occ_info) / len(occ_info) if occ_info else 0.0
