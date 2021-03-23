#!/usr/bin/env python3

import re

import analyze
from analyze import Block, Logs


def _block_time(block: Block):
    end = block.single('ScheduleVerifiedSuccessfully')['time']
    start = block.single('ProcessDag')['time']
    return end - start


@analyze.analyzer.analyzer
def instruction_scheduling_time(logs):
    return sum(_block_time(blk) for blk in logs)


@analyze.analyzer.analyzer
def total_compile_time_seconds(logs):
    last_logs = logs.benchmarks[-1].blocks[-1].raw_log
    m = re.search(r'(\d+) total seconds elapsed', last_logs)
    assert m, \
        'Logs must contain "total seconds elapsed" output by the SPEC benchmark suite'

    return m.group(1)


if __name__ == '__main__':
    @analyze.analyzer.analyzer(variant={
        'help': 'Which variant of timing to use. Valid options: total, sched',
        'choices': ('sched', 'total'),
    })
    def switched_time(logs, /, *, variant):
        if variant == 'total':
            return total_compile_time_seconds(logs)
        else:
            return instruction_scheduling_time(logs)

    print(switched_time.main())
