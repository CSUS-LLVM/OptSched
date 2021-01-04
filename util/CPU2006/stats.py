#!/usr/bin/env python3

import re
import itertools

import analyze


TIMES_REGEX = re.compile(r'(\d+) total seconds elapsed')


class SpillStats(analyze.Analyzer):
    '''
    Computes SPEC-specific spill-cost stats
    '''

    def run_bench(self, args):
        logs = args[0]

        spills = SpillStats._calculate_spills(logs.all_blocks())
        weighted_spills = SpillStats._calculate_weighted_spills(
            logs.all_blocks())

        self.stat(
            spills=sum(x for x in spills.values()),
            weighted_spills=sum(x for x in weighted_spills.values()),
        )

    @staticmethod
    def _calculate_spills(blocks):
        # SPILLS_REGEX = re.compile(r'Function: (.*?)\nEND FAST RA: Number of spills: (\d+)\n')
        SPILLS_REGEX = re.compile(
            r'Function: (.*?)\nGREEDY RA: Number of spilled live ranges: (\d+)')
        # SPILLS_REGEX = re.compile(r'Function: (.*?)\nTotal Simulated Spills: (\d+)')

        spills = {}

        for function, spill_cost in itertools.chain(
                *[SPILLS_REGEX.findall(blk.raw_log) for blk in blocks]):
            spills[function] = int(spill_cost)

        return spills

    @staticmethod
    def _calculate_weighted_spills(blocks):
        SPILLS_WEIGHTED_REGEX = re.compile(r'SC in Function (.*?) (-?\d+)')

        weighted_spills = {}

        for function, spill_cost in itertools.chain(
                *[SPILLS_WEIGHTED_REGEX.findall(blk.raw_log) for blk in blocks]):
            weighted_spills[function] = int(spill_cost)

        return weighted_spills


class BlockStats(analyze.Analyzer):
    '''
    Collects block-level stats
    '''

    def run_bench(self, args):
        logs = args[0]

        num_enumerated = sum(1 for blk in logs.all_blocks()
                             if 'Enumerating' in blk)
        num_optimal = sum(1 for blk in logs.all_blocks()
                          if 'DagSolvedOptimally' in blk)
        num_timed_out = sum(1 for blk in logs.all_blocks()
                            if 'DagTimedOut' in blk)
        assert num_timed_out == num_enumerated - num_optimal

        lb_cost = sum(
            blk.single('CostLowerBound')['cost'] for blk in logs.all_blocks())
        heuristic_cost = lb_cost + sum(
            blk.single('HeuristicResult')['cost'] for blk in logs.all_blocks())

        def get_cost_improvement(blk: analyze.Block):
            if 'DagSolvedOptimally' in blk:
                return blk.single('DagSolvedOptimally')['cost_improvement']
            elif 'DagTimedOut' in blk:
                return blk.single('DagTimedOut')['cost_improvement']
            else:
                return 0

        cost_improvement = sum(get_cost_improvement(blk)
                               for blk in logs.all_blocks())

        self.stat(
            num_enumerated=num_enumerated,
            num_optimal=num_optimal,
            num_timed_out=num_enumerated - num_optimal,
            heuristic_cost=heuristic_cost,
            cost_improvement=cost_improvement,
        )


if __name__ == '__main__':
    from analyze.script import *
    run(SpillStats, _0, benchsuite='spec')
    run(BlockStats, _0, benchsuite='spec')
