# /usr/bin/python3

import sys
import itertools
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from textwrap import dedent
import argparse

import analyze
from analyze import Logs


@dataclass
class DagInfo:
    id: str
    benchmark: str
    num_instructions: int
    pass_num: int

    lower_bound: int
    relative_cost: int
    length: int
    is_optimal: bool

    @property
    def cost(self):
        return self.lower_bound + self.relative_cost


class MismatchKind(Enum):
    BOTH_OPTIMAL_BUT_UNEQUAL = 0
    FIRST_OPTIMAL_BUT_WORSE = 1
    SECOND_OPTIMAL_BUT_WORSE = 2


@dataclass
class Mismatch:
    # The dag id: function name + basic block number
    dag_id: str
    # Which benchmark this region comes from
    benchmark: str
    # The number of instructions for this region
    region_size: int

    # The cost information indexed by "first" == 0, "second" == 1
    lengths: Tuple[int, int]
    costs: Tuple[int, int]

    kind: MismatchKind


@dataclass
class ValidationInfo:
    num_regions_first: int = 0
    num_regions_second: int = 0

    num_optimal_both: int = 0
    num_optimal_first: int = 0
    num_optimal_second: int = 0

    num_mismatch: Dict[MismatchKind, int] = field(default_factory=lambda: defaultdict(lambda: 0))

    mismatches: List[Mismatch] = field(default_factory=list)


# Explain this many of the blocks missing a lower bound
MISSING_LOWER_BOUND_DUMP_COUNT = 3
MISSING_LOWER_BOUND_DUMP_LINES = 10

# If there is no PassFinished, what pass "number" should we consider this to be?
DEFAULT_PASS = [{'num': 0}]


def split_adjacent(iterable, adj_eq=None):
    '''
    Splits the iterable into regions of "equal values" as specified by adj_eq.

    Examples:
        split_adjacent([1, 1, 1, 2, 2, 2, 2, 2, 3, 5]) # -> [(1, 1, 1), (2, 2, 2, 2, 2), (3,), (5,)]
        split_adjacent([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 8, 9], lambda x, y: x <= y)
            # -> [(1, 2, 3, 4), (1, 2, 3, 4), (2, 3, 8, 9)]
    '''
    if adj_eq is None:
        # Default: use ==
        def adj_eq(x, y): return x == y

    values = []
    for x in iterable:
        if values and not adj_eq(values[-1], x):
            yield tuple(values)
            values.clear()
        values.append(x)

    assert len(values) > 0
    yield tuple(values)


def pass_num(block) -> int:
    return block.get('PassFinished', DEFAULT_PASS)[0]['num']


def extract_dag_info(logs: Logs) -> Dict[str, List[List[DagInfo]]]:
    dags = {}

    blocks = list(logs)

    no_lb = [block for block in blocks if 'CostLowerBound' not in block]

    if no_lb:
        print('WARNING: Missing a logged lower bound for {missing}/{total} blocks.'
              .format(missing=len(no_lb), total=len(blocks)), file=sys.stderr)

        trimmed = ('\n'.join(block.raw_log.splitlines()[:MISSING_LOWER_BOUND_DUMP_LINES]) for block in no_lb)

        for i, block in enumerate(itertools.islice(trimmed, MISSING_LOWER_BOUND_DUMP_COUNT)):
            print('WARNING: block {} missing lower-bound:\n{}\n...'.format(i, block),
                  file=sys.stderr)

    for block in blocks:
        try:
            best_result = block.single('BestResult')
            is_optimal = best_result['optimal']
        except KeyError:
            try:
                best_result = block['HeuristicResult'][-1]
                is_optimal = best_result['cost'] == 0 or \
                    'INFO: Marking SLIL list schedule as optimal due to zero PERP.' in block.raw_log
            except KeyError:
                print('ERROR: unable to extract BestResult or HeuristicResult from block', file=sys.stderr)
                print(block.raw_log)
                exit(2)

        dags.setdefault(block.name, []).append(DagInfo(
            id=block.name,
            benchmark=block.benchmark,
            num_instructions=block.single('ProcessDag')['num_instructions'],
            pass_num=pass_num(block),
            lower_bound=block['CostLowerBound'][-1]['cost'],
            relative_cost=best_result['cost'],
            length=best_result['length'],
            is_optimal=is_optimal,
        ))

    for k, block_passes in dags.items():
        # Safe to modify dags while iterating because we use .items() to get a copy
        dags[k] = list(map(list, split_adjacent(block_passes, lambda x, y: x.pass_num < y.pass_num)))

    return dags


def parse_mismatch(blk1: DagInfo, blk2: DagInfo) -> Optional[Mismatch]:
    mismatch = Mismatch(
        dag_id=blk1.id,
        benchmark=blk1.benchmark,
        region_size=blk1.num_instructions,
        lengths=(blk1.length, blk2.length),
        costs=(blk1.cost, blk2.cost),
        kind=None,
    )
    if blk1.is_optimal and blk2.is_optimal:
        mismatch.kind = MismatchKind.BOTH_OPTIMAL_BUT_UNEQUAL
        return mismatch if blk1.cost != blk2.cost else None
    elif blk1.is_optimal:
        mismatch.kind = MismatchKind.FIRST_OPTIMAL_BUT_WORSE
        return mismatch if blk1.cost > blk2.cost else None
    elif blk2.is_optimal:
        mismatch.kind = MismatchKind.SECOND_OPTIMAL_BUT_WORSE
        return mismatch if blk2.cost > blk1.cost else None
    else:
        return None


def classify_optimal(out: ValidationInfo, blk1: DagInfo, blk2: DagInfo):
    if blk1.is_optimal and blk2.is_optimal:
        out.num_optimal_both += 1
    elif blk1.is_optimal:
        out.num_optimal_first += 1
    elif blk2.is_optimal:
        out.num_optimal_second += 1


def classify_mismatch(out: ValidationInfo, mismatch: Mismatch):
    out.num_mismatch[mismatch.kind] += 1


def validate_dags(dags1: Dict[str, List[List[DagInfo]]], dags2: Dict[str, List[List[DagInfo]]]) -> ValidationInfo:
    result = ValidationInfo(num_regions_first=len(dags1), num_regions_second=len(dags2))

    for region_f, region_s in zip(dags1.items(), dags2.items()):
        name_f, grouped_blocks_f = region_f
        name_s, grouped_blocks_s = region_s

        for blocks_f, blocks_s in zip(grouped_blocks_f, grouped_blocks_s):
            # blocks_* is the groups of blocks referring to the same problem, with different pass nums.
            blocks = list(zip(blocks_f, blocks_s))

            block_f, block_s = blocks[0]
            classify_optimal(result, block_f, block_s)

            mismatch = parse_mismatch(block_f, block_s)
            if mismatch is not None:
                classify_mismatch(result, mismatch)
                result.mismatches.append(mismatch)

            for next_block_f, next_block_s in blocks[1:]:
                if not block_f.is_optimal:
                    next_block_f.is_optimal = False
                if not block_s.is_optimal:
                    next_block_s.is_optimal = False

                classify_optimal(result, next_block_f, next_block_s)
                mismatch = parse_mismatch(next_block_f, next_block_s)
                if mismatch is not None:
                    classify_mismatch(result, mismatch)
                    result.mismatches.append(mismatch)

                block_f, block_s = next_block_f, next_block_s

    return result


def print_mismatches(info: ValidationInfo,
                     print_stats_info: Callable[[ValidationInfo], None],
                     print_mismatch_info: Callable[[ValidationInfo], None],
                     print_mismatch_summaries: List[Callable[[List[Mismatch]], None]]):
    print_stats_info(info)
    print_mismatch_info(info)

    if info.mismatches:
        for print_summary in print_mismatch_summaries:
            print_summary(info.mismatches)


def enable_if(cond: bool):
    def wrapped(f):
        return f if cond else lambda *args: None

    return wrapped


# The quantity of blocks with the largest mismatches to print.
NUM_LARGEST_MISMATCHES_PRINT = 10
# The quantity of mismatched blocks with the shortest length to print.
NUM_SMALLEST_BLOCKS_PRINT = 50


def main(first, second,
         quiet: bool = False,
         summarize_biggest_cost_difference: bool = True,
         summarize_smallest_regions: bool = True):
    dags1 = extract_dag_info(first)
    dags2 = extract_dag_info(second)
    info: ValidationInfo = validate_dags(dags1, dags2)

    @enable_if(not quiet)
    def print_stats_info(info: ValidationInfo):
        print('Optimal Block Stats')
        print('-----------------------------------------------------------')
        print('Blocks in log file 1: ' + str(info.num_regions_first))
        print('Blocks in log file 2: ' + str(info.num_regions_second))
        print('Blocks that are optimal in both files: ' + str(info.num_optimal_both))
        print('Blocks that are optimal in log 1 but not in log 2: ' + str(info.num_optimal_first))
        print('Blocks that are optimal in log 2 but not in log 1: ' + str(info.num_optimal_second))
        print('----------------------------------------------------------\n')

    @enable_if(info.mismatches or not quiet)
    def print_mismatch_info(info: ValidationInfo):
        print('Mismatch stats')
        print('-----------------------------------------------------------')
        print('Mismatches where blocks are optimal in both logs but have different costs: ' +
              str(info.num_mismatch[MismatchKind.BOTH_OPTIMAL_BUT_UNEQUAL]))
        print('Mismatches where the block is optimal in log 1 but it has a higher cost than the non-optimal block in log 2: ' +
              str(info.num_mismatch[MismatchKind.FIRST_OPTIMAL_BUT_WORSE]))
        print('Mismatches where the block is optimal in log 2 but it has a higher cost than the non-optimal block in log 1: ' +
              str(info.num_mismatch[MismatchKind.SECOND_OPTIMAL_BUT_WORSE]))
        print('Total mismatches: ' + str(len(info.mismatches)))
        print('-----------------------------------------------------------\n')

    def print_block_info(index: int, mismatch: Mismatch):
        cost_diff = mismatch.costs[0] - mismatch.costs[1]
        print(dedent(f'''\
            {index}:
            Block Name: {mismatch.dag_id}
            Benchmark: {mismatch.benchmark}
            Num Instructions: {mismatch.region_size}
            Length: {mismatch.lengths[0]} --> {mismatch.lengths[1]}
            Difference in cost: {cost_diff}
            Percent cost difference: {(cost_diff / mismatch.costs[0])*100:0.2f} %
            '''
                     ))

    @enable_if(summarize_biggest_cost_difference)
    def print_big_diff_summary(mismatches: List[Mismatch]):
        if NUM_LARGEST_MISMATCHES_PRINT == 0:
            print('Requested 0 mismatched blocks with the largest difference in cost')
            return

        print('The ' + str(NUM_LARGEST_MISMATCHES_PRINT) + ' mismatched blocks with the largest difference in cost')
        print('-----------------------------------------------------------')
        sortedMaxMis = sorted(mismatches, key=lambda m: abs(m.costs[1] - m.costs[0]), reverse=True)
        for index, mismatch in enumerate(sortedMaxMis[:NUM_LARGEST_MISMATCHES_PRINT]):
            print_block_info(index, mismatch)
        print('-----------------------------------------------------------\n')

    @enable_if(summarize_smallest_regions)
    def print_small_summary(mismatches: List[Mismatch]):
        if NUM_SMALLEST_BLOCKS_PRINT == 0:
            print('Requested 0 mismatched blocks with the smallest block size')
            return

        print('The smallest ' + str(NUM_SMALLEST_BLOCKS_PRINT) + ' mismatched blocks')
        print('-----------------------------------------------------------')
        sortedMisSize = sorted(mismatches, key=lambda m: m.region_size)
        for index, mismatch in enumerate(sortedMisSize[:NUM_LARGEST_MISMATCHES_PRINT]):
            print_block_info(index, mismatch)
        print('-----------------------------------------------------------\n')

    print_mismatches(
        info,
        print_stats_info=print_stats_info,
        print_mismatch_info=print_mismatch_info,
        print_mismatch_summaries=[print_big_diff_summary, print_small_summary]
    )
    if info.mismatches:
        exit(f'{len(info.mismatches)} mismatches found')


if __name__ == "__main__":
    dags1 = {}
    dags2 = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('first')
    parser.add_argument('second')

    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Only print mismatch info, and only if there are mismatches')
    parser.add_argument('--no-summarize-largest-cost-difference', action='store_true',
                        help='Do not summarize the mismatches with the biggest difference in cost')
    parser.add_argument('--no-summarize-smallest-mismatches', action='store_true',
                        help='Do not summarize the mismatches with the smallest region size')

    parser.add_argument('--num-largest-cost-mismatches-print', type=int, default=10,
                        help='The number of mismatches blocks with the largest (by cost) mismatches to print')
    parser.add_argument('--num-smallest-mismatches-print', type=int, default=10,
                        help='The number of mismatched blocks with the shortest length to print')

    parser.add_argument('--missing-lb-dump-count', type=int, default=3,
                        help='The number of blocks with missing lower bounds to display')
    parser.add_argument('--missing-lb-dump-lines', type=int, default=10,
                        help='The number of lines of a block with missing lower bound to display')
    args = analyze.parse_args(parser, 'first', 'second')

    NUM_LARGEST_MISMATCHES_PRINT = args.num_largest_cost_mismatches_print
    NUM_SMALLEST_BLOCKS_PRINT = args.num_smallest_mismatches_print
    MISSING_LOWER_BOUND_DUMP_COUNT = args.missing_lb_dump_count
    MISSING_LOWER_BOUND_DUMP_LINES = args.missing_lb_dump_lines

    main(
        args.first, args.second,
        quiet=args.quiet,
        summarize_biggest_cost_difference=not args.no_summarize_largest_cost_difference,
        summarize_smallest_regions=not args.no_summarize_smallest_mismatches,
    )
