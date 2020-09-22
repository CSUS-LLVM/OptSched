#!/usr/bin/env python3
import argparse
import sys
import json
from textwrap import dedent

parser = argparse.ArgumentParser(description='Convert data_dep WriteToFile format to a .dot file')
parser.add_argument('-i', '--input', required=True, nargs='+', help='The WriteToFile format file to convert.')
parser.add_argument('-o', '--output', required=True, nargs='+', help='The destination to write to.')

args = parser.parse_args()
if len(args.input) != len(args.output):
    parser.error(f'Differing number of inputs and outputs: {args.input} vs {args.output}')

identity = lambda x: x

def translate_args(cont, **kwargs):
    def translate(log):
        for name, f in kwargs.items():
            log[name] = f(log[name])

        return cont(log)
    return translate

def format(s, cont=identity):
    return lambda log: cont(s.format(**log))

discard = lambda _: None

TR_TABLE = {
    'StaticLowerBoundDebugInfo': format('INFO: DAG {name} spillCostLB {spill_cost_lb} scFactor {sc_factor} lengthLB {length_lb} lenFactor {len_factor} staticLB {static_lb} (Time = {time} ms)'),
    'Enumerating': format('INFO: Enumerating at target length {target_length} (Time = {time} ms)'),
    'ScheduleVerifiedSuccessfully': discard, # format('INFO: Schedule verified successfully (Time = {time} ms)'),
    'ProcessDag': discard, # format('INFO: Processing DAG {name} with {num_instructions} insts and max latency {max_latency}. (Time = {time} ms)'),
    'HeuristicResult': discard, # format('INFO: The list schedule is of length {length} and spill cost {spill_cost}. Tot cost = {cost} (Time = {time} ms)'),
    'CostLowerBound': discard, # format('INFO: Lower bound of cost before scheduling: {cost} (Time = {time} ms)'),
    'BypassZeroTimeLimit': discard, # format('INFO: Bypassing optimal scheduling due to zero time limit with cost {cost} (Time = {time} ms)'),
    'HeuristicScheduleOptimal': format('INFO: The initial schedule of length {length} and cost {cost} is optimal. (Time = {time} ms)'),
    'BestResult': discard,
        # translate_args(
        #     format('INFO: Best schedule for DAG {name} has cost {cost} and length {length}. The schedule is {optimal} (Time = {time} ms)'),
        #     optimal=lambda opt: 'optimal' if opt else 'not optimal'
        # ),
    'SlilStats': discard, # format('INFO: SLIL stats: DAG {name} static LB {static_lb} gap size {gap_size} enumerated {is_enumerated} optimal {is_optimal} PERP higher {is_perp_higher} (Time = {time} ms)'),
    'NodeExamineCount': format('INFO: Examined {num_nodes} nodes. (Time = {time} ms)'),
    'DagSolvedOptimally': discard, # format('INFO: DAG solved optimally in {solution_time} ms with length={length}, spill cost = {spill_cost}, tot cost = {total_cost}, cost imp={cost_improvement}. (Time = {time} ms)'),
    'DagTimedOut': format('INFO: DAG timed out with length={length}, spill cost = {spill_cost}, tot cost = {total_cost}, cost imp={cost_improvement}. (Time = {time} ms)'),
    'HeuristicLocalRegAllocSimulation': format(
        dedent('''\
        INFO: OPT_SCHED LOCAL RA: DAG Name: {dag_name} ***heuristic_schedule*** Number of spills: {num_spills} (Time = {time} ms)
        INFO: Number of stores {num_stores} (Time = {time} ms)
        INFO: Number of loads {num_loads} (Time = {time} ms)''')),
    'BestLocalRegAllocSimulation': format(
        dedent('''\
        INFO: OPT_SCHED LOCAL RA: DAG Name: {dag_name} Number of spills: {num_spills} (Time = {time} ms)
        INFO: Number of stores {num_stores} (Time = {time} ms)
        INFO: Number of loads {num_loads} (Time = {time} ms)''')),
    'LocalRegAllocSimulationChoice': discard,
    'PassFinished': discard, # lambda log: ('INFO: End of first pass through\n (Time = {time} ms)', 'INFO: End of second pass through (Time = {time} ms)')[log['num'] - 1].format(time=log['time']),
}


def translate(infile, outfile):
    lines = infile.readlines()
    for line in lines:
        if not line.startswith('EVENT:'):
            outfile.write(line)
            continue

        parsed = json.loads(line.split(' ', 1)[1])
        tr = TR_TABLE[parsed['event_id']]
        result = tr(parsed)
        if result is not None:
            print(result, file=outfile)


for inpath, outpath in zip(args.input, args.output):
    with open(inpath, 'r') as infile,\
        open(outpath, 'w') as outfile:
        translate(infile, outfile)
