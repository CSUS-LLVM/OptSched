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

def produce(s, cont=identity):
    return lambda _: cont(s)

discard = produce(None)

TR_TABLE = {
    'StaticLowerBoundDebugInfo': format('DAG {name} spillCostLB {spill_cost_lb} scFactor {sc_factor} lengthLB {length_lb} lenFactor {len_factor} staticLB {static_lb}'),
    'Enumerating': format('Enumerating at target length {target_length}'),
    'ScheduleVerifiedSuccessfully': discard, # produce('Schedule verified successfully'),
    'ProcessDag': discard, # format('Processing DAG {name} with {num_instructions} insts and max latency {max_latency}.'),
    'HeuristicResult': discard, # format('The list schedule is of length {length} and spill cost {spill_cost}. Tot cost = {cost}'),
    'CostLowerBound': discard, # format('Lower bound of cost before scheduling: {cost}'),
    'BypassZeroTimeLimit': discard, # format('Bypassing optimal scheduling due to zero time limit with cost {cost}'),
    'HeuristicScheduleOptimal': format('The initial schedule of length {length} and cost {cost} is optimal.'),
    'BestResult': discard,
        # translate_args(
        #     format('Best schedule for DAG {name} has cost {cost} and length {length}. The schedule is {optimal}'),
        #     optimal=lambda opt: 'optimal' if opt else 'not optimal'
        # ),
    'SlilStats': discard, # format('SLIL stats: DAG {name} static LB {static_lb} gap size {gap_size} enumerated {is_enumerated} optimal {is_optimal} PERP higher {is_perp_higher}'),
    'NodeExamineCount': format('Examined {num_nodes} nodes.'),
    'DagSolvedOptimally': discard, # format('DAG solved optimally in {solution_time} ms with length={length}, spill cost = {spill_cost}, tot cost = {total_cost}, cost imp={cost_improvement}.'),
    'DagTimedOut': format('DAG timed out with length={length}, spill cost = {spill_cost}, tot cost = {total_cost}, cost imp={cost_improvement}.'),
    'HeuristicLocalRegAllocSimulation': format(
        dedent('''\
        OPT_SCHED LOCAL RA: DAG Name: {dag_name} ***heuristic_schedule*** Number of spills: {num_spills}
        Number of stores {num_stores}
        Number of loads {num_loads}''')),
    'BestLocalRegAllocSimulation': format(
        dedent('''\
        OPT_SCHED LOCAL RA: DAG Name: {dag_name} Number of spills: {num_spills}
        Number of stores {num_stores}
        Number of loads {num_loads}''')),
    'LocalRegAllocSimulationChoice': discard,
    'PassFinished': discard, # lambda log: ('End of first pass through\n', 'End of second pass through')[log['num']],
}


def translate(infile, outfile):
    lines = infile.readlines()
    for line in lines:
        if not line.startswith('EVENT:'):
            outfile.write(line)
            continue

        parsed = json.loads(line.split(' ', 1)[1])
        tr = TR_TABLE[parsed['event_id']]
        print(tr(parsed), file=outfile)


for inpath, outpath in zip(args.input, args.output):
    with open(inpath, 'r') as infile,\
        open(outpath, 'w') as outfile:
        translate(infile, outfile)
