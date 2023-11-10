#!/usr/bin/env python3
import argparse
import sys
import os
import json
from textwrap import dedent

parser = argparse.ArgumentParser(
    description='Convert JSON style logfiles to the older INFO only format',
    epilog=dedent('''\
    example usage:
        # Translate to output/a.log and output/dir/b.log:
        python3 json2infolog.py -i a.log dir/b.log -d output

        # Manually specify destinations (a.log -> a.tr.log, b.log -> b.tr.log):
        python3 json2infolog.py -i a.log dir/b.log -o a.tr.log b.tr.log
    '''),
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('-i', '--input', required=True, nargs='+',
                    help='The WriteToFile format file to convert.')
parser.add_argument('-o', '--output', nargs='+',
                    help='The destination to write to.')
parser.add_argument(
    '-d', '--outdir', help='The destination directory to write the output to. Writes each output to OUTDIR/INPUT')
parser.add_argument('-k', '--ignore-unknown', action='store_true', help='Ignores events with an unknown event_id.')

args = parser.parse_args()
if args.output is not None and args.outdir is not None:
    parser.error(
        f'Only one of --output and --outdir may be provided: --output {args.output}, --outdir {args.outdir}')

if args.outdir:
    args.output = [os.path.join(args.outdir, i) for i in args.input]

if len(args.input) != len(args.output):
    parser.error(
        f'Differing number of inputs and outputs: {args.input} vs {args.output}')


# Functions for defining translations:

def identity(line, x): return x


def translate_args(cont, **kwargs):
    def translate(line, log):
        for name, f in kwargs.items():
            log[name] = f(log[name])

        return cont(line, log)
    return translate


def format(s, cont=identity):
    return lambda line, log: cont(line, s.format(**log))


def discard(a, b): return None


def pass_through(line, log): return line


########
# Defines the translations for each event_id.
#
# Commented out translations that follow a `discard` are translations for logs
# which have a duplicate `Logger::Info(...)`` call following the
# `Logger::Event(...)`.
#
# If the `Logger::Info(...)` call is removed,
# the commented out code should replace the `discard`.
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

    'AcoPostSchedComplete': pass_through,
    'ACOSchedComplete': pass_through,
}


def translate(infile, outfile):
    lines = infile.readlines()
    for line in lines:
        if not line.startswith('EVENT:'):
            outfile.write(line)
            continue

        try:
            parsed = json.loads(line.split(' ', 1)[1])
        except json.decoder.JSONDecodeError as e:
            e.colno += len('EVENT:')
            print(f'Invalid JSON for line:\n{line}'
                  + ' ' * e.colno + '^\n', file=sys.stderr)
            raise

        event_id = parsed['event_id']

        if event_id not in TR_TABLE:
            print(f'Unknown event_id: `{event_id}`.', file=sys.stderr)

            if not args.ignore_unknown:
                print(dedent(f'''\
                    To temporarily ignore this error, pass -k or --ignore-unknown.

                    To fix this, add an entry to json2infolog.py's TR_TABLE:
                        '{event_id}': ...,
                    Example `...`s could be `pass_through` or `discard`.'''),
                    file=sys.stderr)
                exit()

        tr = TR_TABLE.get(event_id, pass_through)

        result = tr(
            line[:-1], # Remove the trailing newline
            parsed,
        )
        if result is not None:
            print(result, file=outfile)


for inpath, outpath in zip(args.input, args.output):
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)

    with open(inpath, 'r') as infile, \
            open(outpath, 'w') as outfile:
        translate(infile, outfile)
