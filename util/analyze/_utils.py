from ._types import *


def foreach_bench(analysis_f, *logs):
    '''
    Repeats `analysis_f` for each benchmark in `logs`.
    Also runs it for the entire thing.

    Returns:
        A dictionary containing the per-benchmark results.
        The keys are the benchmark names.
        The run for the entire thing has a key of 'Total'
    '''

    benchmarks = zip(*[log.benchmarks for log in logs])

    return {
        'Total': analysis_f(*logs),
        **{bench[0].name: analysis_f(*bench) for bench in benchmarks}
    }
