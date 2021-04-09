from ._types import *


def sum_dicts(ds):
    '''
    Sums ds[N]['Key'] for each key for each dict. Assumes each dict has the same keys
    E.g. sum_dicts({'a': 1, 'b': 2}, {'a': 2, 'b': 3}) produces {'a': 3, 'b': 5}
    '''
    if not ds:
        return {}
    return {k: sum(d[k] for d in ds) for k in ds[0].keys()}


def foreach_bench(analysis_f, *logs, combine=None):
    '''
    Repeats `analysis_f` for each benchmark in `logs`.
    Also computes the analysis for the entire thing.
    If `combine` is given, uses the function to combine it.
    Otherwise, runs `analysis_f` over the entire thing (takes quite some time)

    Returns:
        A dictionary containing the per-benchmark results.
        The keys are the benchmark names.
        The run for the entire thing has a key of 'Total'
    '''

    if combine is None:
        combine = lambda *args: analysis_f(*logs)

    benchmarks = zip(*[log.benchmarks for log in logs])

    bench_stats = {bench[0].name: analysis_f(*bench) for bench in benchmarks}
    total = combine(bench_stats.values())

    return {
        # Making a new dict so that the "Total" key can be first.
        'Total': total,
        **bench_stats,
    }


def count(iter):
    try:
        return len(iter)
    except:
        return sum(1 for _ in iter)


def zipped_keep_blocks_if(*logs, pred):
    '''
    Given:
      a: [blk1, blk2, blk3, ...] # of type Logs
      b: [blk1, blk2, blk3, ...] # of type Logs
      c: [blk1, blk2, blk3, ...] # of type Logs
      ...

    Returns:
      [
          (a.blk1, b.blk1, c.blk1, ...) if pred(a.blk1, b.blk1, c.blk1, ...)
          ...
      ]

    Also supports pred(b), in which case it's all(pred(b) for b in (a.blk1, b.blk1, ...))
    '''

    try:
        all_p = set(blks[0].uniqueid() for blks in zip(*logs) if pred(*blks))
    except TypeError:
        all_p = set(blks[0].uniqueid()
                    for blks in zip(*logs) if all(pred(b) for b in blks))

    filtered = tuple(log.keep_blocks_if(
        lambda blk: blk.uniqueid() in all_p) for log in logs)

    return filtered


def sum_stat_for_all(stat, logs: Logs) -> int:
    return sum(stat(blk) for blk in logs)
