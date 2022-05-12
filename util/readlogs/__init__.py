import json
import os
import re

# Ignore these functions on the AMDGPU
# They are outputted before scheduling
OPT_IGNORE = [
    'copyBufferRect',
    'copyBufferRectAligned',
    'copyBuffer',
    'copyBufferAligned',
    'fillBuffer',
    'copyBufferToImage',
    'copyImageToBuffer',
    'copyImage',
    'copyImage1DA',
    'fillImage',
    'scheduler'
]

OPT_RE_OCCUPANCY = re.compile('Final occupancy for function (.*):(\d+)')
OPT_RE_REVERT_SCHED = re.compile(
    r'Reverting Scheduling because of a decrease in occupancy from')

def get_bench_log_paths(inputFolder, benchmark):
    '''
    Returns a `dict[benchmark --> path/to/benchmark.log` mapping a benchmark to its corresponding file.

    Parameters:
    inputFolder -- A string containing the name of a directory with plaidbench or SHOC results.
    benchmark - Commandline argument indicating the benchmark (shoc, plaid).
    '''

    OPT_SHOC_BENCHMARKS = [
        # 'BusSpeedDownload',
        # 'BusSpeedReadback',
        # 'MaxFlops',
        # 'DeviceMemory',
        # 'KernelCompile',
        # 'QueueDelay',
        # 'BFS',
        'FFT',
        'GEMM',
        'MD',
        # 'MD5Hash',
        # 'Reduction',
        # 'Scan',
        'Sort',
        'Spmv',
        'Stencil2D',
        # 'Triad',
        # 'S3D'
    ]

    OPT_PLAID_BENCHMARKS = [
        'densenet121',
        'densenet169',
        'densenet201',
        'inception_resnet_v2',
        'inception_v3',
        'mobilenet',
        'nasnet_large',
        'nasnet_mobile',
        'resnet50',
        'vgg16',
        'vgg19',
        'xception',
        'imdb_lstm',
    ]

    filepaths = {}

    # Do a lowercase string comparison to determine the benchmark set
    bench = benchmark.lower()

    # Paths for shoc benchmarks
    if bench == 'shoc':
        logDirectory = os.path.join(inputFolder, 'Logs')
        for bench in OPT_SHOC_BENCHMARKS:
            filename = 'dev0_{}.err'.format(bench)
            filepath = os.path.join(logDirectory, filename)
            filepaths[bench] = filepath

    # Paths for PlaidML benchmarks
    elif bench == 'plaid':
        for bench in OPT_PLAID_BENCHMARKS:
            benchmarkDirectory = os.path.join(inputFolder, bench)
            filename = '{}.log'.format(bench)
            filepath = os.path.join(benchmarkDirectory, filename)
            filepaths[bench] = filepath

    return filepaths

def split_blocks(log):
    '''
    Splits the log into the individual blocks.
    '''
    return log.split("INFO: ********** Opt Scheduling **********")[1:]

def parse_events(block_log):
    '''
    Returns a `dict[event_id --> list[event-json]]` of the events in the given log.

    `EVENT: {"event_id": "some_id", "value"}`
    becomes `{"some_id": [{"event_id": "some_id", "arg": "value"}, ...], ...}`

    If there is only one event of each id, pass the result through
    `parse_as_singular_events(...)` to unwrap the lists.
    '''
    lines = block_log.splitlines()
    event_lines = [line.split(' ', 1)[1] for line in lines if line.startswith('EVENT:')]
    parsed = list(map(json.loads, event_lines))
    result = dict()

    for log in parsed:
        result.setdefault(log['event_id'], []).append(log)

    return result

def parse_blocks(log):
    '''
    Splits the block into individual blocks and parses each block via parse_events().
    '''
    return [parse_events(block) for block in split_blocks(log)]

def keep_only_singular_events(logs):
    '''
    Converts a the event `dict[event_id --> list[event-json]]` to
    `dict[event_id --> event-json]` dropping any event which has a duplicated event_id.
    '''
    result = dict()
    for k, v in logs.items():
        if len(v) == 1: result[k] = v[0]
    return result

def keep_only_first_event(logs):
    '''
    Converts a the event `dict[event_id --> list[event-json]]` to
    `dict[event_id --> event-json]` keeping only the first of any event for a given event_id.
    '''
    result = dict()
    for k, v in logs.items():
        result[k] = v[0]
    return result

def parse_as_singular_events(logs):
    '''
    Converts a the event `dict[event_id --> list[event-json]]` to
    `dict[event_id --> event-json]` requiring exactly one event per event_id.
    '''
    for k, v in logs.items():
        if len(v) != 1: raise AssertionError('Duplicate log events for event ' + k)
    return {k: v[0] for k, v in logs.items()}


def getPercentageString(num, dem):
    '''
    Return string with percentage
    '''
    if dem == 0:
        return '0 (0.00%)'

    formattedPcnt = num / dem * 100
    return '{} ({:.2f}%)'.format(num, formattedPcnt)
