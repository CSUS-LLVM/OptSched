import pickle
import json
import itertools
import re
from collections import namedtuple

from analyze.logs import Logs, Benchmark, Block

_RE_REGION_INFO = re.compile(r'EVENT:.*ProcessDag.*"name": "(?P<name>[^"]*)"')


def import_main(parsefn, *, description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-o', '--output', required=True, help='The output file to write the imported log format to')
    parser.add_argument('input', help='The input logs to process')
    args = parser.parse_args()

    result = parsefn(args.input)

    with open(args.output, 'wb') as f:
        pickle.dump(result, f)


def parse_multi_bench_file(logtext, *, benchstart, filename=None):
    benchmarks = []
    for benchm, nextm in _splititer(benchstart, logtext):
        bench = _parse_benchmark(benchm.groupdict(), logtext,
                                 benchm.end(), nextm.start(),
                                 filenamere=filename)
        benchmarks.append(bench)

    return Logs(benchmarks)


_FileInfo = namedtuple('_FileInfo', ('filename', 'from_pos'))


def _each_cons(iterable, n):
    '''
    Iterates over each consecutive n items of the iterable.

    _each_cons((1, 2, 3, 4), 2) # (1, 2), (2, 3), (3, 4)
    '''
    iters = [None] * n
    iters[0] = iter(iterable)
    for i in range(1, n):
        iters[i - 1], iters[i] = itertools.tee(iters[i - 1])
        next(iters[i], None)
    return zip(*iters)


class _DummyEnd:
    def __init__(self, length):
        self._end = length - 1

    def start(self):
        return self._end

    def end(self):
        return self._end


def _splititer(regex, text, pos=0, endpos=None):
    '''
    'Splits' the string by the regular expression, using an iterable.
    Returns both where the regex matches and where it matched next (or the end).
    '''
    if endpos is None:
        endpos = len(text) - 1

    return _each_cons(
        itertools.chain(regex.finditer(text, pos, endpos),
                        (_DummyEnd(endpos + 1),)),
        2
    )


def _parse_benchmark(info, logtext: str, start, end, *, filenamere):
    NAME = info['name']

    blocks = []

    if filenamere and filenamere.search(logtext, start, end):
        files = [
            *(_FileInfo(filename=r.group(1), from_pos=r.end())
              for r in filenamere.finditer(logtext, start, end)),
            _FileInfo(filename=None, from_pos=len(logtext)),
        ][::-1]
    else:
        files = [
            _FileInfo(filename=None, from_pos=start),
            _FileInfo(filename=None, from_pos=len(logtext)),
        ][::-1]

    blocks = []

    for regionm, nextm in _splititer(_RE_REGION_INFO, logtext, start, end):
        assert regionm.end() > files[-1].from_pos
        if regionm.end() > files[-2].from_pos:
            files.pop()

        try:
            filename = files[-1].filename
        except NameError:
            filename = None

        regioninfo = {
            'name': regionm['name'],
            'file': filename,
            'benchmark': NAME,
        }
        block = _parse_block(regioninfo, logtext,
                             regionm.start() - 1, nextm.start())
        blocks.append(block)

    return Benchmark(info, blocks)


def _parse_block(info, logtext: str, start, end):
    events = _parse_events(logtext, start, end)
    raw_log = logtext[start:end]

    return Block(info, raw_log, events)


_RE_EVENT_LINE = re.compile(r'\nEVENT: (.*)')


def _parse_events(block_log, start=0, end=None):
    '''
    Returns a `dict[event_id --> list[event-json]]` of the events in the given log.

    `EVENT: {"event_id": "some_id", "value"}`
    becomes `{"some_id": [{"event_id": "some_id", "arg": "value"}, ...], ...}`

    If there is only one event of each id, pass the result through
    `parse_as_singular_events(...)` to unwrap the lists.
    '''
    if end is None:
        end = len(block_log)

    event_lines = _RE_EVENT_LINE.findall(block_log, start, end)
    events = '[' + ',\n'.join(event_lines) + ']'

    try:
        parsed = json.loads(events)
    except json.JSONDecodeError:
        print(events, file=sys.stderr)
        raise
