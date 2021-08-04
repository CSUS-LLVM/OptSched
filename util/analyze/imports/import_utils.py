import itertools
import json
import pickle
import re
import sys
from dataclasses import dataclass
from typing import List, Match, Optional, Pattern, Union

from .._types import Benchmark, Block, Logs

_REGION_DELIMITER = 'INFO: ********** Opt Scheduling **********'
_RE_REGION_DELIMITER = re.compile(re.escape(_REGION_DELIMITER))


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


def parse_multi_bench_file(logtext: str, *, benchstart: Union[Pattern, str], filename: Optional[Union[Pattern, str]] = None):
    if filename is not None:
        filename = re.compile(filename)
    benchstart = re.compile(benchstart)

    def parse_bench(benchm: Match, nextm: Union[Match, _DummyEnd], is_first: bool = False):
        # The RE can specify any extra properties.
        info = benchm.groupdict()
        # If this is the first benchmark in the file, we want to start from the
        # start of the file so that we don't lose any information.
        start = 0 if is_first else benchm.start()
        end = nextm.start()
        return _parse_benchmark(info, logtext,
                                start, end,
                                filenamere=filename)

    bench_matches = list(benchstart.finditer(logtext))
    benchmarks = []

    is_first: bool = True
    for benchm, nextm in zip(
            bench_matches,
            [*bench_matches[1:], _DummyEnd(len(logtext))]
    ):
        benchmarks.append(parse_bench(benchm, nextm, is_first))
        is_first = False

    return Logs(benchmarks)


def parse_single_bench_file(logtext, *, benchname, filename: Optional[Union[Pattern, str]] = None):
    if filename is not None:
        filename = re.compile(filename)
    return Logs([
        _parse_benchmark(
            {'name': benchname},
            logtext, 0, len(logtext),
            filenamere=filename,
        )
    ])


@dataclass
class _FileInfo:
    filename: Optional[str]
    from_pos: int


class _DummyEnd:
    def __init__(self, length):
        self._end = length - 1

    def start(self):
        return self._end

    def end(self):
        return self._end


def _filename_info(filenamere: Optional[Pattern], logtext: str, start: int, end: int) -> List[_FileInfo]:
    if filenamere is None:
        filenamere = re.compile(r'.^')  # RE that doesn't match anything
    files = []

    for filem in filenamere.finditer(logtext, start, end):
        filename = filem.group(1)
        filestart = filem.end()
        files.append(_FileInfo(filename=filename, from_pos=filestart))

    return files


def _parse_benchmark(info: dict, logtext: str, start: int, end: int, *, filenamere: Optional[Pattern]):
    BENCHNAME = info['name']

    blocks = []

    files: List[_FileInfo] = _filename_info(filenamere, logtext, start, end)
    if not files:
        # We have an unknown file starting from the very beginning
        files = [_FileInfo(filename=None, from_pos=start)]

    # Allow us to peek ahead by giving a dummy "file" at the end which will never match a block
    files.append(_FileInfo(filename=None, from_pos=end))
    assert len(files) >= 2
    file_pos = 0

    block_matches1, block_matches2 = itertools.tee(_RE_REGION_DELIMITER.finditer(logtext, start, end))
    next(block_matches2)  # Drop first
    block_matches2 = itertools.chain(block_matches2, (_DummyEnd(end),))

    blocks = []

    is_first = True
    for regionm, nextm in zip(block_matches1, block_matches2):
        region_start = regionm.end()
        if region_start > files[file_pos + 1].from_pos:
            file_pos += 1

        assert region_start > files[file_pos].from_pos

        filename = files[file_pos].filename if files[file_pos] else None

        regioninfo = {
            'file': filename,
            'benchmark': BENCHNAME,
        }
        blk_start = start if is_first else regionm.start()
        blk_end = nextm.start()
        blocks.append(_parse_block(regioninfo, logtext,
                                   blk_start, blk_end))
        is_first = False

    return Benchmark(info, blocks)


def _parse_block(info, logtext: str, start, end):
    events = _parse_events(logtext, start, end)
    raw_log = logtext[start:end]
    assert 'ProcessDag' in events
    info['name'] = events['ProcessDag'][0]['name']

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

    result = dict()

    for log in parsed:
        result.setdefault(log['event_id'], []).append(log)

    return result
