import json

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
    return [parse_events(block) for block in log]

def parse_as_singular_events(logs):
    '''
    Converts a the event `dict[event_id --> list[event-json]]` to
    `dict[event_id --> event-json]` requiring exactly one event per event_id.
    '''
    for k, v in logs.items():
        if len(v) != 1: raise AssertionError('Duplicate log events for event ' + k)
    return {k: v[0] for k, v in logs.items()}
