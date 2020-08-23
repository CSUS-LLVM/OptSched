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
