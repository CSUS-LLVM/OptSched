class Logs:
    '''
    Abstracts a log file as a collection of benchmarks

    Properties:
     - logs.benchmarks: a list of the benchmarks this Logs contains.
    '''

    def __init__(self, benchmarks):
        self.benchmarks = benchmarks

    def merge(self, rhs):
        '''
        Merges the logs from the rhs into this.

        The rhs must have different benchmarks from this Logs
        '''
        in_both = set(self.benchmarks) & set(rhs.benchmarks)
        if in_both:
            raise ValueError(
                'Cannot merge Logs which share common benchmarks', in_both)

        self.benchmarks += rhs.benchmarks

        return self

    def benchmark(self, name):
        '''
        Gets the benchmark with the specified name
        '''
        for bench in self.benchmarks:
            if bench.name == name:
                return bench

        raise KeyError(f'No benchmark `{name}` in this Logs')

    def __iter__(self):
        '''
        Iterates over the blocks in every benchmark
        '''
        for bench in self.benchmarks:
            yield from bench.blocks

    def __len__(self):
        return sum(len(bench) for bench in self.benchmarks)

    def __repr__(self):
        benchmarks = ','.join(b.name for b in self.benchmarks)
        return f'<Logs({benchmarks})>'

    def keep_blocks_if(self, p):
        return Logs([bench.keep_blocks_if(p) for bench in self.benchmarks])

    def find_equiv(self, blk):
        uid = blk.uniqueid()
        return [b for b in self.benchmark(blk.info['benchmark']) if b.uniqueid() == uid]

    def find_block(self, name, benchmark=None):
        search = self
        if benchmark is not None:
            search = self.benchmark(benchmark)
        return [b for b in search if b.name == name]


class Benchmark:
    '''
    Abstracts a single benchmark in the logs as a collection of blocks

    Properties:
     - bench.name: the name of this benchmark
     - bench.info: miscellaneous information about this benchmark
     - bench.blocks: the Blocks in this benchmark
    '''

    def __init__(self, info, blocks):
        self.name = info['name']
        self.info = info
        self.blocks = blocks

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)

    @property
    def benchmarks(self):
        return (self,)

    def __repr__(self):
        return f'<Benchmark({self.info}, {len(self.blocks)} blocks)>'

    def keep_blocks_if(self, p):
        return Benchmark(self.info, [blk for blk in self.blocks if p(blk)])

    def find_equiv(self, blk):
        uid = blk.uniqueid()
        return [b for b in self if b.uniqueid() == uid]

    def find_block(self, name, benchmark=None):
        if benchmark is not None:
            if benchmark != self.name:
                return []
        return [b for b in self if b.name == name]


class Block:
    '''
    Abstracts a single block in the logs as a collection of log messages

    Handles EVENT logs nicely.

    Properties:
     - block.name: the name of this block
     - block.info: miscellaneous information about this block
     - block.raw_log: the raw log text for this block
     - block.events: the events in this block
    '''

    def __init__(self, info, raw_log, events):
        self.name = info['name']
        self.info = info
        self.raw_log = raw_log
        self.events = events

        if 'PassFinished' in self:
            self.info['pass'] = self.single('PassFinished')['num']

    def single(self, event_name):
        '''
        Gets an event with the specified name, requiring exactly one match

        raises AssertionError if there is not exactly one event with the specified name
        '''
        result = self.events[event_name]
        if len(result) != 1:
            raise AssertionError(f'Multiple events for {event_name}')

        return result[0]

    def __getitem__(self, event_name):
        '''
        Gets the events with the specified name
        '''
        return self.events[event_name]

    def get(self, event_name, default=None):
        '''
        Gets the events with the specified name, returning the default if the event doesn't exist
        '''
        return self.events.get(event_name, default)

    def __contains__(self, event_name):
        return event_name in self.events

    def __iter__(self):
        return iter(self.events)

    def __repr__(self):
        return f'<Block({self.info}, {len(self.events)} events)>'

    def uniqueid(self):
        return frozenset(self.info.items())

    def dump(self):
        print(self.raw_log)
