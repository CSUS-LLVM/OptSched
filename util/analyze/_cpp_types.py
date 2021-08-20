import sys
from typing import Iterator, Union

from . import _types


def _make_classes(cpp):
    class Block(_types.Block):
        def __init__(self, block: cpp.Block):
            self.__cpp = block
            self.name = self.__cpp.name

        @property
        def raw_log(self):
            return self.__cpp.raw_log

        # uses inherited single(...)

        def __getitem__(self, event_name):
            return self.__cpp[event_name]

        def get(self, event_name, default=None):
            if event_name in self.__cpp:
                return self.__cpp[event_name]
            return default

        def __contains__(self, event_name) -> bool:
            return event_name in self.__cpp

        def __iter__(self) -> Iterator[str]:
            return iter(self.__cpp._event_names())

        def __repr__(self):
            return repr(self.__cpp)

        def uniqueid(self):
            return self.__cpp.uniqueid

    class _BenchmarkBlocks:
        def __init__(self, blocks: cpp._Blocks):
            self.__cpp = blocks

        def __getitem__(self, index: int) -> _types.Block:
            return Block(self.__cpp[index])

        def __len__(self) -> int:
            return len(self.__cpp)

        def __repr__(self):
            return repr(self.__cpp)

    class Benchmark(_types.Benchmark):
        def __init__(self, benchmark: cpp.Benchmark):
            self.__cpp = benchmark
            self.name = self.__cpp.name

        @property
        def blocks(self):
            return _BenchmarkBlocks(self.__cpp.blocks)

        @property
        def raw_log(self):
            return self.__cpp.raw_log

        # Inherit __iter__

        # Inherit .benchmarks

        def __repr__(self):
            return repr(self.__cpp)

        def keep_blocks_if(self, p):
            return _types.Benchmark(
                {'name': self.name},
                list(filter(p, self)),
            )

    class Logs(_types.Logs):
        def __init__(self, logs: cpp.Logs):
            self.__cpp = logs
            self.benchmarks = list(Benchmark(bench) for bench in logs.benchmarks)

        @property
        def raw_log(self):
            return self.__cpp.raw_log

        def benchmark(self, name: str) -> _types.Benchmark:
            for bench in self.benchmarks:
                if bench.name == name:
                    return bench

            raise KeyError(f'No benchmark `{name}` in this Logs')

        def __iter__(self):
            for bench in self.benchmarks:
                yield from bench

        def __repr__(self):
            return repr(self.__cpp)

        def keep_blocks_if(self, p):
            return _types.Logs([b.keep_blocks_if(p) for b in self.benchmarks])

    return {
        'Logs': Logs,
        'Benchmark': Benchmark,
        'Block': Block,
    }


class _M:
    def __init__(self):
        self.__cpp = None

    @property
    def VERSION(self):
        return self.__cpp.VERSION

    @property
    def __doc__(self):
        return self.__cpp.__doc__

    def load_module(self, cpp):
        self.__cpp = cpp
        classes = _make_classes(self.__cpp)
        self.Logs = classes['Logs']
        self.Benchmark = classes['Benchmark']
        self.Block = classes['Block']

    def parse_blocks(self, file, benchspec: Union[str, int]) -> _types.Logs:
        return self.Logs(self.__cpp.parse_blocks(file, benchspec))


sys.modules[__name__] = _M()
