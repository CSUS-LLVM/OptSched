import csv
import sys


class Analyzer:
    POSITIONAL = [
        ('logs', 'The logs to analyze')
    ]

    OPTIONS = {}

    def __init__(self):
        subclass = type(self).__name__
        assert (type(self).run is not Analyzer.run
                or type(self).run_bench is not Analyzer.run_bench), \
            f'Subclass `{subclass}` needs to implement run() or run_bench()'

        self.stats = [{'benchmark': 'Total'}]
        self.__current_benchmark = None

    @property
    def current_benchmark(self):
        return self.__current_benchmark

    @current_benchmark.setter
    def current_benchmark(self, value):
        self.__current_benchmark = value
        self.stats.append({'benchmark': value})

    def run(self, args):
        # First, run for everything
        self.run_bench(args)

        # Then, run for each individual benchmark
        for benchmarks in zip(*args):
            assert all(it.name == benchmarks[0].name for it in benchmarks), \
                'Mismatching benchmark sequences'
            self.current_benchmark = benchmarks[0].name
            self.run_bench(benchmarks)

    def run_bench(self, args):
        raise NotImplementedError(
            f'Subclass `{type(self).__name__}` needs to implement run_bench()')

    def stat(self, *args, **kwargs):
        assert not args or not kwargs, \
            'Use named-argument syntax or function call syntax, but not both.'

        result = self.stats[-1]
        if args:
            name, value = args
            result[name] = value
        else:
            result.update(kwargs)

    def print_report(self, out, options):
        out = csv.DictWriter(out, fieldnames=self.stats[0].keys())
        out.writeheader()
        out.writerows(self.stats)
