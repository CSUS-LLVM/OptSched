import csv
import json
import fnmatch

import inspect
from inspect import Parameter
import functools
import argparse
from . import logs
from . import import_cpu2006
from . import import_plaidml
from . import import_shoc


def _get_annotations(sig: inspect.Signature):
    return set(p.annotation for p in sig.parameters.values())


def _get_logs_args(sig: inspect.Signature):
    if logs.Logs in _get_annotations(sig):
        return set(name for name, param in sig.parameters.items() if param.annotation is logs.Logs)

    result = []

    for param in sig.parameters.values():
        if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD) \
                and param.default == Parameter.empty:
            result.append(param.name)

    return set(result)


def _compute_options(parser: argparse.ArgumentParser, sig: inspect.Signature, *, paraminfo: dict):
    for param in sig.parameters.values():
        args = []
        kwargs = {}

        if param.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            if param.name in paraminfo and 'abbrev' in paraminfo[param.name]:
                abbrev = paraminfo[param.name]['abbrev']
                args.append(f'-{abbrev}')
            name = param.name.replace('_', '-')
            args.append(f'--{name}')
            is_positional = False
        else:
            args.append(param.name)
            is_positional = True

        the_type = param.annotation
        if the_type != Parameter.empty and type(the_type) is type:
            kwargs['type'] = the_type

        if param.default != Parameter.empty:
            kwargs['default'] = param.default
        elif not is_positional:
            kwargs['required'] = True

        if param.name in paraminfo and 'help' in paraminfo[param.name]:
            kwargs['help'] = paraminfo[param.name]['help']
        if param.name in paraminfo and 'choices' in paraminfo[param.name]:
            kwargs['choices'] = paraminfo[param.name]['choices']

        parser.add_argument(*args, **kwargs)


class AnalyzerDecorator:
    def __init__(self, fn, paraminfo):
        self._sig = inspect.signature(fn)
        self._paraminfo = paraminfo
        self._logs_args = _get_logs_args(self._sig)

        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def main(self, **kwargs):
        parser = argparse.ArgumentParser(**kwargs)

        parser.add_argument(
            '--benchsuite',
            required=True,
            choices=('spec', 'plaidml', 'shoc'),
            help='Select the benchmark suite which the input satisfies. Valid options: spec, plaidml, shoc',
        )
        parser.add_argument(
            '--keep-blocks-if',
            default='{}',
            help='Keep blocks matching (JSON format)',
        )
        _compute_options(parser, self._sig, paraminfo=self._paraminfo)

        args = vars(parser.parse_args())
        blk_filter = json.loads(args['keep_blocks_if'])

        def log_matches(log, pattern):
            if not isinstance(pattern, dict):
                if isinstance(pattern, str):
                    return fnmatch.fnmatchcase(str(log), pattern)
                return log == pattern

            return all(
                k in log and log_matches(log[k], v)
                for k, v in pattern.items()
            )

        def blk_filter_f(blk):
            return all(
                event in blk and all(log_matches(log, matcher) for log in blk[event])
                for event, matcher in blk_filter.items()
            )

        FILE_PARSERS = {
            'spec': import_cpu2006.parse,
            'plaidml': import_plaidml.parse,
            'shoc': import_shoc.parse,
        }
        parser = FILE_PARSERS[args['benchsuite']]

        for arg, value in args.items():
            if arg in self._logs_args:
                args[arg] = parser(value).keep_blocks_if(blk_filter_f)

        positional = []
        kwargs = {}
        for name, param in self._sig.parameters.items():
            if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(args[name])
            else:
                assert param.kind == Parameter.KEYWORD_ONLY
                kwargs[name] = args[name]

        return self(*positional, **kwargs)


def analyzer(analyzer_fn=None, **paraminfo):
    def _decorate(analyzer_fn):
        return functools.wraps(analyzer_fn)(AnalyzerDecorator(analyzer_fn, paraminfo))

    if analyzer_fn is not None:
        return _decorate(analyzer_fn)
    return _decorate


class Analyzer:
    POSITIONAL = [
        ('logs', 'The logs to analyze')
    ]

    # Documentation for any options
    OPTIONS = {}

    def __init__(self, **options):
        subclass = type(self).__name__
        assert (type(self).run is not Analyzer.run
                or type(self).run_bench is not Analyzer.run_bench), \
            f'Subclass `{subclass}` needs to implement run() or run_bench()'

        self.stats = [{'benchmark': 'Total'}]
        self.__current_benchmark = None
        self.options = options

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

    def print_report(self, out):
        out = csv.DictWriter(out, fieldnames=self.stats[0].keys())
        out.writeheader()
        out.writerows(self.stats)
