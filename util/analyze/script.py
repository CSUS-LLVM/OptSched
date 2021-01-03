import atexit
import sys

from analyze.logs import Logs, Benchmark, Block
from analyze import basemain

RUN_CMDS = []


class Pos:
    def __init__(self, pos):
        self.pos = pos

    def __repr__(self):
        return f'${self.pos}'


class Run:
    def __init__(self, script_cls, *args, **kwargs):
        self.script_cls = script_cls
        self.args = args
        self.kwargs = kwargs
        self.__preds = []
        RUN_CMDS.append(self)

    def run_with(self, files, options):
        self.__script = self.script_cls(**options)
        args = [x if not isinstance(x, Pos) else files[x.pos]
                for x in self.args]
        return self.__script.run([self.__transform(arg) for arg in args])

    def print_report(self, *args, **kwargs):
        self.__script.print_report(*args, **kwargs)

    def keep_if(self, pred):
        self.__preds.append(pred)
        return self

    def __transform(self, logs: Logs):
        if not self.__preds:
            return logs

        benchmarks = [Benchmark(bench.info, [
            blk for blk in bench if all(p(blk) for p in self.__preds)
        ]) for bench in logs]
        return Logs(benchmarks)


run = lambda *args, **kwargs: Run(*args, **kwargs)


def __getattr__(name):
    if name.startswith('_'):
        try:
            return Pos(int(name[1:]))
        except:
            pass

    raise AttributeError(f"module '{__package__}' has no attribute '{name}'")


__all__ = ['run']
for i in range(10):
    __all__.append(f'_{i}')


def __find_positional(cmds):
    if not cmds:
        return []
    positional = cmds[0].script_cls.POSITIONAL

    for cmd in cmds[1:]:
        cmd_pos = cmd.script_cls.POSITIONAL
        if len(positional) < len(cmd_pos):
            positional.extend(cmd_pos[len(positional):])

    return positional


def __find_options(cmds):
    if not cmds:
        return {}
    cmds = list(reversed(cmds))

    options = {k: v for k, v in cmds[0].script_cls.OPTIONS.items()}

    for cmd in cmds[1:]:
        cmd_opts = cmd.script_cls.OPTIONS
        options.update(cmd_opts)

    return options


def __find_manual_options(cmds):
    if not cmds:
        return {}
    cmds = list(reversed(cmds))

    options = {k: v for k, v in cmds[0].kwargs.items()}

    for cmd in cmds[1:]:
        cmd_opts = cmd.kwargs
        options.update(cmd_opts)

    return options


@atexit.register
def __run():
    # import argparse
    # from analyze import import_cpu2006
    # parser = argparse.ArgumentParser()

    def script_action(outfile, files, option_values):
        for cmd in RUN_CMDS[:-1]:
            cmd.run_with(files, option_values)
            cmd.print_report(outfile)
            print(file=outfile)

        RUN_CMDS[-1].run_with(files, option_values)
        RUN_CMDS[-1].print_report(outfile)

    positional = __find_positional(RUN_CMDS)
    options = __find_options(RUN_CMDS)
    manual_options = __find_manual_options(RUN_CMDS)

    basemain(
        positional=positional,
        options=options,
        description='<script>',
        action=script_action,
        manual_options=manual_options,
    )

    # for name, help in positional:
    #     parser.add_argument(name, help=help)

    # parser.add_argument(
    #     '--benchsuite',
    #     default=None,
    #     choices=('spec',),
    #     help='Select the benchmark suite which the input satisfies. Valid options: spec',
    # )
    # parser.add_argument(
    #     '-o', '--output',
    #     default=None,
    #     help='Where to output the report',
    # )

    # args = parser.parse_args()
    # pos = [getattr(args, name) for name, help in positional]

    # FILE_PARSERS = {
    #     # None: load_filepath,
    #     'spec': import_cpu2006.parse,
    # }
    # parser = FILE_PARSERS[args.benchsuite]

    # pos_data = [parser(f) for f in pos]

    # if args.output is None:
    #     outfile = sys.stdout
    # else:
    #     outfile = open(args.output, 'w')

    # try:
    #     for cmd in RUN_CMDS:
    #         cmd.run_with(pos_data)
    #         cmd.print_report(outfile, options={})
    #         print(file=outfile)
    # except:
    #     if args.output is not None:
    #         outfile.close()
    #     raise
