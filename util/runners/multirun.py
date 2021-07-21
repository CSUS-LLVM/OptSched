#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from runners import runwith

# %% Setup


def main(outdir: Path, optsched_cfg: Path, labels: List[str], withs: List[Dict[str, str]], cmd: List[str], append_logs: bool = False, git_state: Optional[str] = None, validate_cmd: Optional[str] = None, analyze_cmds: List[str] = [], analyze_files: List[str] = []):
    assert len(labels) == len(withs)
    assert not analyze_files or len(analyze_files) == len(analyze_cmds)

    outdir = outdir.resolve()
    logfiles = []

    for label, with_ in zip(labels, withs):
        print(f'Running {label} with settings:', ' '.join(f'{k}={v}' for k, v in with_.items()))
        logfile = runwith.main(
            outdir=outdir,
            optsched_cfg=optsched_cfg,
            label=label,
            with_=with_,
            cmd=cmd,
            append_logs=append_logs,
            git_state=git_state,
        )
        logfiles.append(logfile)

    if validate_cmd:
        subprocess.run(validate_cmd + ' ' + shlex.join(logfiles), cwd=outdir,
                       check=True, stdout=subprocess.STDOUT, stderr=subprocess.STDERR)

    if not analyze_files:
        analyze_files = [None] * len(analyze_cmds)

    for analyze_cmd, outfile in zip(analyze_cmds, analyze_files):
        result = subprocess.run(analyze_cmd + ' ' + shlex.join(logfiles), cwd=outdir,
                                capture_output=True, encoding='utf-8')
        if result.returncode != 0:
            print(
                f'Analysis command {shlex.join(analyze_cmd)} failed with error code: {result.returncode}', file=sys.stderr)

        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        if outfile:
            with open(outdir / outfile, 'w') as f:
                f.write(result.stdout)


# %% Main
if __name__ == '__main__':
    OPTSCHEDCFG = os.getenv('OPTSCHEDCFG')
    RUN_CMD = os.getenv('RUN_CMD')
    RUN_CMD = shlex.split(RUN_CMD) if RUN_CMD else RUN_CMD
    VALIDATE_CMD = os.getenv('VALIDATE_CMD')
    ANALYZE_CMD = os.getenv('ANALYZE_CMD')

    parser = argparse.ArgumentParser(description='Run the commands with the sched.ini settings')
    parser.add_argument('-c', '--optsched-cfg',
                        required=OPTSCHEDCFG is None,
                        default=OPTSCHEDCFG,
                        help='The path to the optsched config to use. Defaults to the env variable OPTSCHEDENV if it exists, else is required. The sched.ini is expected to be there')
    parser.add_argument('-o', '--outdir', required=True, help='The path to place the output files at')
    parser.add_argument('-L', '--labels', required=True,
                        help='Comma separated labels to use for these runs. Must be equal to the number of --with flags')
    parser.add_argument('--with', nargs='*', action='append', metavar='KEY=VALUE',
                        help="The sched.ini settings to set for each run. Each run's settings should have a new --with flag.")
    parser.add_argument(
        'cmd', nargs='+', help='The command (with args) to run. Use - to default to the environment variable RUN_CMD.')
    parser.add_argument('--append', action='store_true',
                        help='Allow a <label>.log file to exist, appending to it if so')
    parser.add_argument('--git-state', help='The path to a git repository to snapshot its state in our <outdir>.')

    parser.add_argument('--validate', default=VALIDATE_CMD,
                        help='The command (single string) to run after all runs to validate that the runs were correct. Defaults to the env variable VALIDATE_CMD. The output log files will be passed to the command, one additional arg for each run.')
    parser.add_argument('--analyze', nargs='*', default=[ANALYZE_CMD],
                        help='The commands (each a single string) to run after all runs to analyze the runs and produce output. Defaults to the single command from the env variable ANALYZE_CMD. The output log files will be passed to each command, one additional arg for each run.')
    parser.add_argument('--analyze-files', nargs='*', default=[],
                        help='The filenames to place the stdout of each analyze command.')

    args = parser.parse_args()

    main(
        outdir=Path(args.outdir),
        optsched_cfg=Path(args.optsched_cfg),
        labels=args.labels,
        withs=[runwith.parse_withs(with_) for with_ in getattr(args, 'with', [])],
        cmd=args.cmd if args.cmd != '-' else RUN_CMD,
        append_logs=args.append,
        git_state=args.git_state,
        validate_cmd=args.validate,
        analyze_cmds=args.analyze,
        analyze_files=args.analyze_files,
    )
