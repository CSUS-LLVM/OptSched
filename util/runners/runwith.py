#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

# %% Setup


class InvalidSchedIniSettingError(Exception):
    def __init__(self, message: str, keys: List[str]):
        self.keys = keys
        super().__init__(f'{message}: keys={keys}')


class LogFileExistsError(FileExistsError):
    pass


class GitStateChangedError(Exception):
    def __init__(self, message: str, old: Path, new: Path):
        self.old = old
        self.new = new
        super().__init__(f'{message}: old={old}, new={new}')


def setup_outdir(outdir: Path, optsched_cfg: Path, keys: Iterable[str], git_state: str):
    outdir.mkdir(parents=True, exist_ok=True)
    if not (outdir / 'optsched-cfg').exists():
        shutil.copytree(optsched_cfg, outdir / 'optsched-cfg')

    if git_state:
        p: Path = outdir / 'git.status'
        p2: Path = outdir / 'git.status.2'
        with open(p2, 'w') as f:
            f.write(git_state)

        if p.exists():
            with open(p, 'r') as f:
                if f.read() != git_state:
                    raise GitStateChangedError(
                        'Git state changed between runs! If this was intended, delete the old git.status file. See state files', p, p2)
            os.remove(p)

        p2.rename(p)

    for key in keys:
        p: Path = outdir / key
        p.touch()


def edit_sched_ini(sched_ini: str, with_: Dict[str, str]) -> str:
    missing_keys = []

    for key, value in with_.items():
        kv_re = re.compile(rf'(?<=^{key} ).*$', flags=re.MULTILINE)
        if not kv_re.search(sched_ini):
            missing_keys.append(key)
        sched_ini = kv_re.sub(value, sched_ini)

    if missing_keys:
        raise InvalidSchedIniSettingError('Unable to find these keys in the sched.ini file', missing_keys)

    return sched_ini


def save_sched_ini(outdir: Path, sched_ini: Path, label: str):
    shutil.copy(sched_ini, outdir / f'{label}.sched.ini')


def edit_file(path: Path, edit: Callable[[str], str]):
    # assert path.is_file()
    with open(path, 'r+') as f:
        contents = edit(f.read())
        f.seek(0)
        f.truncate()
        f.write(contents)


def run_cmd(cmd: List[str], outdir: Path, label: str, logmode='w'):
    logfile = outdir / f'{label}.log'
    if logfile.exists() and logmode == 'w':
        raise LogFileExistsError(
            f'File already exists. Either use a fresh output directory, or specify that we should append to the file: {logfile}')

    with open(outdir / f'{label}.log', logmode) as f:
        subprocess.run(shlex.join(cmd), stdout=f, stderr=subprocess.STDOUT, cwd=outdir, check=True, shell=True)

    return logfile


def get_git_state(git_state: Optional[str]) -> str:
    if not git_state:
        return ''

    git_repo = str(Path(git_state).resolve())
    commit = subprocess.run(['git', '-C', git_repo, 'log', '-n1'], encoding='utf-8', capture_output=True, check=True)
    status = subprocess.run(['git', '-C', git_repo, 'status'], encoding='utf-8', capture_output=True, check=True)
    diff = subprocess.run(['git', '-C', git_repo, 'diff'], encoding='utf-8', capture_output=True, check=True)

    return f'{git_repo}\n{commit.stdout}\n\n{status.stdout}\n\n{diff.stdout}'


def main(outdir: Path, optsched_cfg: Path, label: str, with_: Dict[str, str], cmd: List[str], append_logs: bool = False, git_state: Optional[str] = None):
    outdir = outdir.resolve()
    optsched_cfg = optsched_cfg.resolve()

    git_state = get_git_state(git_state)
    setup_outdir(outdir, optsched_cfg, with_.keys(), git_state)

    sched_ini = optsched_cfg / 'sched.ini'
    if with_:
        edit_file(sched_ini, lambda f: edit_sched_ini(f, with_))
    save_sched_ini(outdir, sched_ini, label)

    return run_cmd(cmd, outdir, label, logmode='a' if append_logs else 'w')


def parse_withs(withs: List[str]) -> Dict[str, str]:
    return dict(with_.split('=', maxsplit=1) for with_ in withs)


# %% Main
if __name__ == '__main__':
    OPTSCHEDCFG = os.getenv('OPTSCHEDCFG')
    RUN_CMD = os.getenv('RUN_CMD')
    RUN_CMD = shlex.split(RUN_CMD) if RUN_CMD else RUN_CMD

    parser = argparse.ArgumentParser(description='Run the commands with the sched.ini settings')
    parser.add_argument('-c', '--optsched-cfg',
                        required=OPTSCHEDCFG is None,
                        default=OPTSCHEDCFG,
                        help='The path to the optsched config to use. Defaults to the env variable OPTSCHEDCFG if it exists, else is required. The sched.ini is expected to be there')
    parser.add_argument('-o', '--outdir', required=True, help='The path to place the output files at')
    parser.add_argument('-L', '--label', required=True,
                        help='A label for this run, used in the output directory and for namespacing.')
    parser.add_argument('--with', nargs='*', metavar='KEY=VALUE', help='The sched.ini settings to set.')
    parser.add_argument(
        'cmd', nargs='+', help='The command (with args) to run. Use - to default to the environment variable RUN_CMD.')
    parser.add_argument('--append', action='store_true',
                        help='Allow a <label>.log file to exist, appending to it if so')
    parser.add_argument('--git-state', help='The path to a git repository to snapshot its state in our <outdir>.')

    args = parser.parse_args()

    main(
        outdir=Path(args.outdir),
        optsched_cfg=Path(args.optsched_cfg),
        label=args.label,
        with_=parse_withs(getattr(args, 'with') if getattr(args, 'with') is not None else []),
        cmd=args.cmd if args.cmd != '-' else RUN_CMD,
        append_logs=args.append,
        git_state=args.git_state,
    )
