#!/usr/bin/env python3

import argparse
import sys
import analyze

__INTERACTIVE = bool(getattr(sys, 'ps1', sys.flags.interactive))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs')
    logs = args.logs

    if __INTERACTIVE:
        print('Parsed logs into variable `logs`')
    else:
        import code
        code.interact(banner='Parsed logs into variable `logs`', exitmsg='', local={'logs': logs})
