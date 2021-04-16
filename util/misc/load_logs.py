#!/usr/bin/env python3

# Intended to be used with python -i to load the logs to then be worked on in the REPL

import argparse
import analyze

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs')
    logs = args.logs
