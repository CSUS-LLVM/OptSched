import argparse
import analyze

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', help='The logs to analyze')
    args = analyze.parse_args(parser, 'logs')
    logs = args.logs

    print('Parsed logs into variable `logs`')
