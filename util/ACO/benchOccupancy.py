#!/usr/bin/python3
import argparse
import json
def main(args):
    kernels = {}
    benches = {}

    with open(args.kernel_occupancy_path, 'r') as file:
        for line in file:
            kernel, occ = line.split(",")
            kernels[kernel] = occ

    with open(args.kernel_trace_path, "r") as file: benches = json.loads(file.read())

    for bench in benches.keys():
        _list = benches[bench]
        d = {}
        for kernel in _list:
            if kernel in kernels: d[kernel] = int(kernels[kernel])
        benches[bench] = d

    with open(args.out_path, "w") as file: file.write(json.dumps(benches, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
     )

    parser.add_argument('--occpath', '-c',
                        dest='kernel_occupancy_path',
                        type=str,
                        default=None,
                        required=True
                        )

    parser.add_argument('--tracepath', '-t',
                        dest='kernel_trace_path',
                        type=str,
                        default=None,
                        required=True
                        )

    parser.add_argument('--output', '-o',
        dest='out_path',
        type=str,
        default=None,
        required=True,
        help='Path to write output csv file.'
        )

    args = parser.parse_args()
    main(args)


