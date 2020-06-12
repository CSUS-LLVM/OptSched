#!/usr/bin/env python3
import subprocess
import argparse
import os

#**************************************************************************************
#Description:	Run all plaidbench benchmarks and redirect output to a directory
#               that will contain the log file for each benchmark.
#Author:	    Austin Kerbow
#Modified By:	Justin Bassett
#Last Update:	May 4, 2020
#**************************************************************************************
# Requires write permission in the current directory.
#
# OUTPUT:
#   1.) Directories containing the log for each benchmark located in their own
#       directory.

NETWORKS = (
    "densenet121",
    "densenet169",
    "densenet201",
    "inception_resnet_v2",
    "inception_v3",
    "mobilenet",
    "nasnet_large",
    "nasnet_mobile",
    "resnet50",
    "vgg16",
    "vgg19",
    "xception",
    "imdb_lstm",
)

EXAMPLES = 4096
BATCH_SIZE = 16

parser = argparse.ArgumentParser(description='Run all plaidbench benchmarks, redirecting output to a directory which contains the log file for each benchmark')
parser.add_argument('-n', '--num-iterations', type=int, default=1, help='Number of iterations')
parser.add_argument('output', metavar='DIR', help='The output directory base path')

args = parser.parse_args()

NUM_ITERATIONS = args.num_iterations
DIRECTORY_BASE_PATH = args.output

for i in range(NUM_ITERATIONS):
    DIR_NAME = DIRECTORY_BASE_PATH + '-' + str(i)

    for network in NETWORKS:
        RESULT_DIR = os.path.join(DIR_NAME, network)
        os.makedirs(RESULT_DIR, exist_ok=True)

        with open(os.path.join(RESULT_DIR, network + '.log'), 'w') as outfile:
            subprocess.run(['plaidbench', '--examples', str(EXAMPLES),
                '--batch-size', str(BATCH_SIZE),
                '--results', DIR_NAME,
                'keras', '--no-fp16', '--no-train', network,
                ], check=True, stderr=subprocess.STDOUT, stdout=outfile)
