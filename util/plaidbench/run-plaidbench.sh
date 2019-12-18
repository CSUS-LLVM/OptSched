#!/bin/bash
#**************************************************************************************
#Description:	Run all plaidbench benchmarks and redirect output to a directory 
#               that will contain the log file for each benchmark.
#Author:	    Austin Kerbow
#Modified By:	Vang Thao
#Last Update:	November 27, 2019
#**************************************************************************************
# Requires write permission in the current directory.
# Note: DirectoryName can be changed after each run to output to a different directory.

declare -a Networks=("densenet121" "densenet169" "densenet201" "inception_resnet_v2" "inception_v3" "mobilenet" "nasnet_large" "nasnet_mobile" "resnet50" "vgg16" "vgg19" "xception" "imdb_lstm")
Examples=4096
BatchSize=16
Command="plaidbench --examples $Examples --batch-size $BatchSize --results "
# Note: The run number at the end such as "01" should be kept. This number is used
# in other scripts. 
DirectoryName="plaidbench-optsched-01"
Keras="keras --no-fp16 --no-train"

for i in 1
do
  for Network in "${Networks[@]}"
  do
    mkdir -p $DirectoryName/$Network
    $Command$DirectoryName/$Network $Keras $Network &> $DirectoryName/$Network/$Network.log
  done
done
