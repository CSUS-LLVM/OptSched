#!/bin/bash
#Build benchmarks and copy them to the A7 machine.

BENCH="401.bzip2 429.mcf 433.milc 445.gobmk 456.hmmer 458.sjeng 462.libquantum 464.h264ref 470.lbm 482.sphinx3  444.namd 447.dealII 453.povray 471.omnetpp 473.astar"

# source the shrc
. ./shrc

# Try to scrub benchmarks. Catch unchecked error in runspec where the benchmarks are not actually cleaned if
# they were built by another user or root.
echo 'Cleaning benchmarks'
rslt=$(runspec --loose -size=test -iterations=1 -config=Intel_llvm_3.9.cfg --tune=base -r 1 -I -a scrub $BENCH 2>&1 | \
awk '/Couldn'\''t unlink/ { print "1"; exit 1 }' -)
if [ ! -z $rslt ];
then
  echo "Error scrubbing benchmarks. Try with sudo."
  echo "\"sudo sh -c '. ./shrc; runspec --loose -size=test -iterations=1 -config=Intel_llvm_3.9.cfg --tune=base -r 1 -I -a scrub all'\""
  exit 1
fi

echo 'Building benchmarks'
runspec --loose -size=test -iterations=1 -config=Intel_llvm_3.9.cfg --tune=base -r 1 -I -a build $BENCH 2>&1 > /dev/null

#echo 'Creating fake run directories'
#runspec --fake --loose --size test --tune base --config Intel_llvm_3.9.cfg $BENCH

cd ./benchspec/CPU2006/

echo 'Creating archive'
tar cJf ziped_benches.tar.xz */exe

echo 'Copying to A7 machine'
scp -q ziped_benches.tar.xz ghassan@99.113.71.118:~

echo 'Cleaning benchmarks again'
runspec --loose -size=test -iterations=1 -config=Intel_llvm_3.9.cfg --tune=base -r 1 -I -a scrub $BENCH 2>&1 > /dev/null

echo 'Done!'
