#!/bin/sh
# Run a dry run of the CPU2006 benchmarks and extract the commands necessary
# for running the binaries on a different machine without runspec.
# eg: When cross-compiling.

# ref (reference) or test (test) input size for the benchmarks.
SIZE=test

runspec --fake --loose --size $SIZE --tune base --config Intel_llvm_3.9.cfg $1 | \
  awk '/Benchmark invocation/ {record=1} /Benchmark verification/ {record=0} record' - | \
  awk '/echo/ {split($0, res, "\""); print res[2] }'
