[![CSUS](http://www.csus.edu/Brand/assets/Logos/Core/Primary/Stacked/Primary_Stacked_3_Color_wht_hndTN.png)](http://www.csus.edu/)

# OptSched - Optimizing Scheduler
Combinatorial instruction scheduling research project at CSU Sacramento.

This plugin for the [LLVM](https://llvm.org/) compiler is an optional machine scheduler. We implement an Ant Colony Optimization instruction scheduling algorithm on the GPU. The enumerating combinatorial scheduler will not work on this branch, this branch is for the ACO instruction scheduler running on the GPU.

## Requirements

- Ubuntu 18.04 or 20.04
- CMake 3.20.4 or later
- LLVM 7.0 or later
- Nvidia GPU: At least Turing recommended (CC 7.5 or later)

## Building

**See [BUILD.md](BUILD.md) for build instructions.**

The OptSched plugin can be found in “llvm/lib” after building.

## Configuration files

OptSched reads from configuration files at runtime to initialize the scheduler. There are templates in the [example](https://github.com/OptSched/OptSched/tree/master/example/optsched-cfg) directory. The default search location for these files is ```~/.optsched-cfg```. You can optionally specify the path to this directory or any of the configuration files individually with [command-line options](#Command-Line-Options).

## Usage Examples

`clang++ -O3 -fplugin=<path/to/OptSched.so> -mllvm -misched=optsched -mllvm -optsched-cfg=<path/to/optsched-cfg>  <example.cpp>`

`llc -load <path/to/OptSched.so> -misched=optsched -optsched-cfg=<path/to/optsched-cfg> <example.ll>`

## Command-Line Options

When using Clang, pass options to LLVM with `-mllvm`.

| CL Opt | Description |
| ------ | ----------- |
| -enable-misched | Enable the machine scheduling pass in LLVM (Targets can override this option). |
| -misched=optsched | Select the optimizing scheduler. |
| -debug-only=optsched | Print debug information from the scheduler. |
| -optsched-cfg=\<string\> | Path to the directory containing configuration files for opt-sched. |
| -optsched-cfg-hotfuncs=\<string\> | Path to the list of hot functions to schedule using opt-sched. |
| -optsched-cfg-machine-model=\<string\> | Path to the machine model specification file for opt-sched. |
| -optsched-cfg-sched=\<string\> | Path to the scheduler options configuration file for opt-sched. |
