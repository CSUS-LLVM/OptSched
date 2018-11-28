[![CSUS](http://www.csus.edu/Brand/assets/Logos/Core/Primary/Stacked/Primary_Stacked_3_Color_wht_hndTN.png)](http://www.csus.edu/)

# OptSched - Optimizing Scheduler
Combinatorial instruction scheduling research project at CSU Sacramento.

This plugin for the [LLVM](https://llvm.org/) compiler is an optional machine scheduler. We implement a branch and bound instruction scheduling algorithm.

## Requirements

- Ubuntu 16.04 is recommended
- CMake 3.4.3 or later
- LLVM 6.0 or later

## Building

1. Clone the repository to the [“llvm/projects”](https://github.com/llvm-mirror/llvm/tree/master/projects) directory in the [LLVM](https://llvm.org/) source tree.
2. [Build](https://llvm.org/docs/CMake.html) LLVM with CMake. The OptSched plugin can be found in “llvm/lib” after building.

## Configuration files

OptSched currently needs to read from configuration files to initialize the scheduler. There are example files in the [“optsched-cfg”](https://github.com/OptSched/OptSched/tree/master/optsched-cfg) directory. The default search location for these files is ```~/.optsched-cfg```. You can optionally specify the path to this directory or any of the configuration files with [command-line options](#Command-Line-Options).

## Usage

`clang++ -O3 -Xclang -load -Xclang <path/to/OptSched.so> -mllvm -misched=optsched <example.cpp>`
  
`llc -load <path/to/OptSched.so> -misched=optsched -optsched-cfg=<path/to/optsched-cfg> <example.ll>`
  
## Command-Line Options

| CL Opt | Description |
| ------ | ----------- |
| -enable-misched | Enable the machine scheduling pass in LLVM (Targets can override this option). |
| -misched=optsched | Use the OptSched scheduler. |
| -optsched-cfg=\<string\> | Path to the directory containing configuration files for opt-sched. |
| -optsched-cfg-hotfuncs=\<string\> | Path to the list of hot functions to schedule using opt-sched. |
| -optsched-cfg-machine-model=\<string\> | Path to the machine model specification file for opt-sched. |
| -optsched-cfg-sched=\<string\> | Path to the scheduler options configuration file for opt-sched. |
| -mllvm \<opt\>| Add llvm options. |
| -Xclang \<opt\>| Tell the clang driver to pass options directly to the frontend. |
  
  
