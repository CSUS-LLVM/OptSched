[![CSUS](http://www.csus.edu/Brand/assets/Logos/Core/Primary/Stacked/Primary_Stacked_3_Color_wht_hndTN.png)](http://www.csus.edu/)

# OptSched - Optimizing Scheduler
Combinatorial instruction scheduling research project at CSU Sacramento.

This plugin for the [LLVM](https://llvm.org/) compiler is an optional machine scheduler. We implement a branch and bound instruction scheduling algorithm.

## Requirements

- Ubuntu 16.04 is recommended
- CMake 3.4.3 or later
- LLVM 6.0 or later

## Building

1. Clone the repository to the [“llvm/projects”](https://github.com/llvm/llvm-project/tree/master/llvm/projects) directory in the [LLVM](https://llvm.org/) source tree.
2. [Build](https://llvm.org/docs/CMake.html) LLVM with CMake. The OptSched plugin can be found in “llvm/lib” after building.

See [Getting started](#Getting-started).

## Configuration files

OptSched reads from configuration files at runtime to initialize the scheduler. There are templates in the [example](https://github.com/OptSched/OptSched/tree/master/example/optsched-cfg) directory. The default search location for these files is ```~/.optsched-cfg```. You can optionally specify the path to this directory or any of the configuration files individually with [command-line options](#Command-Line-Options).

## Usage

`clang++ -O3 -fplugin=<path/to/OptSched.so> -mllvm -misched=optsched -mllvm -optsched-cfg=<path/to/optsched-cfg>  <example.cpp>`
  
`llc -load <path/to/OptSched.so> -misched=optsched -optsched-cfg=<path/to/optsched-cfg> <example.ll>`
  
## Command-Line Options

| CL Opt | Description |
| ------ | ----------- |
| -enable-misched | Enable the machine scheduling pass in LLVM (Targets can override this option). |
| -misched=optsched | Select the optimizing scheduler. |
| -optsched-cfg=\<string\> | Path to the directory containing configuration files for opt-sched. |
| -optsched-cfg-hotfuncs=\<string\> | Path to the list of hot functions to schedule using opt-sched. |
| -optsched-cfg-machine-model=\<string\> | Path to the machine model specification file for opt-sched. |
| -optsched-cfg-sched=\<string\> | Path to the scheduler options configuration file for opt-sched. |
| -mllvm \<opt\> | Add llvm options. |
| -fplugin=\<string\> | Load a shared library. |

## Getting started

Steps to build LLVM and clang with OptSched.

1. Clone the LLVM project:

`git clone https://github.com/llvm/llvm-project.git`

2. Clone OptSched in the projects directory:

`cd llvm-project/llvm/projects && git clone https://github.com/CSUS-LLVM/OptSched.git`

3. Create a build directory:

`mkdir build && cd build`

4. Build LLVM/clang/OptSched. See [https://llvm.org/docs/CMake.html]( https://llvm.org/docs/CMake.html) for more build options:

`cmake -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../.. && make -j8`

5. Test the build:

`echo 'int main(){};' | ./bin/clang -xc - -O3 -fplugin=lib/OptSched.so -mllvm -misched=optsched -mllvm -enable-misched -mllvm -optsched-cfg=../OptSched/example/optsched-cfg`

You should not see any errors.
