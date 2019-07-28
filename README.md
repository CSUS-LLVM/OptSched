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

See [Quick Start Guide](#New-Student-Quick-Start-Guide).

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

## New Student Quick Start Guide

### Ubuntu Basic Setup

#### Prerequisites

**Attention:** This section is only for building on your own machine. For building on Grace 2,
please skip to the [build instructions](#build-optsched-with-llvm-6-and-clang-ubuntu-and-macos).

Starting with a fresh install of Ubuntu 16.04 is recommended.

1. Install dependencies:

`sudo apt update && sudo apt upgrade`

`sudo apt install cmake git`

#### Install Ninja (Optional)

It is recommended to build LLVM using [Ninja](https://ninja-build.org/) to avoid running out of memory during linking. Using Ninja should also result in faster builds.

1. Download and install Ninja 1.9:

`wget -q https://github.com/ninja-build/ninja/releases/download/v1.9.0/ninja-linux.zip && unzip -q ninja-linux.zip && sudo cp ninja /usr/bin && rm ninja ninja-linux.zip`

### Build OptSched with LLVM 6 and Clang (Ubuntu and MacOS)

1. Clone LLVM:

`git clone https://github.com/llvm/llvm-project.git`

2. Checkout LLVM release 6:

`cd llvm-project && git checkout release/6.x` 

3. Clone OptSched in the projects directory:

`cd llvm/projects && git clone https://github.com/CSUS-LLVM/OptSched.git`

4. Create a build directory:

`mkdir build && cd build`

#### MacOS: Follow the instructions in [README-MacOS.md]

#### Ubuntu only:

5. Build LLVM/clang/OptSched. See [https://llvm.org/docs/CMake.html]( https://llvm.org/docs/CMake.html) for more build options:

In debug builds, linking uses a lot of memory. Set LLVM_PARALLEL_LINK_JOBS=2 if you have >= 32G memory, otherwise use LLVM_PARALLEL_LINK_JOBS=1.

`cmake -GNinja -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../..` 

`ninja -j32`

or if you want to use ‘make’:

`cmake -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../..`

`make`

A debug build of LLVM on a single thread will take a long time.

### Test the build (Ubuntu and MacOS)

`echo 'int main(){};' | ./bin/clang -xc - -O3 -fplugin=lib/OptSched.so -mllvm -misched=optsched -mllvm -enable-misched -mllvm -optsched-cfg=../OptSched/example/optsched-cfg -mllvm -debug-only=optsched`

You can rebuild OptSched without building/linking LLVM libraries and binaries with `ninja OptSched` or `make OptSched` depending on which generator you used.

[README-MacOS.md]: README-MacOS.md

