[![CSUS](http://www.csus.edu/Brand/assets/Logos/Core/Primary/Stacked/Primary_Stacked_3_Color_wht_hndTN.png)](http://www.csus.edu/)

# OptSched - Optimizing Scheduler
Combinatorial instruction scheduling research project at CSU Sacramento.

This plugin for the [LLVM](https://llvm.org/) compiler is an optional machine scheduler. We implement an Ant Colony Optimization instruction scheduling algorithm on the GPU. The enumerating combinatorial scheduler will not work on this branch, this branch is for the ACO instruction scheduler running on the GPU.

## Requirements

- Ubuntu 18.04 or 20.04
- LLVM 15.0 or later
- AMD GPU: Radeon VII recommended, but any card supporting ROCm may work

## Building

**See [BUILD.md](BUILD.md) for build instructions.**

## Configuration files

OptSched reads from configuration files at runtime to initialize the scheduler. There are templates in the [example](https://github.com/OptSched/OptSched/tree/master/example/optsched-cfg) directory. The default search location for these files is ```~/.optsched-cfg```.

## Usage Examples

`hipcc <example.cpp>`

Note that the environment variable `HIP_CLANG_PATH` must point to `<llvm-project/build/bin>` after building LLVM with OptSched.
