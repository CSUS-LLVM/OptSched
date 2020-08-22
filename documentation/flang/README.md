[![CSUS](http://www.csus.edu/Brand/assets/Logos/Core/Primary/Stacked/Primary_Stacked_3_Color_wht_hndTN.png)](http://www.csus.edu/)

# OptSched - Optimizing Scheduler
This directory contains specific instructions on how to build Flang.

## Requirements

- Ubuntu 16.04 is recommended
- CMake 3.4.3 or later
- LLVM 6.0 or later

## Usage with OptSched

`<path-to-flang-install-directory>/bin/flang -O3 -fplugin=<path/to/OptSched.so> -mllvm -misched=optsched -mllvm -optsched-cfg=<path/to/optsched-cfg> <example.cpp>`

## Building Flang

#### Flang Build Directory

1. Open a bash terminal

2. Create a directory where you would like flang to be installed to. For example "flang-install" then navigate inside it:

`mkdir flang-install && cd flang-install`

3. Export an environment variable that will be passed to cmake to indicate flang's installation directory:

`export FLANG_INSTALL=$(pwd)`

4. Navigate outside of the folder:

`cd ..`

#### Building Flang LLVM

1. Clone Flang LLVM:

`git clone https://github.com/flang-compiler/llvm.git`

2. Navigate to inside the LLVM folder and swap to the release_60 branch:

`cd llvm && git checkout release_60`

3. Download the patch to print spilling info under the OptSched/patches/llvm6.0/ folder named:

`flang-llvm6-print-spilling-info.patch`

4. Move the patch file to inside the Flang LLVM directory

5. Apply the patch

`git am flang-llvm6-print-spilling-info.patch`

3. Create a build directory and navigate inside it:

`mkdir build && cd build`

6. Build Flang LLVM

`cmake -DCMAKE_BUILD_TYPE=Debug '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_INCLUDE_TESTS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL ..`

`make && make install`

7. Navigate outside of the Flang LLVM directory:

`cd ../..`

#### Building the Flang driver

1. Clone the Flang driver

`git clone https://github.com/flang-compiler/flang-driver.git`

2. Navigate to inside the flang driver folder and swap to the release_60 branch:

`cd flang-driver && git checkout release_60`

3. Create a build directory and navigate inside it:

`mkdir build && cd build`

4. Build the Flang driver:

`cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DLLVM_CONFIG=$FLANG_INSTALL/bin/llvm-config -DCLANG_ENABLE_STATIC_ANALYZER=ON ..`

`make && make install`

5. Navigate outside of the flang driver directory:

`cd ../..`

#### Building the OpenMP runtime library

1. Clone the OpenMP runtime library:

`git clone https://github.com/llvm-mirror/openmp.git`

2. Navigate to the OpenMP runtime library directory:

`cd openmp/runtime/`

3. Create a build directory and navigate inside it:

`mkdir build && cd build`

4. Build the OpenMP runtime library:

`cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DCMAKE_CXX_COMPILER=$FLANG_INSTALL/bin/clang++ -DCMAKE_C_COMPILER=$FLANG_INSTALL/bin/clang ../..`

`make && make install`

5. Navigate outside of the OpenMP runtime directory:

`cd ../../..`

#### Building libpgmath

1. Clone Flang

`git clone https://github.com/flang-compiler/flang.git`

2. Navigate to inside the flang directory:

`cd flang`

If you are having issues with AVX-512 when building libpgmath, you may need to swap to an older commit

For Ubuntu 16.04: [45d7aeb5886c5965a8e793ef3fa632e7e73de56c](https://github.com/flang-compiler/flang/issues/434#issuecomment-403449362)

`git checkout 45d7aeb5886c5965a8e793ef3fa632e7e73de56c`

For Ubuntu 18.04: [37e6062d969bf337b964fe8119767046fcbdcdfa](https://github.com/flang-compiler/flang/issues/685)

`git checkout 37e6062d969bf337b964fe8119767046fcbdcdfa`

3. Navigate to inside the libpgmath dircetory:

`cd runtime/libpgmath`

4. Create a build directory and navigate inside it:

`mkdir build && cd build`

5. Build libpgmath:

`cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DCMAKE_CXX_COMPILER=$FLANG_INSTALL/bin/clang++ -DCMAKE_C_COMPILER=$FLANG_INSTALL/bin/clang -DCMAKE_Fortran_COMPILER=$FLANG_INSTALL/bin/flang ..`

`make && make install`

You may need to install gawk if you are encountering a segmentation fault:

`sudo apt-get install gawk`

6. Navigate back to the root directory of flang

`cd ../../..`

#### Building flang

1. While still in the flang directory, create a build directory for flang and navigate inside it:

`mkdir build && cd build`

2. Build flang:

`cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DCMAKE_CXX_COMPILER=$FLANG_INSTALL/bin/clang++ -DCMAKE_C_COMPILER=$FLANG_INSTALL/bin/clang -DCMAKE_Fortran_COMPILER=$FLANG_INSTALL/bin/flang -DLLVM_CONFIG=$FLANG_INSTALL/bin/llvm-config ..`

`make && make install`

3. Navigate outside of the flang directory:

`cd ../..`


#### Testing the build with a hello world fortran file

1. Navigate to the directory where flang was installed. In this example, it was flang-install

`cd flang-install`

2. Download the hello.f fortran file and put it in your flang-install directory

3. Compile the file:

`./bin/flang hello.f`

4. Run the generated file:

`./a.out`

If you are getting the error:

`"libflang.so: cannot open shared object file: No such file or directory"`

You will need to link the flang-install/lib directory to the environment variable LD_LIBRARY_PATH:

`export LD_LIBRARY_PATH="$(pwd)/lib"`

The resulting output should be:

`Hello World!`
