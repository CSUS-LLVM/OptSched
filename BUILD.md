# Building OptSched with LLVM 7, Clang, and Flang

## Setup

###### Only Ubuntu is known to work, but these instructions are likely to work other platforms. Please let us know if you successfully build on a different platform, so we can improve these instructions.

### Ubuntu

_**Attention:** Please only run these instructions on your own machine. If you are building on the **Tesla T4 machine**, please skip_
_forward to adding CUDA toolkit paths to your .bashrc._

###### Starting with a fresh install of [Ubuntu 20.04] is recommended.

#### Update Packages

###### This may not be necessary, but is often a good idea.

`
sudo apt update && sudo apt upgrade
`

#### Install [CMake] and [Git]

```
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt-get update
sudo apt install cmake git
```

#### Install [Ninja]


It is recommended to build LLVM using Ninja to avoid running out of memory during linking. Using Ninja should also result in faster builds.


##### Downloading manually (recommended)

`
wget -q https://github.com/ninja-build/ninja/releases/download/v1.9.0/ninja-linux.zip && unzip -q ninja-linux.zip && sudo cp ninja /usr/bin && rm ninja ninja-linux.zip
`

##### Using APT (Ubuntu 18.04 or later)

`
apt install ninja-build
`

#### Install Python2
`
sudo apt install python2
`

#### Install the latest CUDA toolkit. The last version successfully used for the project is 11.3.
###### First check if you already have it installed:
`
/usr/local/cuda/bin/nvcc --version
`
###### If already installed, skip the installation of CUDA to the optional adding of CUDA paths to your .bashrc
###### If not installed, install the CUDA toolkit from: https://developer.nvidia.com/cuda-downloads
###### Select Linux > x86_64 > Ubuntu > Your version of Ubuntu > deb (network) 
###### Then follow the presented installation instructions, simply copy the presented commands

#### (Required) Add the CUDA toolkit paths to your .bashrc for easier use of the tools and to allow cmake to find NVCC and cuda includes.
###### Open .bashrc
`
vim ~/.bashrc
`
###### Paste the following at the end of the file (be sure to change version numbers to match yours), save your changes. These environmental variables will now automatically be set when you log in/turn on your machine. You can also simply copy and paste these into terminal to set the variables.
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

#### Test CUDA compiler and driver install
###### Reboot the machine if you installed the CUDA toolkit:
`
sudo reboot
`
###### note if you did not set your environmental variables, you will need to use the full path to the binary. CUDA tools are installed to /usr/local/cuda/bin by default
```
nvcc --version
nvidia-smi
```

## Manual Build

To manually build this, such as if you want to place OptSched inside an existing clone of flang llvm.

**1. Set up install directory and clone the [Flang LLVM source code] from GitHub.**

```
mkdir -p v7flang/flang-install
cd v7flang/flang-install
FLANG_INSTALL=`pwd`
cd ..
git clone https://github.com/flang-compiler/llvm.git
```

**2. Checkout Flang LLVM release 7.**

`
cd llvm && git checkout release_70
`

**3. Clone OptSched into the projects directory and checkout GPU_ACO branch.**

```
cd projects && git clone https://github.com/CSUS-LLVM/OptSched
cd OptSched && git checkout GPU_ACO
cd ../..
```

**4. Create a build directory.**

`
mkdir build && cd build
`

**5. Apply [this patch][spilling-info-patch] to print spilling info.**

`
git am ../projects/OptSched/patches/llvm7.0/llvm7-print-spilling-info.patch
`

**6. Move PointerIntPair to public from protected.**
###### I am not sure why this is required, but the project will not build if `PointerIntPair<InstrTy*, 1, bool> I;` is in protected. Open the file in vim, move the `PointerIntPair<InstrTy*, 1, bool> I;` to public, and save your work with `:wq`

`
vim ../include/llvm/IR/CallSite.h
`

**7. Build LLVM and OptSched using Ninja**
###### Note: You must change the -DCMAKE_CUDA_ARCHITECTURES flag to match the Compute Capability of your GPU. The Tesla T4 is CC 7.5.The -DCMAKE_CXX_FLAGS_INIT flag must point to your CUDA install's include folder, the default directory is used here. If you want to build the device code in debug mode, replace -lineinfo with -G in the -DCMAKE_CUDA_FLAGS_INIT variable. Building LLVM in debug mode is not guaranteed to work, asserts may be outdated.

```
cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS_INIT='-lineinfo --ptxas-options=-v' -DCMAKE_CXX_FLAGS_INIT='-isystem/usr/local/cuda/include' -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD='X86;AArch64' ..
ninja install
#go back to the directory v7flang
cd ../..
```

**8. Install flang-driver**
###### Note: you can modify `make -j2 install` to use as many threads as you have available, using `-j2` only uses 2 threads.

```
git clone https://github.com/flang-compiler/flang-driver.git
cd flang-driver
git checkout release_70
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DLLVM_CONFIG=$FLANG_INSTALL/bin/llvm-config -DCMAKE_BUILD_TYPE=Release ..
make -j2 install
#get back to v7flang
cd ../../
```
**9. Install flang openmp**

```
git clone https://github.com/llvm-mirror/openmp.git
cd openmp
git checkout release_70
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DCMAKE_CXX_COMPILER=$FLANG_INSTALL/bin/clang++ -DCMAKE_C_COMPILER=$FLANG_INSTALL/bin/clang -DCMAKE_BUILD_TYPE=Release ..
make -j2 install
#get back to v7flang
cd ../..
```

**10. Build Flang**
```
#build libpgmath
git clone https://github.com/flang-compiler/flang.git
cd flang/runtime/libpgmath
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DCMAKE_CXX_COMPILER=$FLANG_INSTALL/bin/clang++ -DCMAKE_C_COMPILER=$FLANG_INSTALL/bin/clang -DCMAKE_BUILD_TYPE=Release ..
make -j2 install
#go back to v7flang/flang
cd ../../..
#build Flang
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$FLANG_INSTALL -DLLVM_CONFIG=$FLANG_INSTALL/bin/llvm-config -DCMAKE_CXX_COMPILER=$FLANG_INSTALL/bin/clang++ -DCMAKE_C_COMPILER=$FLANG_INSTALL/bin/clang -DCMAKE_Fortran_COMPILER=$FLANG_INSTALL/bin/flang -DCMAKE_BUILD_TYPE=Release ..
make -j2 install
#back out of the directory structure we just created
cd ../../..
```

**11. Add $FLANG_INSTALL to your .bashrc and add the fortran runtime to your library path. (Optional)**
```
echo "export FLANG_INSTALL=$FLANG_INSTALL" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$FLANG_INSTALL/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```
You now have all of the requirements to run CPU2017. Clang and Flang are located in $FLANG_INSTALL/bin (v7flang/flang-install/bin/) and the OptSched.so plugin is located at v7flang/llvm/build/lib/OptSched.so.

<!-- Outside links -->
[ubuntu 20.04]: http://releases.ubuntu.com/20.04/
[cmake]: https://cmake.org/
[git]: https://git-scm.com/
[homebrew]: https://brew.sh/
[cmake downloads page]: https://cmake.org/download/
[ninja]: https://ninja-build.org/
[flang llvm source code]: https://github.com/flang-compiler/llvm.git
[building with cmake]: https://llvm.org/docs/CMake.html
[spilling-info-patch]: patches/llvm7.0/llvm7-print-spilling-info.patch
