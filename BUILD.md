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

## Build Instructions


**1. Clone our [Modified llvm] from GitHub.**

```
git clone https://github.com/JoshHuttonCode/llvm-project.git
```

**2. Checkout HIP_llvm15.**

`
cd llvm-project && git checkout HIP_llvm15
`

**3. Clone OptSched into the AMDGPU directory and checkout HIP_ACO_llvm15 branch.**

```
cd llvm/lib/Target/AMDGPU
git clone https://github.com/JoshHuttonCode/OptSched.git
cd OptSched && git checkout HIP_ACO_llvm15
```

**4. Navigate back to llvm-project and create a build directory.**

`
cd ../../../../..
mkdir build && cd build
`

**5. Build LLVM and OptSched using Ninja. Building LLVM in debug mode is not guaranteed to work, asserts may be outdated. The build was only tested with the clang and clang++ bundled with rocm, but other versions should work, just change the paths as needed**

`cmake -GNinja -DCMAKE_C_COMPILER=/opt/rocm-5.4.1/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-5.4.1/llvm/bin/clang++ -DLLVM_ENABLE_ASSERTIONS=False '-DLLVM_TARGETS_TO_BUILD=AMDGPU;X86' -DLLVM_ENABLE_Z3_SOLVER=OFF -DLLVM_CCACHE_BUILD=ON -DLLVM_CCACHE_DIR=$(pwd) -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" -DLLVM_AMDGPU_ALLOW_NPI_TARGETS=ON -DCMAKE_CXX_FLAGS="-O3" -DBUILD_SHARED_LIBS=ON ../llvm/ -DCMAKE_BUILD_TYPE=Release`

`
ninja
`

<!-- Outside links -->
[ubuntu 20.04]: http://releases.ubuntu.com/20.04/
[cmake]: https://cmake.org/
[git]: https://git-scm.com/
[homebrew]: https://brew.sh/
[cmake downloads page]: https://cmake.org/download/
[ninja]: https://ninja-build.org/
[Modified llvm]: https://github.com/JoshHuttonCode/llvm-project/tree/HIP_llvm15
[building with cmake]: https://llvm.org/docs/CMake.html
