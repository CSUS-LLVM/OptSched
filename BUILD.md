# Build Instructions

## Using Docker
A Dockerfile included in the "docker" directory can be used to build a container with our complete experimental environment. In this case, the only software dependencies are kernel-mode ROCm drivers, and Docker itself.

Assuming the ”docker” directory is in the current working directory, and the user wants to build for the ”gfx906” architecture, the build command is:
`docker build --build-arg="amdgpu_arch=’gfx906’" docker`

## Bare metal

**1. Install dependencies**

Ensure that `git`, `cmake`, `ninja-build`, `clang`, `ccache`, `rocrand`, `rocrand-dev`, and `rocm-hip-libraries` are installed using your system's package manager.

**2. Clone our [Modified llvm] from GitHub.**

```
git clone https://github.com/CSUS-LLVM/llvm-project.git
```

**3. Checkout CGO24.**

`
cd llvm-project && git checkout HIP_llvm15
`

**4. Clone OptSched into the AMDGPU directory and checkout CGO24 branch.**

```
cd llvm/lib/Target/AMDGPU
git clone https://github.com/CSUS-LLVM/OptSched.git
cd OptSched && git checkout CGO24
```

**5. Navigate back to llvm-project and create a build directory.**

`
cd ../../../../..
mkdir build && cd build
`

**6. Build LLVM and OptSched using Ninja. Building LLVM in debug mode is not guaranteed to work, asserts may be outdated. The build was only tested with the clang and clang++ bundled with rocm, but other versions should work, just change the paths as needed**

`cmake -GNinja -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DLLVM_ENABLE_ASSERTIONS=False '-DLLVM_TARGETS_TO_BUILD=AMDGPU;X86' -DLLVM_ENABLE_Z3_SOLVER=OFF -DLLVM_CCACHE_BUILD=ON -DLLVM_CCACHE_DIR=$(pwd) -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" -DLLVM_AMDGPU_ALLOW_NPI_TARGETS=ON -DCMAKE_CXX_FLAGS="-O3" -DBUILD_SHARED_LIBS=ON ../llvm/ -DCMAKE_BUILD_TYPE=Release`

`
ninja
`
<!-- Outside links -->
[Modified llvm]: https://github.com/CSUS-LLVM/llvm-project/tree/CGO24
