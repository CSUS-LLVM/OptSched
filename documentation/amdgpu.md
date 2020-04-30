# Using OptSched on the AMDGPU machine (optimizer-amd)

## Building OptSched

The superbuild script at **cmake/amdgpu-superbuild** can configure and build this for you.
However, you first need to build clang:

### Building Clang

```
git clone https://github.com/llvm/llvm-project

cd llvm-project

mkdir build && cd build

cmake -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_BUILD_TYPE=Release '-DLLVM_TARGETS_TO_BUILD=X86' -DLLVM_BUILD_TOOLS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON ../llvm

make -j32
```

The built **bin/clang** and **bin/clang++** are what we will use to build OptSched and ROCm OpenCL.

### Building ROCm OpenCL and OptSched

**1. Create a build directory.**

`
mkdir build && cd build
`

**2. Configure**

We want to configure against the **cmake/amdgpu-superbuild** directory.

`
cmake -DCMAKE_CXX_COMPILER=/home/me/llvm-project/build/bin/clang++ -DCMAKE_C_COMPILER=/home/me/llvm-project/build/bin/clang -DCMAKE_BUILD_TYPE=Release ../cmake/amdgpu-superbuild
`

If you already have Google's **repo** tool installed, this will use that.
Otherwise, this will error. You can have this automatically download **repo** to this
build directory by passing `-DOPTSCHEDSUPER_AUTO_DOWNLOAD_REPO=ON`

**3. Run the Super-build**

`
make -j32
`

The super-build script will clone, configure, and build ROCm and OptSched.
The sub-build directory will be at **ROCm-prefix/src/ROCm-build**.
If you need to run the build again, you may need to run the build command from there.

As part of this, it will generate a **rocm-env.shrc** file. You should `source` this
file as part of your **.bashrc**; it sets up necessary environment variables.
