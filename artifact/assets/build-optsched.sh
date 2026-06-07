#!/usr/bin/env bash
# Clone, patch, and build CSUS-LLVM (TACO-26) with the OptSched GPU ACO
# scheduler integrated into the AMDGPU backend.  No GPU is required to build
# (the device arch is set explicitly); a GPU is only needed to *run*.
set -euo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
SRC="${SRC_DIR:-/work/llvm-project}"
JOBS="${JOBS:-$(nproc)}"

# GFX_ARCH may be a single arch ("gfx908") or a list ("gfx906;gfx908;gfx90a"
# or comma/space separated) to build one image that runs on multiple GPUs.
GFX_LIST=$(echo "${GFX_ARCH:-gfx908}" | tr ',;' '  ')
FIRST_ARCH=$(echo $GFX_LIST | awk '{print $1}')
GFX_CMAKE=$(echo $GFX_LIST | tr ' ' ';')                  # CMake list for AMDGPU_TARGETS
OFFLOAD_OPTS=""                                            # one --offload-arch per arch
for a in $GFX_LIST; do OFFLOAD_OPTS="${OFFLOAD_OPTS};--offload-arch=${a}"; done

LLVM_REPO=https://github.com/CSUS-LLVM/llvm-project.git
LLVM_SHA="${LLVM_SHA:-a9f4c7f0cbfeab934ca6ca1f1782a4414adbba06}"   # branch TACO-26
OPTSCHED_REPO=https://github.com/CSUS-LLVM/OptSched.git
OPTSCHED_SHA="${OPTSCHED_SHA:-60e7ffe34e6cff2574c4288b9b16e2e54835b40d}" # branch TACO-26

# ---- clone (shallow, pinned) ------------------------------------------------
git clone --filter=blob:none --no-checkout "$LLVM_REPO" "$SRC"
git -C "$SRC" checkout "$LLVM_SHA"
git clone --filter=blob:none --no-checkout "$OPTSCHED_REPO" \
    "$SRC/llvm/lib/Target/AMDGPU/OptSched"
git -C "$SRC/llvm/lib/Target/AMDGPU/OptSched" checkout "$OPTSCHED_SHA"

# ---- patch the LLVM CMake glue ---------------------------------------------
# llvm/CMakeLists.txt hardcodes the research machine's ROCm path (5.4.1) and GPU
# (gfx906); retarget to this image's ROCm and GPU.
sed -i "s#/opt/rocm-5.4.1#${ROCM}#g; s/gfx906/${GFX_CMAKE}/g" "$SRC/llvm/CMakeLists.txt"

# Build the rest of LLVM as plain C++ with clang++ (host only) and compile ONLY
# the OptSched .hip.cpp GPU sources as HIP. (Using hipcc globally device-compiles
# all of LLVM and fails to link host statics like ARMAttributeParser.)
cat >> "$SRC/llvm/lib/Target/AMDGPU/CMakeLists.txt" <<EOF

# --- Artifact patch: compile only the OptSched GPU sources (.hip.cpp) as HIP ---
file(GLOB_RECURSE OPTSCHED_HIP_SRCS RELATIVE \${CMAKE_CURRENT_SOURCE_DIR} OptSched/lib/*.hip.cpp)
set_source_files_properties(\${OPTSCHED_HIP_SRCS} PROPERTIES COMPILE_OPTIONS "-x;hip${OFFLOAD_OPTS}")
EOF

# ---- patch ROCm headers -----------------------------------------------------
ROCM_PATH="$ROCM" "$(dirname "$0")/patch-rocm-headers.sh"

# ---- configure & build ------------------------------------------------------
export HIP_PATH="$ROCM/hip" HCC_AMDGPU_TARGET="$FIRST_ARCH"
mkdir -p "$SRC/build" && cd "$SRC/build"
cmake -GNinja \
  -DCMAKE_C_COMPILER="$ROCM/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=False \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DLLVM_ENABLE_Z3_SOLVER=OFF \
  -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" \
  -DCMAKE_CXX_FLAGS="-I$ROCM/hip/include -I$ROCM/hsa/include" \
  -DBUILD_SHARED_LIBS=ON \
  ../llvm/
ninja -j"$JOBS"

echo "Build complete: $SRC/build/bin/clang++"
