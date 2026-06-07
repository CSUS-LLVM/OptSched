# Artifact: CSUS-LLVM (TACO-26) + OptSched GPU Ant-Colony-Optimization
# instruction scheduler for the AMDGPU backend, implemented in HIP.
#
# Build (no GPU needed):
#   docker build -t optsched-artifact .
# Run / verify (GPU required):
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri \
#       --group-add video --group-add render --security-opt seccomp=unconfined \
#       optsched-artifact /opt/artifact/verify.sh
#
# Override the target GPU at build time with --build-arg GFX_ARCH=gfxNNN
# (research machine used gfx906; this artifact defaults to gfx908 / MI100).
FROM rocm/dev-ubuntu-22.04:5.4.2

ARG GFX_ARCH=gfx908
ARG JOBS
ENV ROCM_PATH=/opt/rocm \
    GFX_ARCH=${GFX_ARCH} \
    DEBIAN_FRONTEND=noninteractive

# Build tools + rocRAND (provides the hipRAND headers/cmake the build needs).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake ninja-build git python3 python-is-python3 build-essential \
        ca-certificates rocrand-dev clinfo && \
    rm -rf /var/lib/apt/lists/*

# Patch scripts, build script, config, and test programs.
COPY assets/ /opt/artifact/
RUN chmod +x /opt/artifact/*.sh

# Clone (pinned), patch ROCm headers + LLVM CMake glue, and build.
RUN SRC_DIR=/work/llvm-project GFX_ARCH=${GFX_ARCH} JOBS=${JOBS:-$(nproc)} \
        /opt/artifact/build-optsched.sh

# Install the example OptSched configuration to its default search location.
RUN cp -r /opt/artifact/optsched-cfg /root/.optsched-cfg

ENV LLVM_BIN=/work/llvm-project/build/bin \
    PATH=/work/llvm-project/build/bin:/opt/rocm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib

WORKDIR /opt/artifact
CMD ["/bin/bash"]
