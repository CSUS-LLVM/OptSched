FROM rocm/dev-ubuntu-20.04:5.4.2
ARG amdgpu_arch

ENV AMDGPU_ARCH=$amdgpu_arch
ENV PATH=$PATH:/root/aco/util/:/opt/rocm/bin:/opt/rocm/llvm/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

WORKDIR /root/aco
RUN mkdir benchmarks
RUN mkdir util

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git cmake ninja-build clang ccache rocrand rocrand-dev rocm-hip-libraries
RUN pip3 install pandas tqdm
RUN git clone -b CGO24 https://github.com/balinck/llvm-project.git
RUN git clone -b CGO24 https://github.com/balinck/OptSched.git llvm-project/llvm/lib/Target/AMDGPU/OptSched
RUN cp -r llvm-project/llvm/lib/Target/AMDGPU/OptSched/example/optsched-cfg /root/.optsched-cfg
RUN cp -r llvm-project/llvm/lib/Target/AMDGPU/OptSched/util/ACO/* /root/aco/util/

RUN sed -i -e 's/ICMP_NE/_ICMP_NE/g' /opt/rocm/include/hip/amd_detail/amd_device_functions.h
RUN sed -i -e '81,753s/detail/::detail/g' /opt/rocm/include/hiprand/hiprand_kernel_hcc.h

RUN mkdir llvm-project/build && cd llvm-project/build && cmake -GNinja -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DLLVM_ENABLE_ASSERTIONS=False '-DLLVM_TARGETS_TO_BUILD=AMDGPU;X86' -DLLVM_ENABLE_Z3_SOLVER=OFF -DLLVM_CCACHE_BUILD=ON -DLLVM_CCACHE_DIR=$(pwd) -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" -DLLVM_AMDGPU_ALLOW_NPI_TARGETS=ON -DCMAKE_CXX_FLAGS="-O3" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ../llvm/ && ninja
ENV HIP_CLANG_PATH=/root/aco/llvm-project/build/bin

RUN git clone -b rocm-5.4.2 https://github.com/ROCmSoftwarePlatform/rocPRIM.git ./benchmarks/rocPRIM
