# OptSched GPU-ACO Scheduler — Reproducible Artifact

This artifact builds the research compiler from the paper: a modified LLVM
([CSUS-LLVM/llvm-project @ TOPLAS-25](https://github.com/CSUS-LLVM/llvm-project/tree/TOPLAS-25))
whose AMDGPU backend uses **OptSched**
([CSUS-LLVM/OptSched @ TOPLAS-25](https://github.com/CSUS-LLVM/OptSched/tree/TOPLAS-25)),
an instruction scheduler that runs an **Ant Colony Optimization (ACO)
algorithm on the GPU** via HIP.

It is packaged so the build and a functional demonstration can be reproduced on
a different machine. It does **not** reproduce the paper's performance numbers —
it verifies that the compiler builds and that the GPU ACO scheduler runs.

> This directory (`artifact/`) is self-contained: run all `docker` commands
> below from here. The build clones the modified LLVM and OptSched at pinned
> commits, so it does not depend on the surrounding checkout.

* Base image: `rocm/dev-ubuntu-22.04:5.4.2`
* Default GPU target: `gfx908` (AMD Instinct MI100). The original experiments
  used `gfx906` (Radeon VII); override with `--build-arg GFX_ARCH=gfxNNN`.
* Pinned commits: llvm `a9f4c7f`, OptSched `60e7ffe`.

## Build (no GPU required)

```bash
docker build -t optsched-artifact .
# different GPU: docker build --build-arg GFX_ARCH=gfx906 -t optsched-artifact .
```

The build takes ~15–20 min and needs no GPU (the device arch is fixed at
configure time). A GPU is only needed to *run* the verification.

## Verify (GPU required)

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  optsched-artifact /opt/artifact/verify.sh
```

`verify.sh` checks: (1) the GPU is visible, (2) the freshly-built compiler
builds and runs a HIP program on the GPU, (3) OptSched's on-GPU ACO scheduler
launches during compilation (`Launching Dev_ACO with N blocks of M threads`)
and the resulting program produces correct results.

For an interactive shell instead: drop the `/opt/artifact/verify.sh` argument.

## Building for a different GPU

The GPU architecture is a single build-time parameter, `GFX_ARCH` (default
`gfx908`, the MI100). To target a different AMD GPU, set it once:

```bash
# Radeon VII (original experiments)
docker build --build-arg GFX_ARCH=gfx906 -t optsched-artifact:gfx906 .
# MI200 / MI250
docker build --build-arg GFX_ARCH=gfx90a -t optsched-artifact:gfx90a .
```

`GFX_ARCH` flows everywhere it is needed: the `AMDGPU_TARGETS` in
`llvm/CMakeLists.txt` and the `--offload-arch` used to compile OptSched's
`.hip.cpp` device sources. `verify.sh` does not need it — it auto-detects the
arch of whatever GPU is present (via `rocminfo`). To find a card's arch
manually, run `rocminfo | grep gfx`.

**Important — one image targets one GPU.** The scheduler's *own* ACO code (the
kernels it launches during compilation) is compiled only for the arch you build
with. Running an image's compiler on a *different* GPU than it was built for will
fail at the ACO kernel launch (no matching code object). So build a matching
image per GPU, **or** build one image that covers several GPUs by listing
multiple archs (a fat binary):

```bash
docker build --build-arg GFX_ARCH="gfx906;gfx908;gfx90a" -t optsched-artifact:multi .
```

(`build-optsched.sh` expands the list into one `--offload-arch=` flag per arch,
producing a fat binary; comma or space separators work too. Each extra arch adds
build time and image size. Only `gfx908` has been runtime-verified on this
hardware — the multi-arch path is wired up but other archs are untested here.)

User programs compiled by this clang can still target *any* AMDGPU arch
normally — the single-arch limitation applies only to the scheduler's own
on-GPU ACO code, not to the code it generates.

## Loading the prebuilt image on another machine

```bash
gunzip -c optsched-artifact.tar.gz | docker load
```

## Using the compiler directly

The compiler is at `/work/llvm-project/build/bin` (already on `PATH`). It is a
plain LLVM `clang`, not the `hipcc` wrapper, so link the HIP runtime explicitly:

```bash
clang++ -x hip --offload-arch=gfx908 --rocm-path=/opt/rocm \
        -mllvm -optsched-cfg=$HOME/.optsched-cfg \
        my_program.hip.cpp -L/opt/rocm/lib -lamdhip64 -o my_program
```

OptSched is the **default** AMDGPU machine scheduler (no flag needed). Its
configuration lives in `~/.optsched-cfg` (also in `assets/optsched-cfg`). The
on-GPU ACO only engages for scheduling regions of ≥ 10 instructions on the
second pass; tiny kernels are scheduled but do not launch the device ACO.

## What was changed to make it build (see `assets/`)

* `patch-rocm-headers.sh` — renames the `ICMP_NE` macro in
  `amd_device_functions.h` (collides with `llvm::CmpInst::ICMP_NE`) and the
  top-level `detail` namespace in `hiprand_kernel_hcc.h` (ambiguous with
  `llvm::detail`).
* `build-optsched.sh` — clones/pins the repos, retargets the hardcoded
  `/opt/rocm-5.4.1` + `gfx906` in `llvm/CMakeLists.txt`, and compiles **only**
  the OptSched `.hip.cpp` sources as HIP (the rest of LLVM is built as ordinary
  C++ with `clang++`; using `hipcc` for everything makes the device link fail).
* `optsched-cfg/sched.ini` — the upstream example is stale for this branch;
  missing keys are added and two-pass scheduling is enabled (required — the
  scheduler dereferences per-region state that only exists in two-pass mode).

## Files

| Path | Purpose |
|------|---------|
| `Dockerfile` | builds the artifact image |
| `assets/build-optsched.sh` | clone + patch + build LLVM/OptSched |
| `assets/patch-rocm-headers.sh` | ROCm header fixes |
| `assets/optsched-cfg/` | OptSched runtime configuration |
| `assets/verify.sh` | end-to-end verification |
| `assets/vectoradd_hip.cpp` | minimal HIP smoke test |
| `assets/aco_demo.hip.cpp` | large-region kernel that triggers the GPU ACO |
| `assets/big_region.hip.cpp` | device-only variant of the above |
