#!/usr/bin/env bash
# Run inside the container (with the GPU passed in) to verify the artifact:
#  1. the GPU is visible,
#  2. the freshly-built compiler builds & runs a HIP program on the GPU,
#  3. OptSched's on-GPU ACO scheduler runs during compilation and the
#     resulting code executes correctly.
set -euo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"
# Detect the arch of the GPU actually present; fall back to the first arch the
# image was built for (GFX_ARCH may be a list for multi-GPU images).
GFX="$("$ROCM/bin/rocminfo" 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)"
GFX="${GFX:-$(echo "${GFX_ARCH:-gfx908}" | tr ',;' '  ' | awk '{print $1}')}"
BIN="${LLVM_BIN:-/work/llvm-project/build/bin}"
CFG="${OPTSCHED_CFG:-$HOME/.optsched-cfg}"
ASSETS="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
CXX="$BIN/clang++"
HIPFLAGS=(-x hip --offload-arch="$GFX" --rocm-path="$ROCM" -L"$ROCM/lib" -lamdhip64)
tmp="$(mktemp -d)"

line() { printf '\n========== %s ==========\n' "$1"; }

line "1. GPU visibility"
"$ROCM/bin/rocminfo" | grep -E "Name:.*$GFX|Marketing Name:.*Instinct" | head
clinfo 2>/dev/null | grep -E "Board name|Name:.*$GFX" | head -2 || true

line "2. Built compiler"
"$CXX" --version | head -2

line "3. HIP smoke test (vector add) on the GPU"
"$CXX" "${HIPFLAGS[@]}" "$ASSETS/vectoradd_hip.cpp" -o "$tmp/vadd"
"$tmp/vadd"

line "4. Build aco_demo WITH OptSched GPU ACO scheduler"
"$CXX" -O3 "${HIPFLAGS[@]}" -mllvm -optsched-cfg="$CFG" \
    "$ASSETS/aco_demo.hip.cpp" -o "$tmp/aco_demo" 2> "$tmp/aco_build.log"
launches=$(grep -c "Launching Dev_ACO" "$tmp/aco_build.log" || true)
echo "OptSched on-GPU ACO kernel launches during compilation: $launches"
grep -E "Launching Dev_ACO|ACO finished after|Post Kernel Error" "$tmp/aco_build.log" | head

line "5. Run the OptSched-scheduled program on the GPU"
"$tmp/aco_demo"

line "RESULT"
if [[ "$launches" -ge 1 ]]; then
  echo "OK: compiler built, HIP runs on $GFX, and OptSched's GPU ACO scheduler executed."
else
  echo "WARNING: program built/ran but no Dev_ACO launch detected (check $CFG)."
fi
rm -rf "$tmp"
