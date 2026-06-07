#!/usr/bin/env bash
# Patch ROCm 5.4.2 headers so the OptSched (.hip.cpp) sources, which include
# both HIP device headers and LLVM headers, compile cleanly.
#
# 1) hip/amd_detail/amd_device_functions.h defines a macro `ICMP_NE 33` that
#    clobbers llvm::CmpInst::ICMP_NE.  Rename the macro -> HIP_ICMP_NE.
# 2) hiprand_kernel_hcc.h opens a top-level `namespace detail` that becomes
#    ambiguous with llvm::detail.  Rename it -> hiprand_detail.
#
# Idempotent: \bdetail\b does not match inside "hiprand_detail".
set -euo pipefail

ROCM="${ROCM_PATH:-/opt/rocm}"

adf="$ROCM/include/hip/amd_detail/amd_device_functions.h"
if [[ -f "$adf" ]]; then
  sed -i 's/\bICMP_NE\b/HIP_ICMP_NE/g' "$adf"
  echo "patched: $adf (ICMP_NE -> HIP_ICMP_NE)"
fi

# Patch every real copy of the hiprand kernel header.
while IFS= read -r f; do
  sed -i 's/\bdetail\b/hiprand_detail/g' "$f"
  echo "patched: $f (namespace detail -> hiprand_detail)"
done < <(find -L "$ROCM" -name hiprand_kernel_hcc.h ! -path '*lib-debug*' 2>/dev/null | sort -u)

echo "ROCm header patches applied."
