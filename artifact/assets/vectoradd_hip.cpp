// Minimal HIP smoke test: vector add on the GPU, verify result on host.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t e = (cmd);                                                       \
    if (e != hipSuccess) {                                                      \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),         \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

__global__ void vadd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);

  int dev = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  printf("Running on device: %s (gcnArch %s)\n", prop.name, prop.gcnArchName);

  float *ha = (float *)malloc(bytes), *hb = (float *)malloc(bytes),
        *hc = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) {
    ha[i] = (float)i;
    hb[i] = (float)(2 * i);
  }

  float *da, *db, *dc;
  HIP_CHECK(hipMalloc(&da, bytes));
  HIP_CHECK(hipMalloc(&db, bytes));
  HIP_CHECK(hipMalloc(&dc, bytes));
  HIP_CHECK(hipMemcpy(da, ha, bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, hb, bytes, hipMemcpyHostToDevice));

  int threads = 256, blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(vadd, dim3(blocks), dim3(threads), 0, 0, da, db, dc, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hc, dc, bytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < n; i++)
    if (hc[i] != ha[i] + hb[i])
      errors++;

  printf(errors ? "FAILED: %d mismatches\n" : "PASSED: vector add correct\n",
         errors);

  free(ha); free(hb); free(hc);
  HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dc));
  return errors ? 1 : 0;
}
