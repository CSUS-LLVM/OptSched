// Full, runnable HIP program whose kernel has a large scheduling region,
// so building it with OptSched exercises the on-GPU ACO scheduler, and
// running it verifies the OptSched-scheduled device code is correct.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t e = (cmd);                                                       \
    if (e != hipSuccess) {                                                      \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),          \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

__global__ void big(float *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = in[i];
  float b = a * 1.1f + 0.5f;
  float c = a * 2.2f + 1.5f;
  float d = a * 3.3f + 2.5f;
  float e = a * 4.4f + 3.5f;
  float f = b * c + d;
  float g = c * d + e;
  float h = d * e + b;
  float p = e * b + c;
  float q = f * g + h;
  float r = g * h + p;
  float s = h * p + f;
  float t = p * f + g;
  float u = q * r - s;
  float v = r * s - t;
  float w = s * t - q;
  float x = t * q - r;
  float y = u * v + w * x;
  float z = v * w + x * u;
  float k = w * x + u * v;
  float l = x * u + v * w;
  out[i] = y * z + k * l + q * r + s * t + a;
}

// Reference computation on the host to validate the GPU result.
static float ref(float a) {
  float b = a * 1.1f + 0.5f, c = a * 2.2f + 1.5f, d = a * 3.3f + 2.5f,
        e = a * 4.4f + 3.5f;
  float f = b * c + d, g = c * d + e, h = d * e + b, p = e * b + c;
  float q = f * g + h, r = g * h + p, s = h * p + f, t = p * f + g;
  float u = q * r - s, v = r * s - t, w = s * t - q, x = t * q - r;
  float y = u * v + w * x, z = v * w + x * u, k = w * x + u * v, l = x * u + v * w;
  return y * z + k * l + q * r + s * t + a;
}

int main() {
  const int n = 1 << 16;
  const size_t bytes = n * sizeof(float);
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);

  float *hin = (float *)malloc(bytes), *hout = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) hin[i] = (float)(i % 97) * 0.3f;

  float *din, *dout;
  HIP_CHECK(hipMalloc(&din, bytes));
  HIP_CHECK(hipMalloc(&dout, bytes));
  HIP_CHECK(hipMemcpy(din, hin, bytes, hipMemcpyHostToDevice));

  int threads = 256, blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(big, dim3(blocks), dim3(threads), 0, 0, dout, din, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hout, dout, bytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < n; i++) {
    float r = ref(hin[i]);
    float diff = hout[i] - r;
    if (diff < 0) diff = -diff;
    if (diff > 1e-2f * (r < 0 ? -r : r) + 1e-3f) errors++;
  }
  printf(errors ? "FAILED: %d mismatches\n" : "PASSED: OptSched-scheduled kernel correct\n",
         errors);
  free(hin); free(hout);
  HIP_CHECK(hipFree(din)); HIP_CHECK(hipFree(dout));
  return errors ? 1 : 0;
}
