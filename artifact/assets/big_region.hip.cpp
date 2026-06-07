// A kernel with one large basic block (a long chain/forest of independent
// arithmetic) so the AMDGPU scheduler sees a region well above REGION_MIN_SIZE.
// Used to exercise OptSched's on-GPU ACO scheduler during compilation.
#include <hip/hip_runtime.h>

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
