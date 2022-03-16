// File contains #defs that are common to a lot of device code

// Formula for determining global thread ID on device
#define GLOBALTID blockIdx.x * blockDim.x + threadIdx.x

// Check for and print out errors on CUDA API calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
