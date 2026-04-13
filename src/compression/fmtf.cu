#include "compression.h"
#include "utils.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>

// #include <thrust/device_vector.h>
// #include <thrust/iterator/constant_iterator.h>

__device__ void MTF_per_thread(uint8_t *mtfin) {}

__global__ void calculate_partial_MTF_Lists(uint8_t *d_in, int d_in_len,
                                            int thread_workload,
                                            uint8_t *MTF_lists) {}

__global__ void find_uniques_kernel(const uint8_t *input, char *global_flags,
                                    int n) {
  __shared__ uint8_t local_flags[256];
  local_flags[threadIdx.x] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    local_flags[input[i]] = 1;
  }
  __syncthreads();
  if (local_flags[threadIdx.x] == 1) {
    global_flags[threadIdx.x] = 1;
  }
}

int make_symbols_table(const uint8_t *d_in, int d_in_len,
                       uint8_t *&symbols_table) {
  char *symbols_flags;
  CUDA_ERROR_CHECK(cudaMalloc(&symbols_flags, 256));
  CUDA_ERROR_CHECK(cudaMemset(symbols_flags, 0, 256));
  constexpr int threadsPerBlock = 256;
  int blocksPerGrid = (d_in_len + threadsPerBlock - 1) / threadsPerBlock;
  find_uniques_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, symbols_flags,
                                                          d_in_len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  char symbols_flags_host[256];
  CUDA_ERROR_CHECK(cudaMemcpy(symbols_flags_host, symbols_flags,
                              sizeof(symbols_flags_host),
                              cudaMemcpyDeviceToHost));
  int count = 0;
  for (int i = 0; i < 256; i++) {
    count += symbols_flags_host[i];
  }
  symbols_table = new uint8_t[count];
  int j = 0;
  for (int i = 0; i < 256; i++) {
    if (symbols_flags_host[i]) {
      symbols_table[j++] = i;
    }
  }
  return count;
}
