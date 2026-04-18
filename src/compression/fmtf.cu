#include "compression.h"
#include "utils.h"

#include <cstdint>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

struct MTFState {
  uint8_t chars[256];
  uint16_t size;
};

struct MTFAppendUnique {
  __host__ __device__ MTFState operator()(const MTFState &left,
                                          const MTFState &right) const {
    MTFState res;
    res.size = right.size;

    bool seen[256] = {false};

    for (int i = 0; i < right.size; ++i) {
      res.chars[i] = right.chars[i];
      seen[right.chars[i]] = true;
    }

    for (int i = 0; i < left.size; ++i) {
      uint8_t c = left.chars[i];
      if (!seen[c]) {
        res.chars[res.size++] = c;
        seen[c] = true;
      }
    }
    return res;
  }
};

__global__ void mtf_per_thread(const uint8_t *d_in, MTFState *d_partial, int N,
                               int J) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int start_idx = tid * J;
  if (start_idx >= N)
    return;

  int end_idx = min(start_idx + J, N);

  MTFState state;
  state.size = 0;
  bool seen[256] = {false};

  for (int i = end_idx - 1; i >= start_idx; --i) {
    uint8_t c = d_in[i];
    if (!seen[c]) {
      state.chars[state.size++] = c;
      seen[c] = true;
    }
  }

  d_partial[tid] = state;
}

__global__ void apply_mtf_kernel(const uint8_t *d_in, uint8_t *d_out,
                                 const MTFState *d_scanned, int N, int J) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int start_idx = tid * J;
  if (start_idx >= N)
    return;

  int end_idx = min(start_idx + J, N);

  uint8_t mtfList[256];
  for (int i = 0; i < 256; ++i) {
    mtfList[i] = d_scanned[tid].chars[i];
  }

  for (int i = start_idx; i < end_idx; ++i) {
    uint8_t c = d_in[i];

    int pos = 0;
    for (; pos < 256; ++pos) {
      if (mtfList[pos] == c)
        break;
    }

    d_out[i] = (uint8_t)pos;

    for (int k = pos; k > 0; --k) {
      mtfList[k] = mtfList[k - 1];
    }
    mtfList[0] = c;
  }
}

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

void fmtf(const uint8_t *d_in, int in_len, uint8_t *&d_out) {
  if (in_len == 0)
    return;

  const int N = in_len;
  const int J = 64;
  int num_threads = (N + J - 1) / J;

  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_out, N * sizeof(uint8_t)));

  thrust::device_vector<MTFState> d_partial(num_threads);
  thrust::device_vector<MTFState> d_scanned(num_threads);

  int blockSize = 256;
  int gridSize = (num_threads + blockSize - 1) / blockSize;

  mtf_per_thread<<<gridSize, blockSize>>>(
      d_in, thrust::raw_pointer_cast(d_partial.data()), N, J);
  cudaDeviceSynchronize();

  uint8_t *symbol_table;
  int table_size = make_symbols_table(d_in, in_len, symbol_table);

  MTFState identity;
  identity.size = table_size;
  for (int i = 0; i < table_size; ++i) {
    identity.chars[i] = symbol_table[i];
  }

  thrust::exclusive_scan(thrust::device, d_partial.begin(), d_partial.end(),
                         d_scanned.begin(), identity, MTFAppendUnique());

  apply_mtf_kernel<<<gridSize, blockSize>>>(
      d_in, d_out, thrust::raw_pointer_cast(d_scanned.data()), N, J);
  cudaDeviceSynchronize();
}
