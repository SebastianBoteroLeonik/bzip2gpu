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

__global__ void mtf_per_thread_bwt(const uint8_t *d_orig, const int *d_sa,
                                   MTFState *d_partial, int N, int J) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int start_idx = tid * J;
  if (start_idx >= N)
    return;

  int end_idx = min(start_idx + J, N);

  MTFState state;
  state.size = 0;
  bool seen[256] = {false};

  for (int i = end_idx - 1; i >= start_idx; --i) {
    int sa_val = d_sa[i];
    uint8_t c = (sa_val == 0) ? d_orig[N - 1] : d_orig[sa_val - 1];
    if (!seen[c]) {
      state.chars[state.size++] = c;
      seen[c] = true;
    }
  }

  d_partial[tid] = state;
}

__global__ void apply_mtf_kernel_bwt(const uint8_t *d_orig, const int *d_sa,
                                     uint8_t *d_out, const MTFState *d_scanned,
                                     int N, int J) {
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
    int sa_val = d_sa[i];
    uint8_t c = (sa_val == 0) ? d_orig[N - 1] : d_orig[sa_val - 1];

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
                       uint8_t *&symbols_table, cudaStream_t stream) {
  char *symbols_flags;
  CUDA_ERROR_CHECK(cudaMallocAsync(&symbols_flags, 256, stream));
  CUDA_ERROR_CHECK(cudaMemsetAsync(symbols_flags, 0, 256, stream));
  constexpr int threadsPerBlock = 256;
  int blocksPerGrid = (d_in_len + threadsPerBlock - 1) / threadsPerBlock;
  find_uniques_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      d_in, symbols_flags, d_in_len);
  char symbols_flags_host[256];
  CUDA_ERROR_CHECK(cudaMemcpyAsync(symbols_flags_host, symbols_flags,
                                   sizeof(symbols_flags_host),
                                   cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
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
  CUDA_ERROR_CHECK(cudaFreeAsync(symbols_flags, stream));
  return count;
}

struct is_zero {
  __host__ __device__ bool operator()(const int x) { return x == 0; }
};

int fmtf(const uint8_t *in_original, const int *in_suffix_array, int in_len,
         uint8_t *&d_out, int &orig_ptr, cudaStream_t stream) {
  if (in_len == 0)
    return 0;

  const int N = in_len;
  const int J = 64;
  int num_threads = (N + J - 1) / J;

  CUDA_ERROR_CHECK(
      cudaMallocAsync((void **)&d_out, N * sizeof(uint8_t), stream));

  thrust::device_vector<MTFState> d_partial(num_threads);
  thrust::device_vector<MTFState> d_scanned(num_threads);

  int blockSize = 256;
  int gridSize = (num_threads + blockSize - 1) / blockSize;

  mtf_per_thread_bwt<<<gridSize, blockSize, 0, stream>>>(
      in_original, in_suffix_array, thrust::raw_pointer_cast(d_partial.data()),
      N, J);

  uint8_t *symbol_table;
  int table_size =
      make_symbols_table(in_original, in_len, symbol_table, stream);

  MTFState identity;
  identity.size = table_size;
  for (int i = 0; i < table_size; ++i) {
    identity.chars[i] = symbol_table[i];
  }

  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_partial.begin(),
                         d_partial.end(), d_scanned.begin(), identity,
                         MTFAppendUnique());

  apply_mtf_kernel_bwt<<<gridSize, blockSize, 0, stream>>>(
      in_original, in_suffix_array, d_out,
      thrust::raw_pointer_cast(d_scanned.data()), N, J);

  delete[] symbol_table;

  auto sa_ptr = thrust::device_pointer_cast(in_suffix_array);
  auto iter = thrust::find_if(thrust::cuda::par.on(stream), sa_ptr,
                              sa_ptr + in_len, is_zero());
  orig_ptr = iter - sa_ptr;
  return table_size;
}
