#include "compression.h"
#include "utils.h"
#include <cstdint>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

// Functor to calculate the encoded size of a run
struct RunSizeCalc {
  __host__ __device__ int operator()(const int &run_len) const {
    int chunks = run_len / 259;
    int rem = run_len % 259;
    int rem_size = (rem < 4) ? rem : 5;
    return (chunks * 5) + rem_size;
  }
};

__global__ void write_bzip2_rle1_kernel(const uint8_t *chars, const int *counts,
                                        const int *offsets, int num_runs,
                                        uint8_t *out_data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_runs) {
    return;
  }
  uint8_t c = chars[idx];
  int count = counts[idx];
  int offset = offsets[idx]; // Where this thread starts writing

  int chunks = count / 259;
  int rem = count % 259;

  // Write full 259-length chunks
  for (int i = 0; i < chunks; ++i) {
    out_data[offset++] = c;
    out_data[offset++] = c;
    out_data[offset++] = c;
    out_data[offset++] = c;
    out_data[offset++] = 255; // 255 + 4 = 259
  }

  // Write the remainder
  if (rem < 4) {
    for (int i = 0; i < rem; ++i) {
      out_data[offset++] = c;
    }
  } else {
    out_data[offset++] = c;
    out_data[offset++] = c;
    out_data[offset++] = c;
    out_data[offset++] = c;
    out_data[offset++] = (uint8_t)(rem - 4);
  }
}

int rle1_compress(const uint8_t *d_in, int in_len, uint8_t *&d_out, cudaStream_t stream) {
  if (in_len == 0)
    return 0;

  auto exec_policy = thrust::cuda::par.on(stream);
  thrust::device_ptr<const uint8_t> d_in_ptr(d_in);

  uint8_t *d_chars_raw;
  int *d_counts_raw;
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_chars_raw, in_len * sizeof(uint8_t), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_counts_raw, in_len * sizeof(int), stream));
  thrust::device_ptr<uint8_t> d_chars(d_chars_raw);
  thrust::device_ptr<int> d_counts(d_counts_raw);

  auto new_end = thrust::reduce_by_key(exec_policy, d_in_ptr, d_in_ptr + in_len,
                                       thrust::make_constant_iterator(1),
                                       d_chars, d_counts);
  int num_runs = new_end.first - d_chars;

  int *d_sizes_raw;
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_sizes_raw, num_runs * sizeof(int), stream));
  thrust::device_ptr<int> d_sizes(d_sizes_raw);
  thrust::transform(exec_policy, d_counts, d_counts + num_runs,
                    d_sizes, RunSizeCalc());

  int *d_offsets_raw;
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_offsets_raw, num_runs * sizeof(int), stream));
  thrust::device_ptr<int> d_offsets(d_offsets_raw);
  thrust::exclusive_scan(exec_policy, d_sizes, d_sizes + num_runs,
                         d_offsets);

  int last_size, last_offset;
  CUDA_ERROR_CHECK(cudaMemcpyAsync(&last_size, d_sizes_raw + num_runs - 1,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaMemcpyAsync(&last_offset, d_offsets_raw + num_runs - 1,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  int total_out_size = last_offset + last_size;

  CUDA_ERROR_CHECK(cudaMallocAsync((void **)&d_out, total_out_size, stream));

  constexpr int threadsPerBlock = 256;
  int blocks = (num_runs + threadsPerBlock - 1) / threadsPerBlock;

  write_bzip2_rle1_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
      d_chars_raw, d_counts_raw, d_offsets_raw, num_runs, d_out);

  CUDA_ERROR_CHECK(cudaFreeAsync(d_chars_raw, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_counts_raw, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_sizes_raw, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_offsets_raw, stream));

  return total_out_size;
}
