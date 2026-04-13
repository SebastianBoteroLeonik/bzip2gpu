#include "compression.h"
#include "utils.h"
#include <cstdint>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

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

int rle1_compress(const uint8_t *d_in, int in_len, uint8_t *&d_out) {
  if (in_len == 0)
    return 0;

  thrust::device_ptr<const uint8_t> d_in_ptr(d_in);

  thrust::device_vector<uint8_t> d_chars(in_len);
  thrust::device_vector<int> d_counts(in_len);

  auto new_end = thrust::reduce_by_key(d_in_ptr, d_in_ptr + in_len,
                                       thrust::make_constant_iterator(1),
                                       d_chars.begin(), d_counts.begin());
  int num_runs = new_end.first - d_chars.begin();

  thrust::device_vector<int> d_sizes(num_runs);
  thrust::transform(d_counts.begin(), d_counts.begin() + num_runs,
                    d_sizes.begin(), RunSizeCalc());

  thrust::device_vector<int> d_offsets(num_runs);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.begin() + num_runs,
                         d_offsets.begin());

  int last_size = d_sizes[num_runs - 1];
  int last_offset = d_offsets[num_runs - 1];
  int total_out_size = last_offset + last_size;

  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_out, total_out_size));

  constexpr int threadsPerBlock = 256;
  int blocks = (num_runs + threadsPerBlock - 1) / threadsPerBlock;

  write_bzip2_rle1_kernel<<<blocks, threadsPerBlock>>>(
      thrust::raw_pointer_cast(d_chars.data()),
      thrust::raw_pointer_cast(d_counts.data()),
      thrust::raw_pointer_cast(d_offsets.data()), num_runs, d_out);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  return total_out_size;
}
