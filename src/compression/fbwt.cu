#include "compression.h"
#include "utils.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct pd_key_generator {
  const int *out;
  const int *rank;
  int k, length;

  pd_key_generator(const int *o, const int *r, int k, int l)
      : out(o), rank(r), k(k), length(l) {}

  __device__ uint64_t operator()(const int &i) const {
    int a = out[i];
    uint32_t r1 = rank[a];
    
    uint32_t next_a = (uint32_t)a + k;
    if (next_a >= length)
      next_a -= length;
      
    uint32_t r2 = rank[next_a];
    
    return ((uint64_t)r1 << 32) | r2;
  }
};

struct pd_difference {
  const uint64_t *keys;

  pd_difference(const uint64_t *k) : keys(k) {}

  __device__ int operator()(const int &i) const {
    if (i == 0)
      return 0;
    return keys[i] != keys[i - 1] ? 1 : 0;
  }
};

void fbwt(const uint8_t *d_in, int in_len, int *&d_out, cudaStream_t stream) {
if (in_len == 0)
    return;

  CUDA_ERROR_CHECK(cudaMallocAsync(&d_out, in_len * sizeof(int), stream));
  thrust::device_ptr<int> d_out_ptr(d_out);

  int *d_rank, *d_new_rank, *d_diff;
  uint64_t *d_keys;
  
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_rank, in_len * sizeof(int), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_new_rank, in_len * sizeof(int), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_diff, in_len * sizeof(int), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_keys, in_len * sizeof(uint64_t), stream));

  thrust::copy(thrust::cuda::par.on(stream), d_in, d_in + in_len,
               thrust::device_pointer_cast(d_rank));
  thrust::sequence(thrust::cuda::par.on(stream), d_out_ptr, d_out_ptr + in_len);

  for (int k = 1; k < in_len; k *= 2) {
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(in_len),
                      thrust::device_pointer_cast(d_keys),
                      pd_key_generator(d_out, d_rank, k, in_len));

    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        thrust::device_pointer_cast(d_keys),
                        thrust::device_pointer_cast(d_keys) + in_len,
                        d_out_ptr);

    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(in_len),
                      thrust::device_pointer_cast(d_diff),
                      pd_difference(d_keys));

    thrust::inclusive_scan(thrust::cuda::par.on(stream),
                           thrust::device_pointer_cast(d_diff),
                           thrust::device_pointer_cast(d_diff) + in_len,
                           thrust::device_pointer_cast(d_diff));

    thrust::scatter(thrust::cuda::par.on(stream),
                    thrust::device_pointer_cast(d_diff),
                    thrust::device_pointer_cast(d_diff) + in_len, d_out_ptr,
                    thrust::device_pointer_cast(d_new_rank));

    std::swap(d_rank, d_new_rank);
  }

  CUDA_ERROR_CHECK(cudaFreeAsync(d_rank, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_new_rank, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_diff, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_keys, stream));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPU CRASHED: %s\n", cudaGetErrorString(err));
  }
}