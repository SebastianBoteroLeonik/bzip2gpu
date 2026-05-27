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

struct pd_comparator {
  const int *rank;
  int k, length;

  pd_comparator(const int *r, int k, int l) : rank(r), k(k), length(l) {}

  __device__ bool operator()(const int &a, const int &b) const {
    if (rank[a] != rank[b])
      return rank[a] < rank[b];

    int next_a = a + k;
    if (next_a >= length)
      next_a -= length;
    int next_b = b + k;
    if (next_b >= length)
      next_b -= length;

    return rank[next_a] < rank[next_b];
  }
};

struct pd_difference {
  const int *out;
  const int *rank;
  int k, length;

  pd_difference(const int *o, const int *r, int k, int l)
      : out(o), rank(r), k(k), length(l) {}

  __device__ int operator()(const int &i) const {
    if (i == 0)
      return 0;
    int a = out[i - 1], b = out[i];
    if (rank[a] != rank[b])
      return 1;

    int next_a = a + k;
    if (next_a >= length)
      next_a -= length;
    int next_b = b + k;
    if (next_b >= length)
      next_b -= length;

    return rank[next_a] != rank[next_b] ? 1 : 0;
  }
};

void fbwt(const uint8_t *d_in, int in_len, int *&d_out, cudaStream_t stream) {
  if (in_len == 0)
    return;

  CUDA_ERROR_CHECK(cudaMallocAsync(&d_out, in_len * sizeof(int), stream));
  thrust::device_ptr<int> d_out_ptr(d_out);

  int *d_rank, *d_new_rank, *d_diff;
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_rank, in_len * sizeof(int), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_new_rank, in_len * sizeof(int), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_diff, in_len * sizeof(int), stream));

  thrust::copy(thrust::cuda::par.on(stream), d_in, d_in + in_len,
               thrust::device_pointer_cast(d_rank));
  thrust::sequence(thrust::cuda::par.on(stream), d_out_ptr, d_out_ptr + in_len);

  for (int k = 1; k < in_len; k *= 2) {
    thrust::sort(thrust::cuda::par.on(stream), d_out_ptr, d_out_ptr + in_len,
                 pd_comparator(d_rank, k, in_len));

    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(in_len),
                      thrust::device_pointer_cast(d_diff),
                      pd_difference(d_out, d_rank, k, in_len));

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

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPU CRASHED: %s\n", cudaGetErrorString(err));
  }
}