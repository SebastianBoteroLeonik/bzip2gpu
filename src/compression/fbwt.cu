#include "compression.h"
#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

struct suffix_comparator {
  const uint8_t *str;
  const int length;

  suffix_comparator(const uint8_t *s, int length) : str(s), length(length) {}

  __device__
  bool operator()(const int &a, const int &b) const {
    int i = 0;

    while (i < length && str[(a + i) % length] == str[(b + i) % length]) {
      i++;
    }

    return str[(a + i) % length] < str[(b + i) % length];
  }
};


void fbwt(const uint8_t *d_in, int in_len, int *&d_out) {
  if (in_len == 0)
    return;

  CUDA_ERROR_CHECK(cudaMalloc(&d_out, in_len * sizeof(int)));

  thrust::device_ptr<int> d_out_ptr(d_out);
  thrust::sequence(d_out_ptr, d_out_ptr + in_len);

  thrust::sort(d_out_ptr, d_out_ptr + in_len, suffix_comparator(d_in, in_len));

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("GPU CRASHED: %s\n", cudaGetErrorString(err));
  }
}
