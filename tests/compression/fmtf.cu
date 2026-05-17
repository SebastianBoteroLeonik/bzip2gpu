#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "compression.h"
#include "utils.h"
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>

struct ShiftedIdentity {
  int N;
  ShiftedIdentity(int N) : N(N) {}
  __host__ __device__ int operator()(int idx) const {
    return (idx + 1) % N;
  }
};

TEST(compress, mtf_symbol_table_generation) {
  uint8_t reference_symbol_table[256];
  srand(2137);
  int reference_count = 0;
  for (int i = 0; i < 256; i++) {
    if (rand() % 3 == 0) {
      reference_symbol_table[reference_count++] = i;
    }
  }
  uint8_t data[100000];
  for (int i = 0; i < sizeof(data); i++) {
    data[i] = reference_symbol_table[rand() % reference_count];
  }
  uint8_t *device_data;
  CUDA_ERROR_CHECK(cudaMalloc(&device_data, sizeof(data)));
  CUDA_ERROR_CHECK(
      cudaMemcpy(device_data, data, sizeof(data), cudaMemcpyHostToDevice));
  uint8_t *symbol_table;
  int count = make_symbols_table(device_data, sizeof(data), symbol_table);
  CUDA_ERROR_CHECK(cudaFree(device_data));
  ASSERT_EQ(count, reference_count);
  for (int i = 0; i < count; i++) {
    ASSERT_EQ(symbol_table[i], reference_symbol_table[i]);
  }
  delete[] symbol_table;
}

void simple_fmtf(const uint8_t *in, int in_len, uint8_t *&out) {
  out = new uint8_t[in_len];
  uint8_t mtfList[256];
  bool seen[256] = {false};
  for (int i = 0; i < in_len; i++) {
    seen[in[i]] = true;
  }
  int pos_init = 0;
  for (int i = 0; i < 256; i++) {
    if (seen[i]) {
      mtfList[pos_init++] = i;
    }
  }
  for (int i = 0; i < in_len; i++) {
    uint8_t c = in[i];
    int pos = 0;
    for (; pos < 256; ++pos) {
      if (mtfList[pos] == c)
        break;
    }
    out[i] = pos;
    for (int k = pos; k > 0; --k) {
      mtfList[k] = mtfList[k - 1];
    }
    mtfList[0] = c;
  }
}

#define RUN_FMTF(ALPHABET_SIZE)                                                \
  srand(2137);                                                                 \
  constexpr int in_len = 1000000;                                              \
  uint8_t input[in_len];                                                       \
  for (int i = 0; i < in_len; i++) {                                           \
    input[i] = rand() % ALPHABET_SIZE;                                         \
  }                                                                            \
  uint8_t *reference_output = nullptr;                                         \
  simple_fmtf(input, in_len, reference_output);                                \
  uint8_t *input_cuda;                                                         \
  CUDA_ERROR_CHECK(cudaMalloc(&input_cuda, sizeof(input)));                    \
  CUDA_ERROR_CHECK(                                                            \
      cudaMemcpy(input_cuda, input, sizeof(input), cudaMemcpyHostToDevice));   \
  int *sa_cuda;                                                                \
  CUDA_ERROR_CHECK(cudaMalloc(&sa_cuda, in_len * sizeof(int)));                \
  thrust::transform(thrust::device, thrust::make_counting_iterator(0),         \
                    thrust::make_counting_iterator(in_len),                    \
                    thrust::device_pointer_cast(sa_cuda),                      \
                    ShiftedIdentity(in_len));                                  \
  uint8_t *cuda_output = nullptr;                                              \
  int orig_ptr = 0;                                                            \
  fmtf(input_cuda, sa_cuda, in_len, cuda_output, orig_ptr);                    \
  CUDA_ERROR_CHECK(cudaFree(input_cuda));                                      \
  CUDA_ERROR_CHECK(cudaFree(sa_cuda));                                         \
  ASSERT_NE(reference_output, nullptr);                                        \
  ASSERT_NE(cuda_output, nullptr);                                             \
  uint8_t *cuda_output_host = new uint8_t[in_len];                             \
  CUDA_ERROR_CHECK(cudaMemcpy(cuda_output_host, cuda_output, in_len,           \
                              cudaMemcpyDeviceToHost));                        \
  ASSERT_NE(cuda_output_host, nullptr);                                        \
  CUDA_ERROR_CHECK(cudaFree(cuda_output));

TEST(compress, fmtf_normal) {
  RUN_FMTF(256)
  for (int i = 0; i < in_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}

TEST(compress, fmtf_small_alphabet) {
  RUN_FMTF(4)
  for (int i = 0; i < in_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}
