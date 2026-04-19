#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>

#include "compression.h"
#include "utils.h"

void simple_fbwt(const uint8_t *in, int in_len, int *&out) {
  out = new int[in_len];
  std::iota(out, out + in_len, 0);

  std::sort(out, out + in_len, [in, in_len](int a, int b) {
    for (int i = 0; i < in_len; ++i) {
      uint8_t c_a = in[(a + i) % in_len];
      uint8_t c_b = in[(b + i) % in_len];
      if (c_a != c_b) return c_a < c_b;
    }
    return a < b;
  });
}

#define RUN_FBWT(ALPHABET_SIZE)                                                \
  srand(2137);                                                                 \
  constexpr int in_len = 5000;                                                 \
  uint8_t input[in_len];                                                       \
  for (int i = 0; i < in_len; i++) {                                           \
    input[i] = rand() % ALPHABET_SIZE;                                         \
  }                                                                            \
  int *reference_output = nullptr;                                             \
  simple_fbwt(input, in_len, reference_output);                                \
  uint8_t *input_cuda;                                                         \
  CUDA_ERROR_CHECK(cudaMalloc(&input_cuda, sizeof(input)));                    \
  CUDA_ERROR_CHECK(                                                            \
      cudaMemcpy(input_cuda, input, sizeof(input), cudaMemcpyHostToDevice));   \
  int *cuda_output = nullptr;                                                  \
  fbwt(input_cuda, in_len, cuda_output);                                       \
  CUDA_ERROR_CHECK(cudaFree(input_cuda));                                      \
  ASSERT_NE(reference_output, nullptr);                                        \
  ASSERT_NE(cuda_output, nullptr);                                             \
  int *cuda_output_host = new int[in_len];                                     \
  CUDA_ERROR_CHECK(cudaMemcpy(cuda_output_host, cuda_output,                   \
                   in_len * sizeof(int), cudaMemcpyDeviceToHost));             \
  CUDA_ERROR_CHECK(cudaFree(cuda_output));

TEST(compress, fbwt_normal) {
  RUN_FBWT(256)
  for (int i = 0; i < in_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}

TEST(compress, fbwt_small_alphabet) {
  RUN_FBWT(4)
  for (int i = 0; i < in_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}

TEST(compress, fbwt_single_character) {
  RUN_FBWT(1)
  for (int i = 0; i < in_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}
