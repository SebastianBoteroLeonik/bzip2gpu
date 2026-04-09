#include <cstdlib>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "compression.h"
#include "utils.h"

// Reference function
int simple_rle1_compress(uint8_t *in, int in_len, uint8_t *&out) {
  out = new uint8_t[in_len * 2];
  // out = (uint8_t *)malloc(in_len * 2);
  int out_idx = 0;
  int run = 1;

  for (int i = 0; i < in_len; i++) {
    if (i + 1 < in_len && in[i] == in[i + 1] && run < 259) {
      run++;
    } else {
      if (run < 4) {
        for (int j = 0; j < run; j++) {
          out[out_idx++] = in[i];
        }
      } else {
        for (int j = 0; j < 4; j++) {
          out[out_idx++] = in[i];
        }
        out[out_idx++] = (uint8_t)(run - 4);
      }
      run = 1;
    }
  }
  return out_idx; // Return the actual compressed size
}

#define RUN_RLE1(MAX_RUN_LENGTH)                                               \
  srand(2137);                                                                 \
  constexpr int in_len = 1000000;                                              \
  uint8_t input[in_len];                                                       \
  uint8_t chr = 0;                                                             \
  int reps = 0;                                                                \
  for (int i = 0; i < in_len; i++) {                                           \
    if (!reps) {                                                               \
      reps = rand() % MAX_RUN_LENGTH + 1;                                      \
      chr += rand() % 100 + 1;                                                 \
    }                                                                          \
    input[i] = chr;                                                            \
    reps--;                                                                    \
  }                                                                            \
  uint8_t *reference_output = nullptr;                                         \
  uint8_t *cuda_output = nullptr;                                              \
  int out_len_ref = simple_rle1_compress(input, in_len, reference_output);     \
  uint8_t *input_cuda;                                                         \
  CUDA_ERROR_CHECK(cudaMalloc(&input_cuda, sizeof(input)));                    \
  CUDA_ERROR_CHECK(                                                            \
      cudaMemcpy(input_cuda, input, sizeof(input), cudaMemcpyHostToDevice));   \
  int out_len_cuda = rle1_compress(input_cuda, in_len, cuda_output);           \
  CUDA_ERROR_CHECK(cudaFree(input_cuda));                                      \
  ASSERT_EQ(out_len_ref, out_len_cuda);                                        \
  ASSERT_NE(reference_output, nullptr);                                        \
  ASSERT_NE(cuda_output, nullptr);                                             \
  uint8_t *cuda_output_host = new uint8_t[out_len_cuda];                       \
  CUDA_ERROR_CHECK(cudaMemcpy(cuda_output_host, cuda_output, out_len_cuda,     \
                              cudaMemcpyDeviceToHost));                        \
  ASSERT_NE(cuda_output_host, nullptr);                                        \
  CUDA_ERROR_CHECK(cudaFree(cuda_output));

TEST(compress, rle1_normal) {
  RUN_RLE1(20)
  for (int i = 0; i < out_len_cuda; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
}

TEST(compress, rle1_incompressible) {
  RUN_RLE1(3)
  for (int i = 0; i < out_len_cuda; i++) {
    ASSERT_EQ(input[i], cuda_output_host[i]);
  }
  for (int i = 0; i < out_len_cuda; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
}

TEST(compress, rle1_long_runs) {
  RUN_RLE1(10000)
  for (int i = 0; i < out_len_cuda; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
}
