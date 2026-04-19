#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "compression.h"
#include "utils.h"

void simple_rle2_compress(const uint8_t *in, int in_len, uint16_t *&out, uint32_t &out_len) {
  out = new uint16_t[in_len * 2];
  int out_pos = 0;
  int i = 0;
  while (i < in_len) {
    if (in[i] != 0) {
      out[out_pos++] = in[i];
      i++;
    } else {
      int run_len = 0;
      while (i < in_len && in[i] == 0) {
        run_len++;
        i++;
      }
      int k = run_len;
      while (k > 0) {
        k--;
        out[out_pos++] = (k & 1) ? 257 : 256;
        k >>= 1;
      }
    }
  }
  out_len = out_pos;
}

#define RUN_RLE2(MAX_RUN_LENGTH)                                               \
  srand(2137);                                                                 \
  constexpr int in_len = 1000000;                                              \
  uint8_t input[in_len];                                                       \
  int reps = 0;                                                                \
  for (int i = 0; i < in_len; i++) {                                           \
    if (!reps) {                                                               \
      reps = rand() % MAX_RUN_LENGTH + 1;                                      \
      input[i] = rand() % 256;                                                 \
    } else {                                                                   \
      input[i] = input[i - 1];                                                 \
    }                                                                          \
    reps--;                                                                    \
  }                                                                            \
  uint16_t *reference_output = nullptr;                                        \
  uint32_t reference_out_len = 0;                                              \
  simple_rle2_compress(input, in_len, reference_output, reference_out_len);    \
  uint8_t *input_cuda;                                                         \
  CUDA_ERROR_CHECK(cudaMalloc(&input_cuda, sizeof(input)));                    \
  CUDA_ERROR_CHECK(                                                            \
      cudaMemcpy(input_cuda, input, sizeof(input), cudaMemcpyHostToDevice));   \
  uint16_t *cuda_output;                                                       \
  CUDA_ERROR_CHECK(cudaMalloc(&cuda_output, sizeof(uint16_t) * in_len));       \
  uint32_t *d_out_len;                                                         \
  CUDA_ERROR_CHECK(cudaMalloc(&d_out_len, sizeof(uint32_t)));                  \
  rle2_compress(input_cuda, in_len, cuda_output, d_out_len);                   \
  uint32_t cuda_out_len = 0;                                                   \
  CUDA_ERROR_CHECK(cudaMemcpy(&cuda_out_len, d_out_len, sizeof(uint32_t), cudaMemcpyDeviceToHost)); \
  CUDA_ERROR_CHECK(cudaFree(input_cuda));                                      \
  CUDA_ERROR_CHECK(cudaFree(d_out_len));                                       \
  ASSERT_EQ(reference_out_len, cuda_out_len);                                  \
  ASSERT_NE(reference_output, nullptr);                                        \
  ASSERT_NE(cuda_output, nullptr);                                             \
  uint16_t *cuda_output_host = new uint16_t[cuda_out_len];                     \
  CUDA_ERROR_CHECK(cudaMemcpy(cuda_output_host, cuda_output, cuda_out_len * sizeof(uint16_t), \
                              cudaMemcpyDeviceToHost));                        \
  ASSERT_NE(cuda_output_host, nullptr);                                        \
  CUDA_ERROR_CHECK(cudaFree(cuda_output));

TEST(compress, rle2_normal) {
  RUN_RLE2(20)
  for (uint32_t i = 0; i < cuda_out_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}

TEST(compress, rle2_incompressible) {
  srand(2137);
  constexpr int in_len = 1000000;
  uint8_t input[in_len];
  for (int i = 0; i < in_len; i++) {
    input[i] = (rand() % 255) + 1;
  }
  uint16_t *reference_output = nullptr;
  uint32_t reference_out_len = 0;
  simple_rle2_compress(input, in_len, reference_output, reference_out_len);
  
  uint8_t *input_cuda;
  CUDA_ERROR_CHECK(cudaMalloc(&input_cuda, sizeof(input)));
  CUDA_ERROR_CHECK(
      cudaMemcpy(input_cuda, input, sizeof(input), cudaMemcpyHostToDevice));
  uint16_t *cuda_output;
  CUDA_ERROR_CHECK(cudaMalloc(&cuda_output, sizeof(uint16_t) * in_len));
  uint32_t *d_out_len;
  CUDA_ERROR_CHECK(cudaMalloc(&d_out_len, sizeof(uint32_t)));
  rle2_compress(input_cuda, in_len, cuda_output, d_out_len);
  uint32_t cuda_out_len = 0;
  CUDA_ERROR_CHECK(cudaMemcpy(&cuda_out_len, d_out_len, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(input_cuda));
  CUDA_ERROR_CHECK(cudaFree(d_out_len));
  ASSERT_EQ(reference_out_len, cuda_out_len);
  ASSERT_NE(reference_output, nullptr);
  ASSERT_NE(cuda_output, nullptr);
  
  uint16_t *cuda_output_host = new uint16_t[cuda_out_len];
  CUDA_ERROR_CHECK(cudaMemcpy(cuda_output_host, cuda_output, cuda_out_len * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
  ASSERT_NE(cuda_output_host, nullptr);
  CUDA_ERROR_CHECK(cudaFree(cuda_output));
  
  for (uint32_t i = 0; i < cuda_out_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}

TEST(compress, rle2_long_runs) {
  RUN_RLE2(10000)
  for (uint32_t i = 0; i < cuda_out_len; i++) {
    ASSERT_EQ(reference_output[i], cuda_output_host[i]);
  }
  delete[] reference_output;
  delete[] cuda_output_host;
}
