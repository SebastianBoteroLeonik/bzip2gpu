#include <cstdlib>
#include <gtest/gtest.h>

#include "compression.h"
#include "utils.h"

// Test if tree building doesn't crash
TEST(compress, huffman_tree_builder_stability) {
  uint8_t len[max_n_groups][max_alphabet_size];
  int32_t code[max_n_groups][max_alphabet_size];
  constexpr int data_len = 10000;
  uint16_t data[data_len];
  uint16_t *device_data;
  srand(2137);
  CUDA_ERROR_CHECK(cudaMalloc(&device_data, data_len));
  for (int alphabet_size = 1; alphabet_size < 256; alphabet_size++) {
    for (int i = 0; i < data_len; i++) {
      data[i] = rand() % alphabet_size;
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(device_data, data, data_len, cudaMemcpyHostToDevice));
    huffman_build_trees(device_data, data_len, alphabet_size, len, code);
  }
  CUDA_ERROR_CHECK(cudaFree(device_data));
}
