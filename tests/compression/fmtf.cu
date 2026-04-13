#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "compression.h"
#include "utils.h"

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
