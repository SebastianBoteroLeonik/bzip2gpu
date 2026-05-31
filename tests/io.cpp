#include "io.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <string>

TEST(io, file_input_stream) {
  int n_buffers = 10;
  int buffer_size = 6;
  std::string filename = "test_files/input_test_file";
  BZFileInputStream in(n_buffers, buffer_size, filename);
  uint8_t *buf;
  int len;
  int i = 0;
  int chunk_idx = 0;
  while ((len = in.fetch(buf, chunk_idx)) > 0) {
    // printf("len %d\n", len);
    ASSERT_EQ(i, chunk_idx);
    std::string reference = "word" + std::to_string(i++) + "\n";
    for (int j = 0; j < len; j++) {
      // printf("%d, %c, %c\n", i, reference[i], buf[i]);
      ASSERT_EQ(reference[j], buf[j]);
    }
    CUDA_ERROR_CHECK(cudaFreeHost(buf));
  }
  ASSERT_EQ(i, 10);
}
