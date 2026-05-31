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
  while ((len = in.fetch(buf)) > 0) {
    // printf("len %d\n", len);
    std::string reference = "word" + std::to_string(i++) + "\n";
    for (int i = 0; i < len; i++) {
      // printf("%d, %c, %c\n", i, reference[i], buf[i]);
      ASSERT_EQ(reference[i], buf[i]);
    }
    CUDA_ERROR_CHECK(cudaFreeHost(buf));
  }
  ASSERT_EQ(i, 10);
}
