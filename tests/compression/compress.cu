#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include "../../src/compression/crc32.h"
#include "compression.h"

#include <cstdint>
#include <fstream>
#include <iostream>

bool writeBinaryFile(const std::string &filename,
                     const std::vector<uint8_t> &data) {
  std::ofstream outFile(filename, std::ios::out | std::ios::binary);

  if (!outFile) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return false;
  }

  outFile.write(reinterpret_cast<const char *>(data.data()), data.size());

  return outFile.good();
}

TEST(compress, empty_input) {
  std::vector<uint8_t> out;
  bzip2_gpu_compress(nullptr, 0, 9, out);

  ASSERT_EQ(out.size(), 14);

  EXPECT_EQ(out[0], 0x42);
  EXPECT_EQ(out[1], 0x5a);
  EXPECT_EQ(out[2], 0x68);
  EXPECT_EQ(out[3], 0x39);

  EXPECT_EQ(out[4], 0x17);
  EXPECT_EQ(out[5], 0x72);
  EXPECT_EQ(out[6], 0x45);
  EXPECT_EQ(out[7], 0x38);
  EXPECT_EQ(out[8], 0x50);
  EXPECT_EQ(out[9], 0x90);

  EXPECT_EQ(out[10], 0);
  EXPECT_EQ(out[11], 0);
  EXPECT_EQ(out[12], 0);
  EXPECT_EQ(out[13], 0);
}

TEST(compress, single_block_basic) {
  const char *input_str = "hello world!\n";
  int in_len = std::strlen(input_str);

  std::vector<uint8_t> out;
  bzip2_gpu_compress((const uint8_t *)input_str, in_len, 1, out);

  writeBinaryFile("test.bz2", out);

  ASSERT_GT(out.size(), 28);

  EXPECT_EQ(out[0], 0x42);
  EXPECT_EQ(out[1], 0x5a);
  EXPECT_EQ(out[2], 0x68);
  EXPECT_EQ(out[3], 0x31);

  EXPECT_EQ(out[4], 0x31);
  EXPECT_EQ(out[5], 0x41);
  EXPECT_EQ(out[6], 0x59);
  EXPECT_EQ(out[7], 0x26);
  EXPECT_EQ(out[8], 0x53);
  EXPECT_EQ(out[9], 0x59);

  uint32_t expected_crc = bzip2_crc32((const uint8_t *)input_str, in_len);
  uint32_t actual_crc =
      (out[10] << 24) | (out[11] << 16) | (out[12] << 8) | out[13];
  EXPECT_EQ(actual_crc, expected_crc);
}

TEST(compress, multiple_blocks) {
  int in_len = 250000;
  std::vector<uint8_t> input(in_len);
  srand(42);
  for (int i = 0; i < in_len; i++) {
    input[i] = rand() % 256;
  }

  std::vector<uint8_t> out;
  bzip2_gpu_compress(input.data(), in_len, 1, out);

  writeBinaryFile("test_multiple.bz2", out);

  ASSERT_GT(out.size(), 56);

  int magic_count = 0;
  uint64_t current = 0;
  int current_bits = 0;
  for (uint8_t b : out) {
    for (int i = 7; i >= 0; i--) {
      current = (current << 1) | ((b >> i) & 1);
      current &= 0xFFFFFFFFFFFFULL;
      if (current == 0x314159265359ULL) {
        magic_count++;
      }
    }
  }

  EXPECT_EQ(magic_count, 3);
}
