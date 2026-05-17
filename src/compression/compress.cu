#include "compression.h"
#include "crc32.h"
#include "utils.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

struct BlockData {
  uint32_t crc;
  int orig_ptr;
  std::vector<uint16_t> rle2_data;
};

class BitWriter {
  std::vector<uint8_t> &out;
  uint8_t buf = 0;
  int bits_in_buf = 0;

public:
  BitWriter(std::vector<uint8_t> &out) : out(out) {}

  void write(uint32_t val, int num_bits) {
    for (int i = num_bits - 1; i >= 0; i--) {
      uint8_t bit = (val >> i) & 1;
      buf = (buf << 1) | bit;
      bits_in_buf++;
      if (bits_in_buf == 8) {
        out.push_back(buf);
        buf = 0;
        bits_in_buf = 0;
      }
    }
  }

  void pad_to_byte_boundary() {
    if (bits_in_buf > 0) {
      buf <<= (8 - bits_in_buf);
      out.push_back(buf);
      buf = 0;
      bits_in_buf = 0;
    }
  }
};

void compress_block(const uint8_t *in_data, int in_len, BlockData &out_data) {
  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

  uint8_t *d_in;
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_in, in_len, stream));
  CUDA_ERROR_CHECK(
      cudaMemcpyAsync(d_in, in_data, in_len, cudaMemcpyHostToDevice, stream));

  uint8_t *d_rle1_out = nullptr;
  int rle1_len = rle1_compress(d_in, in_len, d_rle1_out, stream);

  int *d_bwt_out = nullptr;
  fbwt(d_rle1_out, rle1_len, d_bwt_out, stream);

  uint8_t *d_fmtf_out = nullptr;
  int orig_ptr = 0;
  fmtf(d_rle1_out, d_bwt_out, rle1_len, d_fmtf_out, orig_ptr, stream);

  uint16_t *d_rle2_out = nullptr;
  uint32_t *d_rle2_len = nullptr;

  // Max RLE2 size
  int rle2_max_out = rle1_len * 2 + 100;
  CUDA_ERROR_CHECK(
      cudaMallocAsync(&d_rle2_out, rle2_max_out * sizeof(uint16_t), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_rle2_len, sizeof(uint32_t), stream));

  rle2_compress(d_fmtf_out, rle1_len, d_rle2_out, d_rle2_len, stream);

  uint32_t h_rle2_len = 0;
  CUDA_ERROR_CHECK(cudaMemcpyAsync(&h_rle2_len, d_rle2_len, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

  out_data.rle2_data.resize(h_rle2_len);
  CUDA_ERROR_CHECK(cudaMemcpyAsync(out_data.rle2_data.data(), d_rle2_out,
                                   h_rle2_len * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

  out_data.orig_ptr = orig_ptr;

  CUDA_ERROR_CHECK(cudaFreeAsync(d_in, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_rle1_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_bwt_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_fmtf_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_rle2_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_rle2_len, stream));

  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
}

void bzip2_gpu_compress(const uint8_t *in, int in_len, int n,
                        std::vector<uint8_t> &out) {
  if (n < 1)
    n = 1;
  if (n > 9)
    n = 9;

  BitWriter bw(out);
  bw.write(0x42, 8);     // 'B'
  bw.write(0x5A, 8);     // 'Z'
  bw.write(0x68, 8);     // 'h'
  bw.write(0x30 + n, 8); // '1'-'9'

  if (in_len == 0) {
    bw.write(0x17, 8);
    bw.write(0x72, 8);
    bw.write(0x45, 8);
    bw.write(0x38, 8);
    bw.write(0x50, 8);
    bw.write(0x90, 8);
    bw.write(0, 32); // stream crc 0
    bw.pad_to_byte_boundary();
    return;
  }

  int block_size = n * 100000;
  int num_blocks = (in_len + block_size - 1) / block_size;

  std::vector<BlockData> blocks(num_blocks);
  std::vector<std::thread> threads;

  for (int i = 0; i < num_blocks; ++i) {
    int start = i * block_size;
    int len = std::min(block_size, in_len - start);
    blocks[i].crc = bzip2_crc32(in + start, len);
    threads.emplace_back(
        [=, &blocks]() { compress_block(in + start, len, blocks[i]); });
  }

  for (auto &t : threads) {
    t.join();
  }

  uint32_t stream_crc = 0;

  for (int i = 0; i < num_blocks; ++i) {
    stream_crc = (stream_crc << 1) | (stream_crc >> 31);
    stream_crc ^= blocks[i].crc;

    bw.write(0x31, 8);
    bw.write(0x41, 8);
    bw.write(0x59, 8);
    bw.write(0x26, 8);
    bw.write(0x53, 8);
    bw.write(0x59, 8);

    bw.write(blocks[i].crc, 32);
    bw.write(0, 1);
    bw.write(blocks[i].orig_ptr, 24);
    bw.write(0, 7); // 7 bits zeroes padding

    for (uint16_t v : blocks[i].rle2_data) {
      bw.write(v, 16);
    }
  }

  bw.write(0x17, 8);
  bw.write(0x72, 8);
  bw.write(0x45, 8);
  bw.write(0x38, 8);
  bw.write(0x50, 8);
  bw.write(0x90, 8);

  bw.write(stream_crc, 32);
  bw.pad_to_byte_boundary();
}
