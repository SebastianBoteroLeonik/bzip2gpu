#include "compression.h"
#include "crc32.h"
#include "io.h"
#include "stopwatch.h"
#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

struct BlockData {
  uint32_t crc;
  int orig_ptr;
  int alphabet_size;
  bool present_symbols[256];
  int num_selectors;
  std::vector<uint8_t> selectors_mtf;
  int n_groups;
  uint8_t huff_len[max_n_groups][max_alphabet_size];
  std::vector<uint32_t> huff_data;
  int huff_bits;
};

class BitWriter {
  std::vector<uint8_t> &out;
  uint64_t bit_buffer = 0;
  int bits_in_buffer = 0;

public:
  BitWriter(std::vector<uint8_t> &out) : out(out) {}

  void write(uint32_t val, int num_bits) {
    if (num_bits == 0)
      return;

    val &= (1ULL << num_bits) - 1;

    bit_buffer = (bit_buffer << num_bits) | val;
    bits_in_buffer += num_bits;

    while (bits_in_buffer >= 8) {
      bits_in_buffer -= 8;
      out.push_back(static_cast<uint8_t>(bit_buffer >> bits_in_buffer));
    }
  }

  void pad_to_byte_boundary() {
    if (bits_in_buffer > 0) {
      int padding_bits = 8 - bits_in_buffer;
      bit_buffer <<= padding_bits;
      out.push_back(static_cast<uint8_t>(bit_buffer));
      bit_buffer = 0;
      bits_in_buffer = 0;
    }
  }
};

void compress_block(const uint8_t *in_data, int in_len, BlockData &out_data) {
  Stopwatch stopwatch{};
  stopwatch.start("Create stream");
  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
  stopwatch.end();

  stopwatch.start("Input data transfer");
  uint8_t *d_in;
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_in, in_len, stream));
  CUDA_ERROR_CHECK(
      cudaMemcpyAsync(d_in, in_data, in_len, cudaMemcpyHostToDevice, stream));
  stopwatch.end();

  uint8_t *d_rle1_out = nullptr;
  stopwatch.start("RLE1");
  int rle1_len = rle1_compress(d_in, in_len, d_rle1_out, stream);
  stopwatch.end();

  int *d_bwt_out = nullptr;
  stopwatch.start("BWT");
  fbwt(d_rle1_out, rle1_len, d_bwt_out, stream);
  stopwatch.end();

  uint8_t *d_fmtf_out = nullptr;
  int orig_ptr = 0;
  stopwatch.start("MTF");
  int alphabet_size = fmtf(d_rle1_out, d_bwt_out, rle1_len, d_fmtf_out,
                           orig_ptr, out_data.present_symbols, stream);
  stopwatch.end();

  uint16_t *d_rle2_out = nullptr;
  uint32_t *d_rle2_len = nullptr;

  int rle2_max_out = rle1_len * 2 + 100;
  CUDA_ERROR_CHECK(
      cudaMallocAsync(&d_rle2_out, rle2_max_out * sizeof(uint16_t), stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_rle2_len, sizeof(uint32_t), stream));

  stopwatch.start("RLE2");
  rle2_compress(d_fmtf_out, rle1_len, d_rle2_out, d_rle2_len, alphabet_size,
                stream);
  stopwatch.end();

  uint32_t h_rle2_len = 0;
  CUDA_ERROR_CHECK(cudaMemcpyAsync(&h_rle2_len, d_rle2_len, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

  uint8_t len[max_n_groups][max_alphabet_size];
  int32_t code[max_n_groups][max_alphabet_size];
  uint8_t *selectors;
  int n_groups = 0;
  stopwatch.start("Huffman build trees");
  int num_selectors =
      huffman_build_trees(d_rle2_out, h_rle2_len, alphabet_size, len, code,
                          selectors, n_groups, stream);
  stopwatch.end();

  out_data.selectors_mtf.resize(num_selectors);
  {
    uint8_t pos[max_n_groups];
    for (int i = 0; i < n_groups; i++)
      pos[i] = i;
    for (int i = 0; i < num_selectors; i++) {
      uint8_t ll_i = selectors[i];
      int j = 0;
      uint8_t tmp = pos[j];
      while (ll_i != tmp) {
        j++;
        uint8_t tmp2 = tmp;
        tmp = pos[j];
        pos[j] = tmp2;
      }
      pos[0] = tmp;
      out_data.selectors_mtf[i] = j;
    }
  }

  uint32_t *dev_encoded;
  uint8_t *dev_selectors;
  CUDA_ERROR_CHECK(cudaMallocAsync(&dev_selectors, num_selectors, stream));
  CUDA_ERROR_CHECK(cudaMemcpyAsync(dev_selectors, selectors, num_selectors,
                                   cudaMemcpyHostToDevice, stream));
  stopwatch.start("Huffman encode");
  int encoded_bits =
      huffman_encode(d_rle2_out, h_rle2_len, alphabet_size + 2, dev_encoded,
                     len, code, dev_selectors, num_selectors, n_groups, stream);
  stopwatch.end();

  const int total_words = (encoded_bits + 31) / 32;
  out_data.huff_data.resize(total_words);
  stopwatch.start("Output data transfer");
  CUDA_ERROR_CHECK(cudaMemcpyAsync(out_data.huff_data.data(), dev_encoded,
                                   total_words * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  stopwatch.end();

  out_data.orig_ptr = orig_ptr;
  out_data.alphabet_size = alphabet_size + 2;
  out_data.num_selectors = num_selectors;
  out_data.n_groups = n_groups;
  out_data.huff_bits = encoded_bits;
  for (int g = 0; g < n_groups; g++) {
    for (int s = 0; s < out_data.alphabet_size; s++) {
      out_data.huff_len[g][s] = len[g][s];
    }
  }

  delete[] selectors;
  CUDA_ERROR_CHECK(cudaFreeAsync(dev_selectors, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(dev_encoded, stream));

  CUDA_ERROR_CHECK(cudaFreeAsync(d_in, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_rle1_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_bwt_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_fmtf_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_rle2_out, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_rle2_len, stream));

  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
}

void bzip2_gpu_compress(BZFileInputStream &in_stream, int n,
                        std::vector<uint8_t> &out) {
  if (n < 1)
    n = 1;
  if (n > 9)
    n = 9;
  // fprintf(stderr, "HI0\n");

  // if (cudaHostRegister((void *)in, in_len, cudaHostRegisterDefault) !=
  //     cudaSuccess) {
  //   cudaGetLastError();
  // }

  // out.reserve(out.size() + (in_len / 4) + 1024);

  // fprintf(stderr, "HI1\n");
  BitWriter bw(out);
  bw.write(0x42, 8);     // 'B'
  bw.write(0x5A, 8);     // 'Z'
  bw.write(0x68, 8);     // 'h'
  bw.write(0x30 + n, 8); // '1'-'9'

  // if (in_len == 0) {
  //   bw.write(0x17, 8);
  //   bw.write(0x72, 8);
  //   bw.write(0x45, 8);
  //   bw.write(0x38, 8);
  //   bw.write(0x50, 8);
  //   bw.write(0x90, 8);
  //   bw.write(0, 32); // stream crc 0
  //   bw.pad_to_byte_boundary();
  //   // cudaHostUnregister((void *)in);
  //   return;
  // }

  int block_size = n * 100000;
  // int num_blocks = (in_len + block_size - 1) / block_size;

  // std::vector<BlockData> blocks(num_blocks);
  std::vector<BlockData> blocks;

  int max_threads = std::thread::hardware_concurrency();
  if (max_threads <= 0)
    max_threads = 4;
  int num_workers = std::min(max_threads, 16);
  // int num_workers = 1;

  std::mutex queue_mtx;
  // int current_block = 0;
  std::vector<std::thread> workers;

  // fprintf(stderr, "HI2\n");
  for (int w = 0; w < num_workers; ++w) {
    workers.emplace_back([&]() {
      Stopwatch stopwatch{};
      while (true) {
        // int i = -1;
        uint8_t *data;
        int chunk_idx;
        // fprintf(stderr, "HI3\n");
        stopwatch.start("fetch");
        int len = in_stream.fetch(data, chunk_idx);
        if (len < 0) {
          return;
        }
        stopwatch.end();
        // fprintf(stderr, "HI4\n");

        // int start = i * block_size;
        // int len = std::min(block_size, in_len - start);

        // FIX 2: Compute sequential CRC32 independently in the worker thread
        BlockData block_data;

        stopwatch.start("crc calculation");
        block_data.crc = bzip2_crc32(data, (size_t)len);
        stopwatch.end();

        stopwatch.start("compress block");
        compress_block(data, len, block_data);
        stopwatch.end();
        CUDA_ERROR_CHECK(cudaFreeHost(data));
        {
          std::lock_guard<std::mutex> lock(queue_mtx);
          // if (len <= 0)
          //   return;
          // i = current_block++;
          if (blocks.size() < chunk_idx + 1) {
            blocks.resize(chunk_idx + 1);
          }
          blocks[chunk_idx] = block_data;
        }
      }
    });
  }

  for (auto &t : workers) {
    t.join();
  }

  uint32_t stream_crc = 0;

  for (int i = 0; i < blocks.size(); ++i) {
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

    // 1) SymMap
    uint16_t mapL1 = 0;
    for (int r = 0; r < 16; r++) {
      bool any = false;
      for (int s = 0; s < 16; s++) {
        if (blocks[i].present_symbols[r * 16 + s]) {
          any = true;
          break;
        }
      }
      if (any)
        mapL1 |= (1 << (15 - r));
    }
    bw.write(mapL1, 16);
    for (int r = 0; r < 16; r++) {
      if ((mapL1 >> (15 - r)) & 1) {
        uint16_t mapL2 = 0;
        for (int s = 0; s < 16; s++) {
          if (blocks[i].present_symbols[r * 16 + s]) {
            mapL2 |= (1 << (15 - s));
          }
        }
        bw.write(mapL2, 16);
      }
    }

    // 2) NumTrees
    bw.write(blocks[i].n_groups, 3);

    // 3) NumSels
    bw.write(blocks[i].num_selectors, 15);

    // 4) Selectors
    for (uint8_t sel : blocks[i].selectors_mtf) {
      for (int k = 0; k < sel; k++)
        bw.write(1, 1);
      bw.write(0, 1);
    }

    // 5) Trees
    for (int t = 0; t < blocks[i].n_groups; t++) {
      uint8_t curr = blocks[i].huff_len[t][0];
      bw.write(curr, 5);
      for (int s = 0; s < blocks[i].alphabet_size; s++) {
        uint8_t target = blocks[i].huff_len[t][s];
        while (curr < target) {
          bw.write(2, 2); // 10
          curr++;
        }
        while (curr > target) {
          bw.write(3, 2); // 11
          curr--;
        }
        bw.write(0, 1);
      }
    }

    // 6) Huffman Encoded Data
    int bits_remaining = blocks[i].huff_bits;
    for (uint32_t word : blocks[i].huff_data) {
      int bits_to_write = std::min(32, bits_remaining);
      bw.write(word >> (32 - bits_to_write), bits_to_write);
      bits_remaining -= bits_to_write;
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

  // cudaHostUnregister((void *)in);
}
