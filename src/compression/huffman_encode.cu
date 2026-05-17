#include "compression.h"
#include "utils.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

__constant__ uint8_t c_len[max_n_groups][max_alphabet_size];
__constant__ int32_t c_code[max_n_groups][max_alphabet_size];

// Kernel 1: Calculate the bit-length of each symbol
__global__ void calc_bit_lengths_kernel(const uint16_t *d_data_in,
                                        const uint8_t *d_selectors,
                                        int32_t *d_bit_lengths,
                                        int data_in_len) {
  constexpr int BZ_G_SIZE = 50;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_in_len)
    return;

  uint8_t symbol = d_data_in[idx];
  uint8_t table_idx = d_selectors[idx / BZ_G_SIZE];
  d_bit_lengths[idx] = c_len[table_idx][symbol];
}

// Kernel 2: Pack the bits into a global array using atomic operations
__global__ void pack_bits_kernel(
    const uint16_t *d_data_in, const uint8_t *d_selectors,
    const int32_t *d_bit_offsets, const int32_t *d_bit_lengths,
    uint32_t *d_encoded_words, // Array of 32-bit words initialized to 0
    int data_in_len) {
  constexpr int BZ_G_SIZE = 50;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_in_len)
    return;

  uint8_t symbol = d_data_in[idx];
  uint8_t table_idx = d_selectors[idx / BZ_G_SIZE];

  int32_t code = c_code[table_idx][symbol];
  int32_t length = d_bit_lengths[idx];
  int32_t bit_offset =
      d_bit_offsets[idx]; // Global bit position for this thread

  // int32_t code_flipped = 0;
  // uint8_t *code_bytes = (uint8_t *)&code_flipped;
  // for (int i = 0; i < sizeof(code); i++) {
  //   code_bytes[i] = (code >> (8 * i)) & 0xff;
  // }
  // code = code_flipped;

  // Calculate where to write in the 32-bit word array
  int word_idx = bit_offset / 32;
  int bit_shift = bit_offset % 32;

  // IMPORTANT: bzip2 usually packs bits Most Significant Bit (MSB) first.
  // This logic assumes a standard Left-Shift packing. You may need to reverse
  // the shift direction depending on how your bit-reader expects the data!

  if (bit_shift + length <= 32) {
    // Case 1: The code fits entirely within a single 32-bit word
    uint32_t aligned_code = code << (32 - bit_shift - length);
    atomicOr(&d_encoded_words[word_idx], aligned_code);
  } else {
    // Case 2: The code spans across a 32-bit word boundary
    int bits_in_first = 32 - bit_shift;
    int bits_in_second = length - bits_in_first;

    uint32_t first_part = code >> bits_in_second;
    uint32_t second_part = code << (32 - bits_in_second);

    atomicOr(&d_encoded_words[word_idx], first_part);
    atomicOr(&d_encoded_words[word_idx + 1], second_part);
  }
}

// int huffman_encode(uint8_t *dev_data_in, int data_in_len, int alphabet_size,
//                    uint32_t *&encoded_data,
//                    uint8_t len[max_n_groups][max_alphabet_size],
//                    int32_t code[max_n_groups][max_alphabet_size],
//                    uint8_t *dev_selectors, int32_t num_selectors) {
//   CUDA_ERROR_CHECK(
//       cudaMemcpyToSymbol(c_len, len, max_n_groups * max_alphabet_size));
//   CUDA_ERROR_CHECK(cudaMemcpyToSymbol(
//       c_code, code, sizeof(**code) * max_n_groups * max_alphabet_size));
//
//   int32_t *dev_bit_lengths, *dev_bit_offsets;
//   CUDA_ERROR_CHECK(cudaMalloc(&dev_bit_lengths, data_in_len *
//   sizeof(int32_t))); CUDA_ERROR_CHECK(cudaMalloc(&dev_bit_offsets,
//   data_in_len * sizeof(int32_t)));
//
//   constexpr int block_size = 256;
//   int blocks_count = (data_in_len + block_size - 1) / block_size;
//
//   calc_bit_lengths_kernel<<<blocks_count, block_size>>>(
//       dev_data_in, dev_selectors, dev_bit_lengths, data_in_len);
//   CUDA_ERROR_CHECK(cudaDeviceSynchronize());
//
//   thrust::device_ptr<int32_t> t_lengths(dev_bit_lengths);
//   thrust::device_ptr<int32_t> t_offsets(dev_bit_offsets);
//   thrust::exclusive_scan(thrust::device, t_lengths, t_lengths + data_in_len,
//                          t_offsets);
//
//   int32_t last_length, last_offset;
//   CUDA_ERROR_CHECK(cudaMemcpy(&last_length, &dev_bit_lengths[data_in_len -
//   1],
//                               sizeof(int32_t), cudaMemcpyDeviceToHost));
//   CUDA_ERROR_CHECK(cudaMemcpy(&last_offset, &dev_bit_offsets[data_in_len -
//   1],
//                               sizeof(int32_t), cudaMemcpyDeviceToHost));
//
//   int total_bits = last_offset + last_length;
//   int total_words = (total_bits + 31) / 32;
//
//   // 5. Allocate and Zero-out the output array
//   CUDA_ERROR_CHECK(cudaMalloc(&encoded_data, total_words *
//   sizeof(uint32_t))); CUDA_ERROR_CHECK(cudaMemset(encoded_data, 0,
//   total_words * sizeof(uint32_t)));
//
//   // 6. Run Kernel 2: Scatter the bits
//   pack_bits_kernel<<<blocks_count, block_size>>>(
//       dev_data_in, dev_selectors, dev_bit_offsets, dev_bit_lengths,
//       encoded_data, data_in_len);
//   CUDA_ERROR_CHECK(cudaDeviceSynchronize());
//
//   // Cleanup temp arrays
//   cudaFree(dev_bit_lengths);
//   cudaFree(dev_bit_offsets);
//
//   // Return the size of the compressed output in BYTES
//   return total_words * 4;
// }

#include <thrust/device_vector.h>

int huffman_encode(uint16_t *dev_data_in, int data_in_len, int alphabet_size,
                   uint32_t *&dev_encoded_data,
                   uint8_t len[max_n_groups][max_alphabet_size],
                   int32_t code[max_n_groups][max_alphabet_size],
                   uint8_t *dev_selectors, int32_t num_selectors,
                   cudaStream_t stream) {
  CUDA_ERROR_CHECK(cudaMemcpyToSymbolAsync(c_len, len,
                                           max_n_groups * max_alphabet_size, 0,
                                           cudaMemcpyHostToDevice, stream));
  CUDA_ERROR_CHECK(cudaMemcpyToSymbolAsync(
      c_code, code, sizeof(**code) * max_n_groups * max_alphabet_size, 0,
      cudaMemcpyHostToDevice, stream));

  thrust::device_vector<int32_t> dev_bit_lengths(data_in_len);
  thrust::device_vector<int32_t> dev_bit_offsets(data_in_len);
  {
    constexpr int block_size = 256;
    const int blocks_count = (data_in_len + block_size - 1) / block_size;

    calc_bit_lengths_kernel<<<blocks_count, block_size, 0, stream>>>(
        dev_data_in, dev_selectors,
        thrust::raw_pointer_cast(dev_bit_lengths.data()), data_in_len);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  }

  thrust::exclusive_scan(dev_bit_lengths.begin(), dev_bit_lengths.end(),
                         dev_bit_offsets.begin());

  const int32_t last_length = dev_bit_lengths.back();
  const int32_t last_offset = dev_bit_offsets.back();

  const int total_bits = last_offset + last_length;
  const int total_words = (total_bits + 31) / 32;

  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_encoded_data, total_words * sizeof(uint32_t)));
  CUDA_ERROR_CHECK(cudaMemsetAsync(dev_encoded_data, 0,
                                   total_words * sizeof(uint32_t), stream));
  {
    constexpr int block_size = 50;
    const int blocks_count = (data_in_len + block_size - 1) / block_size;

    pack_bits_kernel<<<blocks_count, block_size, 0, stream>>>(
        dev_data_in, dev_selectors,
        thrust::raw_pointer_cast(dev_bit_offsets.data()),
        thrust::raw_pointer_cast(dev_bit_lengths.data()), dev_encoded_data,
        data_in_len);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  }

  return total_bits;
}
