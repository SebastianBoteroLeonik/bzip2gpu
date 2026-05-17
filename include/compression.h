#ifndef COMPRESSION

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

/**
 * Run bzip2's RLE1 compression
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing compressed data (device)
 *
 * Returns size of the compressed output.
 */
int rle1_compress(const uint8_t *in, int in_len, uint8_t *&out,
                  cudaStream_t stream = 0);

/**
 * Run bzip2's RLE2 compression
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing compressed data (device)
 * out_len: length of the output buffer (device)
 *
 * Returns size of the compressed output.
 */
void rle2_compress(const uint8_t *in, int in_len, uint16_t *out,
                   uint32_t *out_len, cudaStream_t stream = 0);

/**
 * Find unique bytes in the input buffer and build a symbols table.
 *
 * d_in: input buffer
 * d_in_len: length of the buffer
 * symbols_table: output array of unique symbols. This is a host array to be
 * cleaned with delete[]
 *
 * Returns number of unique symbols.
 */
int make_symbols_table(const uint8_t *d_in, int d_in_len,
                       uint8_t *&symbols_table, cudaStream_t stream = 0);

/**
 * Run move-to-front transform
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing transformed data (device)
 */
void fmtf(const uint8_t *in_original, const int *in_suffix_array, int in_len,
          uint8_t *&out, int &orig_ptr, cudaStream_t stream = 0);

/**
 * Run move-to-front transform
 *
 * out: output buffer containing transformed data (device)
 * device_data_in: input buffer (device)
 * data_in_len: length of the input buffer
 * alphabet_size: the number of symbols present in the block
 **/
constexpr int max_n_groups = 6;
constexpr int max_alphabet_size = 258;
int huffman_build_trees(uint16_t *device_data_in, int data_in_len,
                        int alphabet_size,
                        uint8_t len[max_n_groups][max_alphabet_size],
                        int32_t code[max_n_groups][max_alphabet_size],
                        uint8_t *&selectors, cudaStream_t stream);

/**
 * Run Burrows-Wheeler transform
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing transformed data (device)
 */
void fbwt(const uint8_t *in, int in_len, int *&out, cudaStream_t stream = 0);

/**
 * Orchestrates bzip2 compression on the GPU.
 */
void bzip2_gpu_compress(const uint8_t *in, int in_len, int n,
                        std::vector<uint8_t> &out);

int huffman_encode(uint16_t *dev_data_in, int data_in_len, int alphabet_size,
                   uint32_t *&dev_encoded_data,
                   uint8_t len[max_n_groups][max_alphabet_size],
                   int32_t code[max_n_groups][max_alphabet_size],
                   uint8_t *dev_selectors, int32_t num_selectors);

#endif // !COMPRESSION
