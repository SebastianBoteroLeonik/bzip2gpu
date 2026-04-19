#ifndef COMPRESSION

#include <stdint.h>

/**
 * Run bzip2's RLE1 compression
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing compressed data (device)
 *
 * Returns size of the compressed output.
 */
int rle1_compress(const uint8_t *in, int in_len, uint8_t *&out);

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
                       uint8_t *&symbols_table);

/**
 * Run move-to-front transform
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing transformed data (device)
 */
void fmtf(const uint8_t *in, int in_len, uint8_t *&out);

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
void huffman_build_trees(uint16_t *device_data_in, int data_in_len,
                         int alphabet_size,
                         uint8_t len[max_n_groups][max_alphabet_size],
                         int32_t code[max_n_groups][max_alphabet_size]);

/**
 * Run Burrows-Wheeler transform
 *
 * in: input buffer (device)
 * in_len: length of the input buffer
 * out: output buffer containing transformed data (device)
 */
void fbwt(const uint8_t *in, int in_len, int *&out);

#endif // !COMPRESSION
