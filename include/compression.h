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

#endif // !COMPRESSION
