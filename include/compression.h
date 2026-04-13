#ifndef COMPRESSION

#include <stdint.h>

int rle1_compress(const uint8_t *in, int in_len, uint8_t *&out);
int make_symbols_table(const uint8_t *d_in, int d_in_len,
                       uint8_t *&symbols_table);

#endif // !COMPRESSION
