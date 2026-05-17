#ifndef CRC32_H
#define CRC32_H

#include <stdint.h>
#include <stddef.h>

void crc32_init();
uint32_t bzip2_crc32(const uint8_t *data, size_t length);

#endif
