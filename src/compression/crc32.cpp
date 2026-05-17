#include "crc32.h"

static uint32_t crc32_table[256];
static bool crc32_initialized = false;

void crc32_init() {
    if (crc32_initialized) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i << 24;
        for (int j = 0; j < 8; j++) {
            if (c & 0x80000000) {
                c = (c << 1) ^ 0x04C11DB7;
            } else {
                c = c << 1;
            }
        }
        crc32_table[i] = c;
    }
    crc32_initialized = true;
}

uint32_t bzip2_crc32(const uint8_t *data, size_t length) {
    crc32_init();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc = (crc << 8) ^ crc32_table[(crc >> 24) ^ data[i]];
    }
    return ~crc;
}
