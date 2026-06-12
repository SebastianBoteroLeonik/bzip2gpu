#include "utils.h"
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <thread>

#include <condition_variable>

#ifndef IO_H
#define IO_H

typedef struct BZFileInputStream {
  std::istream *in_stream;
  int n_buffers;
  int buffer_size;
  int head_idx = 0;
  int tail_idx = -1;
  int len = 0;
  int chunks_read = 0;
  bool is_closed = false;
  uint8_t **buffers;
  int *buffer_lengths;
  std::mutex mtx;
  std::condition_variable not_full_cond;
  std::condition_variable not_empty_cond;
  void setup_buffers(int _n_buffers, int _buffer_size) {
    n_buffers = _n_buffers;
    buffer_size = _buffer_size;
    buffers = new uint8_t *[n_buffers];
    buffer_lengths = new int[n_buffers];

    for (size_t i = 0; i < n_buffers; i++) {
      buffers[i] = nullptr;
      buffer_lengths[i] = -1;
    }
    std::thread reader_thread(&BZFileInputStream::reader_thread_task, this);
    reader_thread.detach();
  }
  BZFileInputStream(int _n_buffers, int _buffer_size) {
    in_stream = &std::cin;
    setup_buffers(_n_buffers, _buffer_size);
  }
  // BZFileInputStream(int _n_buffers, int _buffer_size, const uint8_t *array,
  //                   int array_len) {
  //   auto buf = new std::stringbuf(std::ios::in);
  //   buf->pubsetbuf(reinterpret_cast<char *>(const_cast<uint8_t *>(array)),
  //                  array_len);
  //   in_stream = new std::istream(buf);
  //   printf("%p\n", in_stream);
  //   setup_buffers(_n_buffers, _buffer_size);
  // }
  BZFileInputStream(int _n_buffers, int _buffer_size, std::string filename) {
    in_stream = new std::ifstream(filename, std::ios::in | std::ios::binary);
    setup_buffers(_n_buffers, _buffer_size);
  }

  void reader_thread_task();

  int fetch(uint8_t *&result, int &chunk_idx);

} BZFileInputStream;
#endif // !IO_H
