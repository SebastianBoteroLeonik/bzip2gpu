#include "utils.h"
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <condition_variable>
typedef struct BZFileInputStream {
  std::istream *in_stream;
  int n_buffers;
  int buffer_size;
  int head_idx = 0;
  int tail_idx = -1;
  int len = 0;
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
  BZFileInputStream(int _n_buffers, int _buffer_size, std::string filename) {
    in_stream = new std::ifstream(filename, std::ios::in | std::ios::binary);
    setup_buffers(_n_buffers, _buffer_size);
  }

  void reader_thread_task() {
    while (*in_stream) {
      std::unique_lock<std::mutex> lock(mtx);
      // std::cerr << "locked mtx in reader for idx fetch\n";
      int next_buf_idx = (tail_idx + 1) % n_buffers;
      while (len == n_buffers) {
        not_full_cond.wait(lock, [this] {
          return ((tail_idx + 1) % n_buffers) != head_idx || is_closed;
        });
      }
      // std::cerr << "head_idx: " << head_idx << " tail_idx: " << tail_idx
      // << "\n";
      // std::cerr << "unlocking\n";
      lock.unlock();
      CUDA_ERROR_CHECK(cudaMallocHost(&buffers[next_buf_idx], buffer_size));
      in_stream->read((char *)buffers[next_buf_idx], buffer_size);
      buffer_lengths[next_buf_idx] = in_stream->gcount();
      lock.lock();
      // std::cerr << "locked mtx in reader for return\n";
      tail_idx++;
      tail_idx %= n_buffers;
      len++;
      not_empty_cond.notify_one();
      if (!*in_stream) {
        is_closed = true;
      }
      // std::cerr << "head_idx: " << head_idx << " tail_idx: " << tail_idx
      // << "\n";
      // std::cerr << "unlocking\n";
      lock.unlock();
    }
  }

  int fetch(uint8_t *&result) {
    std::unique_lock<std::mutex> lock(mtx);
    while (len == 0) {
      if (is_closed) {
        return -1;
      }
      not_empty_cond.wait(lock, [this] { return (len > 0) || is_closed; });
    }
    result = buffers[head_idx];
    int res_len = buffer_lengths[head_idx];
    // std::cerr << "fetched " << res_len << " bytes\n";
    buffers[head_idx] = nullptr;
    buffer_lengths[head_idx] = -1;
    head_idx++;
    head_idx %= n_buffers;
    len--;
    not_full_cond.notify_all();
    return res_len;
  }

} BZFileInputStream;
