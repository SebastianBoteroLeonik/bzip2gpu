#include "io.h"
void BZFileInputStream::reader_thread_task() {

  while (*in_stream) {
    std::unique_lock<std::mutex> lock(mtx);
    // std::cerr << "locked mtx in reader for idx fetch\n";
    int next_buf_idx = (tail_idx + 1) % n_buffers;
    while (len == n_buffers) {
      not_full_cond.wait(lock, [this] {
        return ((tail_idx + 1) % n_buffers) != head_idx || is_closed;
      });
    }
    // std::cerr << "head_idx: " << head_idx << " tail_idx: " << tail_idx <<
    // "\n"; std::cerr << "unlocking\n";
    lock.unlock();
    CUDA_ERROR_CHECK(cudaMallocHost(&buffers[next_buf_idx], buffer_size));
    in_stream->read((char *)buffers[next_buf_idx], buffer_size);
    buffer_lengths[next_buf_idx] = in_stream->gcount();
    lock.lock();
    // std::cerr << "locked mtx in reader for return\n";
    tail_idx++;
    tail_idx %= n_buffers;
    len++;
    not_empty_cond.notify_all();
    if (in_stream->eof()) {
      is_closed = true;
    }
    // std::cerr << "head_idx: " << head_idx << " tail_idx: " << tail_idx <<
    // "\n"; std::cerr << "unlocking\n";
    lock.unlock();
  }
}

int BZFileInputStream::fetch(uint8_t *&result, int &chunk_idx) {
  std::unique_lock<std::mutex> lock(mtx);
  while (len == 0) {
    if (is_closed) {
      return -1;
    }
    // std::cerr << "Buffer empty, waiting\n";
    not_empty_cond.wait(lock, [this] { return (len > 0) || is_closed; });
    // std::cerr << "Cond notified\n";
  }
  result = buffers[head_idx];
  int res_len = buffer_lengths[head_idx];
  // std::cerr << "fetched " << res_len << " bytes\n";
  buffers[head_idx] = nullptr;
  buffer_lengths[head_idx] = -1;
  head_idx++;
  head_idx %= n_buffers;
  len--;
  chunk_idx = chunks_read++;
  not_full_cond.notify_all();
  return res_len;
}
