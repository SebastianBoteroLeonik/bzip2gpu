#include <cstdlib>
#include <gtest/gtest.h>

#include "compression.h"
#include "utils.h"

__global__ void count_frequencies(uint16_t *data_in, int data_len, int *freqs);

TEST(compress, huffman_count_frequencies) {
  constexpr int len = 10000;
  uint16_t h_data[len];
  int h_freqs[258] = {0};
  srand(2137);
  for (int i = 0; i < len; i++) {
    uint16_t symbol = rand() % 300;
    symbol %= 258;
    h_data[i] = symbol;
    h_freqs[symbol]++;
  }
  uint16_t *d_data;
  CUDA_ERROR_CHECK(cudaMalloc(&d_data, sizeof(h_data)));
  CUDA_ERROR_CHECK(
      cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));
  int *d_freqs;
  CUDA_ERROR_CHECK(cudaMalloc(&d_freqs, sizeof(h_freqs)));
  CUDA_ERROR_CHECK(cudaMemset(d_freqs, 0, sizeof(h_freqs)));
  constexpr int block_size = 256;
  const int block_count = (len + block_size - 1) / block_size;
  count_frequencies<<<block_count, block_size>>>(d_data, len, d_freqs);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  int h_freqs_res[258];
  CUDA_ERROR_CHECK(cudaMemcpy(h_freqs_res, d_freqs, sizeof(h_freqs_res),
                              cudaMemcpyDeviceToHost));
  for (int i = 0; i < 258; i++) {
    ASSERT_EQ(h_freqs_res[i], h_freqs[i]);
  }
}

// Test if tree building doesn't crash
TEST(compress, huffman_tree_builder_stability) {
  uint8_t len[max_n_groups][max_alphabet_size];
  int32_t code[max_n_groups][max_alphabet_size];
  constexpr int data_len = 10000;
  uint16_t data[data_len];
  uint16_t *device_data;
  srand(2137);
  CUDA_ERROR_CHECK(cudaMalloc(&device_data, data_len * sizeof(*device_data)));
  for (int alphabet_size = 1; alphabet_size < 256; alphabet_size++) {
    for (int i = 0; i < data_len; i++) {
      data[i] = rand() % alphabet_size;
    }
    CUDA_ERROR_CHECK(
        cudaMemcpy(device_data, data, sizeof(data), cudaMemcpyHostToDevice));
    uint8_t *selectors;
    int num_selectors = huffman_build_trees(
        device_data, data_len, alphabet_size, len, code, selectors, 0);
    for (int i = 0; i < num_selectors; i++) {
      if (selectors[i] > 5) {
        fprintf(stderr, "wrong sel[%d] = %d, num_sels=%d\n", i, selectors[i],
                num_selectors);
      }
      ASSERT_GE(selectors[i], 0);
      ASSERT_LE(selectors[i], 5);
    }
    delete[] selectors;
  }
  CUDA_ERROR_CHECK(cudaFree(device_data));
}

typedef struct bnt {
  bool is_leaf = false;
  uint16_t value = 259;
  bnt *left = nullptr;
  bnt *right = nullptr;
  ~bnt() {
    if (left != nullptr) {
      delete left;
    }
    if (right != nullptr) {
      delete right;
    }
  }
} binary_tree_node;

void realise_tree(uint8_t len[max_alphabet_size],
                  int32_t code[max_alphabet_size], binary_tree_node *&root,
                  int alphabet_size) {
  root = new binary_tree_node();
  for (int i = 0; i < alphabet_size; i++) {
    int l = len[i];
    int c = code[i];
    binary_tree_node *p = root;
    while (l > 0) {
      if (0b1 & (c >> (l - 1))) {
        if (!p->right) {
          p->right = new binary_tree_node();
        }
        p = p->right;
      } else {
        if (!p->left) {
          p->left = new binary_tree_node();
        }
        p = p->left;
      }
      l--;
    }
    p->is_leaf = true;
    p->value = i;
  }
}

typedef struct bit_stream {
  uint32_t *data;
  int full_len;
  int cursor = 0;
  uint8_t peek() {
    int idx = cursor / 32;
    int shift = 31 - cursor % 32;
    return (data[idx] >> shift) & 0b1;
  }
  uint8_t consume_bit() {
    if (cursor >= full_len) {
      return 0xff;
    }
    uint8_t bit = peek();
    cursor++;
    return bit;
  }
} bit_stream;

void decode_huffman(bit_stream str, binary_tree_node *root[6],
                    uint8_t *selectors, int num_sels, uint16_t *decoded) {
  uint8_t bit;
  int emited_count = 0;
  int tree_id = selectors[0];
  binary_tree_node *p = root[tree_id];
  binary_tree_node *old_p = nullptr;
  binary_tree_node *older_p = nullptr;
  while ((bit = str.consume_bit()) != 0xff) {
    older_p = old_p;
    old_p = p;
    if (bit) {
      p = p->right;
    } else {
      p = p->left;
    }
    if (p == nullptr) {
      fprintf(stderr, "p==nullptr: for bit %d after %d chars emmited\n",
              str.cursor, emited_count);
      return;
    }
    if (p->is_leaf) {
      decoded[emited_count++] = p->value;
      tree_id = selectors[emited_count / 50];
      p = root[tree_id];
    }
  }
}

TEST(compress, huffman_encoding) {
  uint8_t len[max_n_groups][max_alphabet_size];
  int32_t code[max_n_groups][max_alphabet_size];
  constexpr int data_len = 10000;
  uint16_t data[data_len];
  uint16_t *device_data;
  srand(2137);
  CUDA_ERROR_CHECK(cudaMalloc(&device_data, data_len * sizeof(*device_data)));
  int alphabet_size = 15;
  int shift = 0;
  srand(2137);
  for (int i = 0; i < data_len; i++) {
    if (i % 50 == 0) {
      shift = rand() % 6;
    }
    data[i] = rand() % (alphabet_size / 2) + shift;
  }
  CUDA_ERROR_CHECK(
      cudaMemcpy(device_data, data, sizeof(data), cudaMemcpyHostToDevice));
  uint8_t *selectors;
  int num_selectors = huffman_build_trees(device_data, data_len, alphabet_size,
                                          len, code, selectors, 0);

  uint32_t *dev_encoded;
  uint8_t *dev_selectors;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_selectors, num_selectors));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_selectors, selectors, num_selectors,
                              cudaMemcpyHostToDevice));

  int encoded_len =
      huffman_encode(device_data, data_len, alphabet_size, dev_encoded, len,
                     code, dev_selectors, num_selectors);
  const int total_words = (encoded_len + 31) / 32;
  uint32_t *host_encoded = new uint32_t[total_words];
  CUDA_ERROR_CHECK(cudaMemcpy(host_encoded, dev_encoded,
                              total_words * sizeof(*host_encoded),
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(device_data));
  CUDA_ERROR_CHECK(cudaFree(dev_selectors));
  CUDA_ERROR_CHECK(cudaFree(dev_encoded));
  binary_tree_node *root[6];
  for (int i = 0; i < 6; i++) {
    realise_tree(&len[i][0], &code[i][0], root[i], alphabet_size);
  }
  bit_stream str;
  str.cursor = 0;
  str.full_len = encoded_len;
  str.data = host_encoded;
  uint16_t decoded[data_len];
  for (int i = 0; i < num_selectors; i++) {
    ASSERT_GE(selectors[i], 0);
    ASSERT_LE(selectors[i], 5);
  }
  decode_huffman(str, root, selectors, num_selectors, decoded);
  for (int i = 0; i < data_len; i++) {
    if (decoded[i] != data[i]) {
      fprintf(stderr, "data differs at [%d]: dec=%d != %d=data\n", i,
              decoded[i], data[i]);
    }
    ASSERT_EQ(decoded[i], data[i]);
  }
  delete[] selectors;
  delete[] host_encoded;
}
