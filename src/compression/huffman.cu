#include "compression.h"
#include "utils.h"

#include <cstdint>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

constexpr int lesser_icost = 0;
constexpr int greater_icost = 15;

#define WEIGHTOF(zz0) ((zz0) & 0xffffff00)
#define DEPTHOF(zz1) ((zz1) & 0x000000ff)
#define MYMAX(zz2, zz3) ((zz2) > (zz3) ? (zz2) : (zz3))

#define ADDWEIGHTS(zw1, zw2)                                                   \
  (WEIGHTOF(zw1) + WEIGHTOF(zw2)) | (1 + MYMAX(DEPTHOF(zw1), DEPTHOF(zw2)))

#define DOWNHEAP(z)                                                            \
  {                                                                            \
    int32_t zz, yy, tmp;                                                       \
    zz = z;                                                                    \
    tmp = heap[zz];                                                            \
    while (true) {                                                             \
      yy = zz << 1;                                                            \
      if (yy > nHeap)                                                          \
        break;                                                                 \
      if (yy < nHeap && weight[heap[yy + 1]] < weight[heap[yy]])               \
        yy++;                                                                  \
      if (weight[tmp] < weight[heap[yy]])                                      \
        break;                                                                 \
      heap[zz] = heap[yy];                                                     \
      zz = yy;                                                                 \
    }                                                                          \
    heap[zz] = tmp;                                                            \
  }

#define UPHEAP(z)                                                              \
  {                                                                            \
    int32_t zz, tmp;                                                           \
    zz = z;                                                                    \
    tmp = heap[zz];                                                            \
    while (weight[tmp] < weight[heap[zz >> 1]]) {                              \
      heap[zz] = heap[zz >> 1];                                                \
      zz >>= 1;                                                                \
    }                                                                          \
    heap[zz] = tmp;                                                            \
  }

void huff_make_code_lengths(int32_t freq[max_alphabet_size],
                            uint8_t len[max_alphabet_size], int32_t alphaSize,
                            int32_t maxLen) {
  /*--
     Nodes and heap entries run from 1.  Entry 0
     for both the heap and nodes is a sentinel.
  --*/
  int32_t nNodes, nHeap, n1, n2, i, j, k;
  bool tooLong;

  int32_t heap[max_alphabet_size + 2];
  int32_t weight[max_alphabet_size * 2];
  int32_t parent[max_alphabet_size * 2];

  for (i = 0; i < alphaSize; i++)
    weight[i + 1] = (freq[i] == 0 ? 1 : freq[i]) << 8;

  while (1) {

    nNodes = alphaSize;
    nHeap = 0;

    heap[0] = 0;
    weight[0] = 0;
    parent[0] = -2;

    for (i = 1; i <= alphaSize; i++) {
      parent[i] = -1;
      nHeap++;
      heap[nHeap] = i;
      UPHEAP(nHeap);
    }

    while (nHeap > 1) {
      n1 = heap[1];
      heap[1] = heap[nHeap];
      nHeap--;
      DOWNHEAP(1);
      n2 = heap[1];
      heap[1] = heap[nHeap];
      nHeap--;
      DOWNHEAP(1);
      nNodes++;
      parent[n1] = parent[n2] = nNodes;
      weight[nNodes] = ADDWEIGHTS(weight[n1], weight[n2]);
      parent[nNodes] = -1;
      nHeap++;
      heap[nHeap] = nNodes;
      UPHEAP(nHeap);
    }

    tooLong = false;
    for (i = 1; i <= alphaSize; i++) {
      j = 0;
      k = i;
      while (parent[k] >= 0) {
        k = parent[k];
        j++;
      }
      len[i - 1] = j;
      if (j > maxLen)
        tooLong = true;
    }

    if (!tooLong)
      break;

    for (i = 1; i <= alphaSize; i++) {
      j = weight[i] >> 8;
      j = 1 + (j / 2);
      weight[i] = j << 8;
    }
  }
}

__global__ void count_frequencies(uint16_t *data_in, int data_len, int *freqs) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= data_len) {
    return;
  }
  uint16_t chr = data_in[idx];
  atomicAdd(&freqs[chr], 1);
}

inline void huff_assign_codes(int32_t *code, uint8_t *length, int32_t minLen,
                              int32_t maxLen, int32_t alphaSize) {
  int vec = 0;
  for (int len = minLen; len <= maxLen; len++) {
    for (int symbol = 0; symbol < alphaSize; symbol++)
      if (length[symbol] == len) {
        code[symbol] = vec;
        vec++;
      };
    vec <<= 1;
  }
}

void generate_initial_assignment(int n_groups, int data_len, int alphabet_size,
                                 int32_t freqs[max_alphabet_size],
                                 uint8_t len[max_n_groups][max_alphabet_size]) {
  int group_start = 0;
  int group_end;
  int remaining_freqs = data_len;
  for (int n_part = n_groups; n_part > 0; n_part--) {
    int t_freq = remaining_freqs / n_part;
    group_end = group_start - 1;

    int a_freq = 0;

    while (a_freq < t_freq && group_end < alphabet_size) {
      group_end++;
      a_freq += freqs[group_end];
    }
    if (group_end > group_start && n_part != n_groups && n_part != 1 &&
        ((n_groups - n_part) % 2 == 1)) {
      a_freq -= freqs[group_end];
      group_end--;
    }
    // fprintf(stderr,
    //         "      initial group %d, [%d .. %d], "
    //         "has %d syms (%4.1f%%)\n",
    //         n_part, group_start, group_end, a_freq,
    //         (100.0 * (float)a_freq) / (float)(data_len));

    for (int symbol = 0; symbol < alphabet_size; symbol++) {
      if (symbol >= group_start && symbol <= group_end) {
        len[n_part - 1][symbol] = lesser_icost;
      } else {
        len[n_part - 1][symbol] = greater_icost;
      }
    }
    group_start = group_end + 1;
    remaining_freqs -= a_freq;
  }
}

__constant__ uint8_t c_lens[max_n_groups][max_alphabet_size];
__global__ void count_segment_frequencies(const uint16_t *data, int32_t *rfreq,
                                          uint8_t *selectors, int data_len,
                                          int alphabet_size, int n_groups) {
  constexpr int group_size = 50;
  int segmentIdx = blockIdx.x;
  int gs = segmentIdx * group_size; // BZ_G_SIZE = 50
  if (gs >= data_len)
    return;

  int ge = gs + group_size - 1;
  if (ge >= data_len) {
    ge = data_len - 1;
  }
  int actual_segment_size = ge - gs + 1;

  __shared__ uint32_t s_costs[max_n_groups];
  if (threadIdx.x < max_n_groups) {
    s_costs[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < actual_segment_size; i += blockDim.x) {
    uint16_t symbol = data[gs + i];
    for (int group = 0; group < n_groups; group++) {
      // Each thread adds its symbol's cost to the shared segment cost
      atomicAdd(&s_costs[group], (uint32_t)c_lens[group][symbol]);
    }
  }
  __syncthreads();

  __shared__ int best_group;
  if (threadIdx.x == 0) {
    uint32_t min_cost = 0xFFFFFFFF;
    int curr_best_group = -1;
    for (int group = 0; group < n_groups; group++) {
      if (s_costs[group] < min_cost) {
        min_cost = s_costs[group];
        curr_best_group = group;
      }
    }
    best_group = curr_best_group;
    selectors[segmentIdx] = (uint8_t)curr_best_group;
  }
  __syncthreads();

  int bg = best_group;
  for (int i = threadIdx.x; i < actual_segment_size; i += blockDim.x) {
    uint16_t symbol = data[gs + i];
    // if (symbol < max_alphabet_size) {
    atomicAdd(&rfreq[bg * max_alphabet_size + symbol], 1);
    // } else {
    //   printf("Rogue symbol detected: %d at data index %d\n", symbol, gs + i);
    // }
  }
}
// __host__ __device__ void VALIDATE_SELS(uint8_t *d_sels, int num_sels) {
//   for (int i = 0; i < num_sels; i++) {
//     if (d_sels[i] >= 6) {
//       printf("%dth selector incorrent. (%d)\n", i, d_sels[i]);
//     }
//   }
// }
//
// __global__ void VALIDATE_DEV_SELS(uint8_t *d_sels, int num_sels) {
//   VALIDATE_SELS(d_sels, num_sels);
// }

int huffman_build_trees(uint16_t *device_data_in, int data_in_len,
                        int alphabet_size,
                        uint8_t len[max_n_groups][max_alphabet_size],
                        int32_t code[max_n_groups][max_alphabet_size],
                        uint8_t *&selectors, cudaStream_t stream) {
  alphabet_size += 2;
  for (int group = 0; group < max_n_groups; group++) {
    for (int symbol = 0; symbol < alphabet_size; symbol++) {
      len[group][symbol] = greater_icost;
    }
  }
  int n_groups;
  if (data_in_len < 200) {
    n_groups = 2;
  } else if (data_in_len < 600) {
    n_groups = 3;
  } else if (data_in_len < 1200) {
    n_groups = 4;
  } else if (data_in_len < 2400) {
    n_groups = 5;
  } else {
    n_groups = 6;
  }
  int32_t freqs[max_alphabet_size];
  {
    int32_t *dev_freqs;
    CUDA_ERROR_CHECK(cudaMalloc(&dev_freqs, sizeof(freqs)));
    constexpr int block_size = 256;
    const int block_count = (data_in_len + block_size - 1) / block_size;
    count_frequencies<<<block_count, block_size, 0, stream>>>(
        device_data_in, data_in_len, dev_freqs);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(freqs, dev_freqs, sizeof(freqs),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR_CHECK(cudaFreeAsync(dev_freqs, stream));
  }
  generate_initial_assignment(n_groups, data_in_len, alphabet_size, freqs, len);
  int32_t rfreq[max_n_groups][max_alphabet_size];
  int32_t *dev_rfreq;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_rfreq, sizeof(rfreq)));
  const int num_selectors = (data_in_len + 49) / 50;
  // uint8_t selectors[num_selectors];
  selectors = new uint8_t[num_selectors];
  uint8_t *dev_selectors;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_selectors, num_selectors * sizeof(*dev_selectors)));
  constexpr int n_iters = 4;
  for (int iter = 0; iter < n_iters; iter++) {
    CUDA_ERROR_CHECK(cudaMemsetAsync(dev_rfreq, 0, sizeof(rfreq), stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyToSymbolAsync(c_lens, len, max_n_groups * max_alphabet_size,
                                0, cudaMemcpyHostToDevice, stream));
    constexpr int block_size = 32;
    CUDA_ERROR_CHECK(cudaMemsetAsync(
        dev_selectors, 0, num_selectors * sizeof(*dev_selectors), stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    count_segment_frequencies<<<num_selectors, block_size, 0, stream>>>(
        device_data_in, dev_rfreq, dev_selectors, data_in_len, alphabet_size,
        n_groups);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(rfreq, dev_rfreq, sizeof(rfreq),
                                     cudaMemcpyDeviceToHost, stream));
    for (int group = 0; group < n_groups; group++) {
      huff_make_code_lengths(rfreq[group], len[group], alphabet_size, 17);
    }
  }
  CUDA_ERROR_CHECK(cudaFreeAsync(dev_rfreq, stream));
  CUDA_ERROR_CHECK(cudaMemcpyAsync(selectors, dev_selectors,
                                   num_selectors * sizeof(*dev_selectors),
                                   cudaMemcpyDeviceToHost, stream));
  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(dev_selectors, stream));
  // /*--- Compute MTF values for the selectors. ---*/
  // {
  //   uint8_t pos[max_n_groups], ll_i, tmp2, tmp;
  //   for (int i = 0; i < n_groups; i++) {
  //     pos[i] = i;
  //   }
  //   for (int i = 0; i < num_selectors; i++) {
  //     ll_i = selectors[i];
  //     int j = 0;
  //     tmp = pos[j];
  //     while (ll_i != tmp) {
  //       j++;
  //       tmp2 = tmp;
  //       tmp = pos[j];
  //       pos[j] = tmp2;
  //     };
  //     pos[0] = tmp;
  //     selectorMtf[i] = j;
  //   }
  //   };
  /*--- Assign actual codes for the tables. --*/
  for (int group = 0; group < n_groups; group++) {
    int minLen = 32;
    int maxLen = 0;
    for (int symbol = 0; symbol < alphabet_size; symbol++) {
      if (len[group][symbol] > maxLen)
        maxLen = len[group][symbol];
      if (len[group][symbol] < minLen)
        minLen = len[group][symbol];
    }
    huff_assign_codes(code[group], len[group], minLen, maxLen, alphabet_size);
  }
  return num_selectors;
}
