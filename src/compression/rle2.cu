#include "compression.h"
#include "utils.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

static constexpr int RLE2_CHUNK_SIZE = 512;
static constexpr int MAX_SYMS_PER_CHUNK = RLE2_CHUNK_SIZE + 24;

__global__ void rle2_phase1(
    const uint8_t *__restrict__ d_in,
    size_t n,
    uint16_t *d_local_syms,
    uint32_t *d_counts,
    int table_size)
{
  const int chunk_id = blockIdx.x;
  const int in_start = chunk_id * RLE2_CHUNK_SIZE;
  const int in_end = min(in_start + RLE2_CHUNK_SIZE, (int)n);
  const int in_len = in_end - in_start;

  if (in_len <= 0)
  {
    d_counts[chunk_id] = 0;
    return;
  }

  uint16_t *out = d_local_syms + (size_t)chunk_id * MAX_SYMS_PER_CHUNK;
  int out_pos = 0;

  int i = 0;
  if (in_start > 0 && d_in[in_start - 1] == 0)
  {
    while (i < in_len && d_in[in_start + i] == 0)
    {
      i++;
    }
  }

    while (i < in_len)
    {
        const uint8_t byte = d_in[in_start + i];

        if (byte != 0)
        {
            out[out_pos++] = (uint16_t)(byte + 1);
            i++;
        }
        else
        {
            int global_run_start = in_start + i;
            int global_pos = global_run_start;
            while (global_pos < (int)n && d_in[global_pos] == 0)
            {
                global_pos++;
            }
            int k = global_pos - global_run_start;

            i = min(global_pos - in_start, in_len);

            while (k > 0)
            {
                k--;
                out[out_pos++] = (k & 1) ? 1 : 0; // RUNB : RUNA
                k >>= 1;
            }
        }
    }

    if (chunk_id == gridDim.x - 1)
    {
        out[out_pos++] = (uint16_t)(table_size + 1); // EOB
    }

    d_counts[chunk_id] = (uint32_t)out_pos;
}

__global__ void rle2_phase3(
    const uint16_t *__restrict__ d_local_syms,
    const uint32_t *__restrict__ d_offsets,
    const uint32_t *__restrict__ d_counts,
    uint16_t *d_out)
{
  const int chunk_id = blockIdx.x;
  const uint32_t cnt = d_counts[chunk_id];
  if (cnt == 0)
    return;

  const uint16_t *src = d_local_syms + (size_t)chunk_id * MAX_SYMS_PER_CHUNK;
  uint16_t *dst = d_out + d_offsets[chunk_id];

  for (uint32_t j = threadIdx.x; j < cnt; j += blockDim.x)
  {
    dst[j] = src[j];
  }
}

void rle2_compress(const uint8_t *d_in,
                   int n,
                   uint16_t *d_out,
                   uint32_t *d_out_len,
                   int table_size,
                   cudaStream_t stream)
{
  if (n == 0)
  {
    CUDA_ERROR_CHECK(cudaMemsetAsync(d_out_len, 0, sizeof(uint32_t), stream));
    return;
  }

  const int num_chunks = (int)((n + RLE2_CHUNK_SIZE - 1) / RLE2_CHUNK_SIZE);

  uint16_t *d_local_syms_raw;
  uint32_t *d_counts_raw;
  uint32_t *d_offsets_raw;
  size_t local_syms_bytes = (size_t)num_chunks * MAX_SYMS_PER_CHUNK * sizeof(uint16_t);
  size_t counts_bytes = (num_chunks + 1) * sizeof(uint32_t);
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_local_syms_raw, local_syms_bytes, stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_counts_raw, counts_bytes, stream));
  CUDA_ERROR_CHECK(cudaMallocAsync(&d_offsets_raw, counts_bytes, stream));
  CUDA_ERROR_CHECK(cudaMemsetAsync(d_counts_raw, 0, counts_bytes, stream));
  CUDA_ERROR_CHECK(cudaMemsetAsync(d_offsets_raw, 0, counts_bytes, stream));

  thrust::device_ptr<uint32_t> d_counts(d_counts_raw);
  thrust::device_ptr<uint32_t> d_offsets(d_offsets_raw);

  rle2_phase1<<<num_chunks, 1, 0, stream>>>(
      d_in, n,
      d_local_syms_raw,
      d_counts_raw,
      table_size);
  CUDA_ERROR_CHECK(cudaGetLastError());

  thrust::exclusive_scan(
      thrust::cuda::par.on(stream),
      d_counts, d_counts + num_chunks + 1,
      d_offsets);

  CUDA_ERROR_CHECK(cudaMemcpyAsync(
      d_out_len,
      d_offsets_raw + num_chunks,
      sizeof(uint32_t),
      cudaMemcpyDeviceToDevice,
      stream));

  constexpr int COPY_THREADS = 32;
  rle2_phase3<<<num_chunks, COPY_THREADS, 0, stream>>>(
      d_local_syms_raw,
      d_offsets_raw,
      d_counts_raw,
      d_out);
  CUDA_ERROR_CHECK(cudaGetLastError());

  CUDA_ERROR_CHECK(cudaFreeAsync(d_local_syms_raw, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_counts_raw, stream));
  CUDA_ERROR_CHECK(cudaFreeAsync(d_offsets_raw, stream));
}
