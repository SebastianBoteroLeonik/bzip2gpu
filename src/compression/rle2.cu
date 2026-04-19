#include "compression.h"
#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

static constexpr uint16_t RLE2_RUNA = 256;
static constexpr uint16_t RLE2_RUNB = 257;
static constexpr int RLE2_CHUNK_SIZE = 512;
static constexpr int MAX_SYMS_PER_CHUNK = RLE2_CHUNK_SIZE + 24;

__global__ void rle2_phase1(
    const uint8_t *__restrict__ d_in,
    size_t n,
    uint16_t *d_local_syms,
    uint32_t *d_counts)
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
      out[out_pos++] = (uint16_t)(byte);
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
        out[out_pos++] = (k & 1) ? RLE2_RUNB : RLE2_RUNA;
        k >>= 1;
      }
    }
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
                   cudaStream_t stream)
{
  if (n == 0)
  {
    CUDA_ERROR_CHECK(cudaMemsetAsync(d_out_len, 0, sizeof(uint32_t), stream));
    return;
  }

  const int num_chunks = (int)((n + RLE2_CHUNK_SIZE - 1) / RLE2_CHUNK_SIZE);

  thrust::device_vector<uint16_t> d_local_syms(
      (size_t)num_chunks * MAX_SYMS_PER_CHUNK);

  thrust::device_vector<uint32_t> d_counts(num_chunks + 1, 0u);
  thrust::device_vector<uint32_t> d_offsets(num_chunks + 1, 0u);

  rle2_phase1<<<num_chunks, 1, 0, stream>>>(
      d_in, n,
      thrust::raw_pointer_cast(d_local_syms.data()),
      thrust::raw_pointer_cast(d_counts.data()));
  CUDA_ERROR_CHECK(cudaGetLastError());

  thrust::exclusive_scan(
      thrust::cuda::par.on(stream),
      d_counts.begin(), d_counts.begin() + num_chunks + 1,
      d_offsets.begin());

  CUDA_ERROR_CHECK(cudaMemcpyAsync(
      d_out_len,
      thrust::raw_pointer_cast(d_offsets.data()) + num_chunks,
      sizeof(uint32_t),
      cudaMemcpyDeviceToDevice,
      stream));

  constexpr int COPY_THREADS = 32;
  rle2_phase3<<<num_chunks, COPY_THREADS, 0, stream>>>(
      thrust::raw_pointer_cast(d_local_syms.data()),
      thrust::raw_pointer_cast(d_offsets.data()),
      thrust::raw_pointer_cast(d_counts.data()),
      d_out);
  CUDA_ERROR_CHECK(cudaGetLastError());
}
