#ifndef UTILS

#define CUDA_ERROR_CHECK(expr)                                                 \
  do {                                                                         \
    cudaError_t cudaStatus = expr;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      fprintf(stderr, "%s failed! At line %d, in %s\nError: %s\n\t %s\n",      \
              #expr, __LINE__, __FILE__, cudaGetErrorName(cudaStatus),         \
              cudaGetErrorString(cudaStatus));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // !UTILS
