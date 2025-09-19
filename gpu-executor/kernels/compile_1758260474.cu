// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758260474
// Type: compile_only

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int m, int n, int k) {
  // Thread ID
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bz = blockIdx.z;

  // Block ID
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Block dimensions
  int bw = blockDim.x;
  int bh = blockDim.y;

  // Shared memory for storing sub-matrices of A and B
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  // Global memory indices
  int row = by * bh + ty;
  int col = bx * bw + tx;

  // Result accumulator
  float Cvalue = 0.0f;

  // Loop over tiles
  for (int i = 0; i < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
    // Load sub-matrices into shared memory
    int aRow = row;
    int aCol = i * TILE_WIDTH + tx;
    int bRow = i * TILE_WIDTH + ty;
    int bCol = col;

    if (aRow < m && aCol < k) {
      As[ty][tx] = A[aRow * k + aCol];
    } else {
      As[ty][tx] = 0.0f;
    }

    if (bRow < k && bCol < n) {
      Bs[ty][tx] = B[bRow * n + bCol];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Perform matrix multiplication within the tile
    for (int j = 0; j < TILE_WIDTH; ++j) {
      Cvalue += As[ty][j] * Bs[j][tx];
    }

    __syncthreads();
  }

  // Write result to global memory
  if (row < m && col < n) {
    C[row * n + col] = Cvalue;
  }
}


#define TILE_WIDTH 32

void matrixMultiply(const float *A, const float *B, float *C, int m, int n, int k) {
  // Define grid and block dimensions
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
  dim3 gridDim((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);


  matrixMultiplyKernel<<<gridDim, blockDim>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize(); // Ensure kernel execution completes
}