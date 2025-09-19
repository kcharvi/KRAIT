// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758260630
// Type: compile_only

__global__ void fastMatrixMultiply(const float *A, const float *B, float *C, int m, int n, int k) {
  // Thread ID
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Block dimensions
  int block_width = blockDim.x;
  int block_height = blockDim.y;

  // Global indices
  int row = by * block_height + ty;
  int col = bx * block_width + tx;

  // Shared memory for tiles
  __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

  float sum = 0.0f;

  // Loop over tiles
  for (int i = 0; i < k; i += TILE_WIDTH) {
    // Load tiles into shared memory
    if (row < m && i + tx < k) {
      shared_A[ty][tx] = A[row * k + i + tx];
    } else {
      shared_A[ty][tx] = 0.0f;
    }
    if (i + ty < k && col < n) {
      shared_B[ty][tx] = B[(i + ty) * n + col];
    } else {
      shared_B[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Perform matrix multiplication within tile
    for (int j = 0; j < TILE_WIDTH; ++j) {
      sum += shared_A[ty][j] * shared_B[j][tx];
    }

    __syncthreads();
  }

  // Store result
  if (row < m && col < n) {
    C[row * n + col] = sum;
  }
}


// Define TILE_WIDTH. Experiment to find optimal value for your H100.
#define TILE_WIDTH 32