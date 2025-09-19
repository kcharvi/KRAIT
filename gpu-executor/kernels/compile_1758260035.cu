// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758260035
// Type: compile_only

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k) {
  // Thread indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check bounds
  if (row < m && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = sum;
  }
}


//Optimized kernel using shared memory
__global__ void matrixMultiplySharedMemKernel(const float* A, const float* B, float* C, int m, int n, int k, int tile_size) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < n; i += TILE_SIZE) {
        if (row < m && i + threadIdx.x < n) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * n + i + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < k && i + threadIdx.y < n) {
            shared_B[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * k + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

#define TILE_SIZE 32