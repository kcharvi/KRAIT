__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}


//Optimized kernel leveraging shared memory
__global__ void matrixMultiplySharedKernel(const float* A, const float* B, float* C, int m, int n, int k, int TILE_WIDTH) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col