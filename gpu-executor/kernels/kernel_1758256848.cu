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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < (k + TILE_WIDTH -1)/TILE_WIDTH; ++i) {
        int tileRow = i * TILE_WIDTH + threadIdx.y;
        int tileCol = i * TILE_WIDTH + threadIdx.x;

        if(tileRow < k && col < n)
            tileB[threadIdx.y][threadIdx.x] = B[tileRow * n + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        if(row < m && tileCol < k)
            tileA[threadIdx.y][threadIdx.x] = A[row * k + tileCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int j=0; j < TILE_WIDTH; ++j){
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}