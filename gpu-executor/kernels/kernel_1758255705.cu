__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int widthA, int widthB, int heightA) {
  // Thread ID
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Block dimensions
  int blockSizeX = blockDim.x;
  int blockSizeY = blockDim.y;

  // Global indices
  int row = by * blockSizeY + ty;
  int col = bx * blockSizeX + tx;

  // Shared memory
  __shared__ float sharedA[32][32];
  __shared__ float sharedB[32][32];

  float sum = 0.0f;

  // Loop ov