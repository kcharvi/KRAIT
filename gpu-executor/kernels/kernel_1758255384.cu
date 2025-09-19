__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k) {
  // Thread ID
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check bounds
  if (row < m && col < k) {
    floa