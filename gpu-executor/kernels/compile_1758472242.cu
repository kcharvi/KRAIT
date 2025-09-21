// COMPILATION REQUEST
// Hardware: NVIDIA T4
// Backend: PYTORCH_CUDA_EXTENSION
// Timestamp: 1758472242
// Type: compile_only

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for matrix multiplication
matmul_kernel = """
#include <torch/script.h>
#include <torch/extension.h>

#define BLOCK_SIZE 32

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

std::vector<torch::Tensor> matmul_cuda(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = torch::zeros({M, N}, A.options());

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matmul_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  cudaDeviceSynchronize();
  return {C};
}
"""

# Load the CUDA kernel using load_inline
matmul_module = load_inline(
    name="matmul_cuda_module",
    cpp_sources="",
    cuda_sources=[matmul_kernel],
    verbose=True,
)

# Python wrapper function for the CUDA kernel
def matmul_wrapper(A, B):
    try:
        result = matmul_module.matmul_cuda(A, B)
        return result[0]
    except RuntimeError as e:
        print(f"CUDA error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Create example tensors
    A = torch.randn(128, 256, device='cuda')
    B = torch.randn(256, 512, device='cuda')

    # Perform matrix multiplication using the custom CUDA kernel
    C = matmul_wrapper(A, B)

    if C is not None:
        print("Matrix multiplication successful.")
        print(f"Shape of C: {C.shape}")
    else:
        print("Matrix multiplication failed.")