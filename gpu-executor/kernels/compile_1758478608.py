# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758478608
    # Type: compile_only

    import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
kernel_code = """
#include <torch/extension.h>

#define BLOCK_SIZE 32

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int M, int N, int K) {
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

std::vector<torch::Tensor> matrixMultiply(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C = torch::zeros({M, N}, A.options());

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matrixMultiplyKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  cudaDeviceSynchronize();
  return {C};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrixMultiply", &matrixMultiply, "Matrix multiplication kernel");
}
"""

# Load the kernel
try:
    custom_kernel = load_inline(
        name="custom_matmul",
        cpp_sources=[kernel_code],
        cuda=True,
        verbose=True
    )
except RuntimeError as e:
    print(f"CUDA compilation failed: {e}")
    exit(1)


def custom_matmul_wrapper(A, B):
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Inputs must be CUDA tensors.")
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise RuntimeError("Inputs must be float32 tensors.")
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError("Inputs must be 2D tensors.")
    if A.size(1) != B.size(0):
        raise RuntimeError("Incompatible matrix dimensions.")

    try:
        result = custom_kernel.matrixMultiply(A, B)
        return result[0]
    except RuntimeError as e:
        print(f"CUDA kernel execution failed: {e}")
        exit(1)


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(1024, 512, device=device, dtype=torch.float32)
    B = torch.randn(512, 256, device=device, dtype=torch.float32)

    try:
        C = custom_matmul_wrapper(A, B)
        print("Custom kernel result shape:", C.shape)
        #Further verification can be added here by comparing with standard matmul
    except RuntimeError as e:
      print(f"Error during example usage: {e}")
      exit(1)