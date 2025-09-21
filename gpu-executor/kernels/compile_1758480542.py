# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758480542
    # Type: compile_only

    import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
cuda_kernel = """
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

std::vector<torch::Tensor> matmul_wrapper(torch::Tensor A, torch::Tensor B) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_wrapper", &matmul_wrapper, "Matrix multiplication kernel");
}
"""

# Load the CUDA kernel
try:
    matmul_module = load_inline(
        name="matmul_cuda",
        cpp_sources=cuda_kernel,
        verbose=True,
    )
except RuntimeError as e:
    print(f"Error loading CUDA kernel: {e}")
    exit(1)


def matmul_cuda(A, B):
    """
    Python wrapper for the CUDA matrix multiplication kernel.
    """
    if not A.is_cuda or not B.is_cuda:
        raise RuntimeError("Input tensors must be on CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise RuntimeError("Input tensors must have dtype torch.float32.")
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError("Input tensors must be 2-dimensional.")
    if A.size(1) != B.size(0):
        raise RuntimeError("Incompatible matrix dimensions for multiplication.")

    try:
        result = matmul_module.matmul_wrapper(A, B)
        return result[0]
    except RuntimeError as e:
        print(f"Error during CUDA kernel execution: {e}")
        exit(1)


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(1024, 512, device=device, dtype=torch.float32)
    B = torch.randn(512, 2048, device=device, dtype=torch.float32)

    try:
        C = matmul_cuda(A, B)
        print("Matrix multiplication successful.")
        print("Result shape:", C.shape)
        #Further verification can be done by comparing with torch.matmul
    except RuntimeError as e:
        print(f"An error occurred: {e}")