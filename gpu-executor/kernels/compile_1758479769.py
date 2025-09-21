# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758479769
    # Type: compile_only

    import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
kernel_code = """
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
  
  // Error Handling: Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
  }

  return {C};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_wrapper", &matmul_wrapper, "Matrix multiplication");
}
"""

# Load the CUDA kernel
try:
    custom_matmul = load_inline(
        name="custom_matmul",
        cpp_sources=kernel_code,
        verbose=True,
    )
except Exception as e:
    print(f"Error loading CUDA kernel: {e}")
    exit(1)


# Python wrapper for the kernel
def custom_matmul_wrapper(A, B):
    try:
        result = custom_matmul.matmul_wrapper(A, B)
        return result[0]
    except Exception as e:
        print(f"Error during kernel execution: {e}")
        return None


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(128, 256, device=device)
    B = torch.randn(256, 512, device=device)

    # Standard PyTorch matmul for comparison
    standard_result = torch.matmul(A, B)

    # Custom CUDA kernel matmul
    custom_result = custom_matmul_wrapper(A,B)

    if custom_result is not None:
        print("Standard Matmul:")
        print(standard_result)
        print("\nCustom Matmul:")
        print(custom_result)
        print("\nDifference:")
        print(torch.abs(standard_result - custom_result).max())