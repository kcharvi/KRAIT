# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758478959
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
  m.def("matmul_wrapper", &matmul_wrapper, "Matrix Multiplication");
}
"""

# Load the CUDA kernel
try:
    custom_matmul = load_inline(
        name="custom_matmul",
        cpp_sources=cuda_kernel,
        verbose=True,
    )
except RuntimeError as e:
    print(f"Error loading CUDA kernel: {e}")
    exit(1)


def custom_matmul_wrapper(A, B):
    try:
        if not A.is_cuda or not B.is_cuda:
            raise RuntimeError("Inputs must be CUDA tensors.")
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("Inputs must be 2D tensors.")
        if A.size(1) != B.size(0):
            raise RuntimeError("Incompatible matrix dimensions.")
        return custom_matmul.matmul_wrapper(A, B)[0]
    except RuntimeError as e:
        print(f"Error during custom matrix multiplication: {e}")
        return None


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(128, 256, device=device)
    B = torch.randn(256, 512, device=device)

    C_custom = custom_matmul_wrapper(A, B)
    
    if C_custom is not None:
        C_pytorch = torch.matmul(A,B)
        diff = torch.max(torch.abs(C_custom-C_pytorch))
        print(f"Max difference between PyTorch and custom kernel: {diff}")
    else:
        print("Custom matrix multiplication failed.")