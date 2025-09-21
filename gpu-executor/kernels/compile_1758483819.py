# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758483819
    # Type: compile_only

    import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code as a single string
cuda_kernel = """
#include <torch/extension.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH 32

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
"""

# C++ wrapper and Pybind11 code as a separate string
cpp_wrapper_and_pybind = """
#include <torch/extension.h>
#include <vector>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K);

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
  m.def("matmul_wrapper", &matmul_wrapper, "Matrix multiplication");
}
"""

# Load the CUDA kernel using load_inline
matmul_module = load_inline(
    name="matmul_cuda", 
    cpp_sources=[cpp_wrapper_and_pybind], 
    cuda_sources=[cuda_kernel],           
    verbose=True,
    extra_cflags=['-O3'],                  
    extra_cuda_cflags=['-O3']             
)


def matmul_cuda(A, B):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if A.device.type != 'cuda' or B.device.type != 'cuda':
        A = A.cuda()
        B = B.cuda()
    
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError("Input tensors must be 2-dimensional.")
    if A.size(1) != B.size(0):
        raise RuntimeError("Incompatible matrix dimensions for multiplication.")
    
    try:
        result = matmul_module.matmul_wrapper(A, B)
        return result[0]
    except RuntimeError as e:
        print(f"Error during CUDA kernel execution: {e}")
        return None


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping example.")
    else:
        A = torch.randn(128, 256, device='cuda', dtype=torch.float32)
        B = torch.randn(256, 512, device='cuda', dtype=torch.float32)

        try:
            C = matmul_cuda(A, B)
            print("CUDA Matrix Multiplication Successful")
            print(f"Result shape: {C.shape}")
        except RuntimeError as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
