# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758476224
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
        source=kernel_code,
        verbose=True,
        with_cuda=True,
        extra_cflags=['-O3', '-std=c++14']
    )
except RuntimeError as e:
    print(f"Error loading CUDA kernel: {e}")
    exit(1)


def custom_matmul(A, B):
    """
    Python wrapper for the custom CUDA matrix multiplication kernel.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if A.device != torch.device("cuda") or B.device != torch.device("cuda"):
        raise RuntimeError("Input tensors must be on CUDA device.")
    
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise RuntimeError("Input tensors must be of type torch.float32")

    try:
        result = matmul_module.matmul_wrapper(A, B)
        return result[0]  #Return the first tensor from the vector
    except RuntimeError as e:
        print(f"Error during CUDA kernel execution: {e}")
        exit(1)


# Example usage
if __name__ == "__main__":
    try:
        device = torch.device("cuda")
        A = torch.randn(1024, 512, device=device, dtype=torch.float32)
        B = torch.randn(512, 2048, device=device, dtype=torch.float32)

        C = custom_matmul(A, B)
        print("Result shape:", C.shape)
        #Further verification or usage of C can be added here.

    except RuntimeError as e:
        print(f"An error occured: {e}")