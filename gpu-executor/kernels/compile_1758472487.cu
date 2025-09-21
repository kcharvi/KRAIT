// COMPILATION REQUEST
// Hardware: NVIDIA T4
// Backend: PYTORCH_CUDA_EXTENSION
// Timestamp: 1758472487
// Type: compile_only

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for matrix multiplication
matmul_kernel = """
#include <torch/extension.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int m, int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n) {
    float sum = 0.0f;
    for (int l = 0; l < k; ++l) {
      sum += A[i * k + l] * B[l * n + j];
    }
    C[i * n + j] = sum;
  }
}

TORCH_EXTENSION_NO_CUDA_REGISTRATION()

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", [](const at::Tensor& A, const at::Tensor& B) {
    // Check for CPU tensors. If they are on the CPU, move them to CUDA.
    at::Tensor A_cuda = A.is_cuda() ? A : A.cuda();
    at::Tensor B_cuda = B.is_cuda() ? B : B.cuda();

    int m = A_cuda.size(0);
    int n = B_cuda.size(1);
    int k = A_cuda.size(1);

    // Check dimensions for compatibility.
    if (A_cuda.size(1) != B_cuda.size(0)) {
      throw std::runtime_error("Matrix dimensions are incompatible for multiplication.");
    }


    // Allocate output tensor on CUDA.
    at::Tensor C_cuda = at::empty({m, n}, A_cuda.options());

    // Determine block and grid dimensions.  Adjust as needed for optimal performance on your hardware.
    dim3 blockDim(32, 32); // Adjust block size for optimal performance
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the kernel.
    matmul_kernel<<<gridDim, blockDim>>>(A_cuda.data_ptr<float>(), B_cuda.data_ptr<float>(), C_cuda.data_ptr<float>(), m, n, k);

    // Synchronize to ensure kernel completion.
    cudaDeviceSynchronize();
    
    // Check for CUDA errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    // Return the result.  If the input was on the CPU, return the result on the CPU.
    return A.is_cuda() ? C_cuda : C_cuda.cpu();
  });
}
"""

# Load the CUDA kernel
matmul_module = load_inline(
    name="matmul_cuda",
    cuda_sources=[matmul_kernel],
    verbose=True,
)


def matmul_wrapper(A, B):
    """
    Wrapper function for the CUDA matrix multiplication kernel.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    try:
        result = matmul_module.matmul(A, B)
        return result
    except RuntimeError as e:
        print(f"Error during CUDA operation: {e}")
        return None


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(128, 1024, device=device)
    B = torch.randn(1024, 256, device=device)

    C = matmul_wrapper(A, B)
    if C is not None:
        print("Result shape:", C.shape)
        #Further processing or verification can be added here.