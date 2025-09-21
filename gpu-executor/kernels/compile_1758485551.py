# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758485551
    # Type: compile_only

    import torch
from torch.utils.cpp_extension import load_inline

# C++ wrapper code (host code)
cpp_code = """
#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K);

// Define BLOCK_SIZE for the C++ code
#define BLOCK_SIZE 32

// C++ wrapper function to call the kernel
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

# CUDA kernel code (device code)
cuda_code = """
#include <torch/extension.h>

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

# Load the CUDA kernel using load_inline
try:
    matmul_module = load_inline(
        name="matmul_cuda",
        cpp_sources=[cpp_code],
        cuda_sources=[cuda_code],
        verbose=True
    )
    matmul_wrapper = matmul_module.matmul_wrapper
    print("CUDA kernel loaded successfully.")
except Exception as e:
    print(f"Error loading CUDA kernel: {e}")
    exit(1)


def pytorch_matmul(A, B):
    """
    Wrapper function for the CUDA kernel. Handles tensor inputs and outputs.
    """
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        
        A = A.cuda()
        B = B.cuda()
        
        result = matmul_wrapper(A, B)
        return result[0].cpu() # Return result to CPU
    except Exception as e:
        print(f"Error during CUDA matrix multiplication: {e}")
        return None


# Example usage
if __name__ == "__main__":
    A = torch.randn(128, 256, dtype=torch.float32)
    B = torch.randn(256, 512, dtype=torch.float32)

    C_pytorch = pytorch_matmul(A, B)

    if C_pytorch is not None:
        print("Result shape:", C_pytorch.shape)
        #Further verification or usage of C_pytorch can be added here.
    else:
        print("Matrix multiplication failed.")
