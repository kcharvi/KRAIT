# COMPILATION REQUEST
    # Hardware: NVIDIA T4
    # Backend: PYTORCH_CUDA_EXTENSION
    # Timestamp: 1758481016
    # Type: compile_only

    import torch
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
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return {C};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrixMultiply", &matrixMultiply, "Matrix Multiplication CUDA Kernel");
}
"""

# Load the kernel
try:
    custom_kernel = load_inline(
        name="my_cuda_kernel", 
        source=kernel_code, 
        verbose=True,
        with_cuda=True,
        extra_cflags=['-std=c++14']
    )
except Exception as e:
    print(f"Error loading CUDA kernel: {e}")
    exit(1)


def my_matmul(A, B):
    """
    Wrapper function for the custom CUDA kernel.
    """
    try:
        if not A.is_cuda or not B.is_cuda:
            raise RuntimeError("Inputs must be CUDA tensors.")
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            raise RuntimeError("Inputs must be float32 tensors.")
        if A.dim() != 2 or B.dim() != 2:
            raise RuntimeError("Inputs must be 2D tensors.")
        if A.shape[1] != B.shape[0]:
            raise RuntimeError("Incompatible matrix dimensions.")

        result = custom_kernel.matrixMultiply(A, B)
        return result[0]
    except Exception as e:
        print(f"Error in CUDA kernel execution: {e}")
        return None


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.randn(1024, 512, device=device, dtype=torch.float32)
    B = torch.randn(512, 2048, device=device, dtype=torch.float32)

    C = my_matmul(A, B)

    if C is not None:
        print("Matrix multiplication successful.")
        #print(C) # Uncomment to print the result (large output)
        print(C.shape)