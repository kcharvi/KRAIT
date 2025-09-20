// EXECUTION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758389123
// Type: execute

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rm = tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in rk:
        a_ptr = A + pid_m * BLOCK_SIZE_M + rm + (pid_k * BLOCK_SIZE_K + k) * stride_ak
        b_ptr = B + pid_k * BLOCK_SIZE_K + k + (pid_n * BLOCK_SIZE_N + rn) * stride_bn
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc += a * b

    c_ptr = C + pid_m * BLOCK_SIZE_M + rm + (pid_n * BLOCK_SIZE_N + rn) * stride_cn
    tl.store(c_ptr, acc)


# Example usage (replace with your actual shapes and data)
M = 1024
N = 1024
K = 1024
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

A = torch.randn((M, K)).cuda()
B = torch.randn((K, N)).cuda()
C = torch.zeros((M, N)).cuda()

grid = (
    (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K,
)

matmul_kernel[grid](
    A,
    B,
    C,
    M,
    N,
    K,
    1,
    K,
    1,
    N,
    1,
    N,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
)

print(C)