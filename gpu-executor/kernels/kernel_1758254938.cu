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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for matrix multiplication C = A x B.
    A: (M, K) matrix
    B: (K, N) matrix
    C: (M, N) matrix
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block indices
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B blocks
        a_block = tl.load(A + (block_m, k), (BLOCK_SIZE_M, BLOCK_SIZE_K), mask=None, other=-1.0)
        b_block = tl.load(B + (k, block_n), (BLOCK_SIZE_K, BLOCK_SIZE_N), mask=None, other=-1.0)

        # Perform matrix multiplication
        acc += tl.dot(a_block, b_block)

    # Write results to output matrix
    c_ptr = C + (block_m, block_n)
    tl.store(c_ptr, acc, mask=None)


@triton.jit
def matmul_kernel_32x32(
    A,
    B,
    C,
    M,
    N,
    K,
):
    matmul_kernel[128, 128](A, B, C, M, N, K, 32, 32, 32)