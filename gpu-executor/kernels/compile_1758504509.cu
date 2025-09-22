// COMPILATION REQUEST
    // Hardware: NVIDIA T4
    // Backend: CUDA
    // Timestamp: 1758504509
    // Type: compile_only

    #include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int widthA, int widthB) {
    // Thread ID
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Shared memory for A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // Iterate over tiles
    for (int k = 0; k < widthA; k += BLOCK_SIZE) {
        // Load data into shared memory
        As[ty][tx] = A[row * widthA + k + tx];
        Bs[ty][tx] = B[(k + ty) * widthB + col];

        __syncthreads();

        // Perform matrix multiplication within tile
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    C[row * widthB + col] = sum;
}


int main() {
    int widthA = 1024;
    int widthB = 1024;
    int widthC = widthB;

    // Allocate host memory
    float *h_A = (float *)malloc(widthA * widthA * sizeof(float));
    float *h_B = (float *)malloc(widthA * widthB * sizeof(float));
    float *h_C = (float *)malloc(widthA * widthC * sizeof(float));

    // Initialize host matrices (example: fill with 1.0f)
    for (int i = 0; i < widthA * widthA; i++) h_A[i] = 1.0f;
    for (int i = 0; i < widthA * widthB; i++) h_B[i] = 1.0f;


    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, widthA * widthA * sizeof(float));
    cudaMalloc((void **)&d_B, widthA * widthB * sizeof(float));
    cudaMalloc((void **)&d_C, widthA * widthC * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, widthA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((widthB + BLOCK_SIZE - 1) / BLOCK_SIZE, (widthA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, widthA, widthB);

    // Copy results from device to host
    cudaMemcpy(h_C, d_C, widthA * widthC * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results (a small portion for demonstration)
    printf("Result (part of C):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_C[i * widthC + j]);
        }
        printf("\n");
    }


    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize(); //added for proper error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}