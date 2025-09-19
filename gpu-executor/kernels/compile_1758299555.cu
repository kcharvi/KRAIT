// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758299555
// Type: compile_only

#include <cuda_runtime.h>
#include <stdio.h>

// Define block size (adjust for optimal performance on H100)
#define BLOCK_SIZE 32

// Kernel function for matrix multiplication
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int widthA, int widthB, int heightA) {
    // Thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Check boundaries
    if (row < heightA && col < widthB) {
        float sum = 0.0f;
        for (int k = 0; k < widthA; ++k) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}


int main() {
    // Matrix dimensions
    int heightA = 1024;
    int widthA = 1024;
    int widthB = 1024;

    // Allocate host memory
    float *h_A = (float *)malloc(heightA * widthA * sizeof(float));
    float *h_B = (float *)malloc(widthA * widthB * sizeof(float));
    float *h_C = (float *)malloc(heightA * widthB * sizeof(float));

    // Initialize host matrices (example)
    for (int i = 0; i < heightA * widthA; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < widthA * widthB; ++i) h_B[i] = 2.0f;


    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, heightA * widthA * sizeof(float));
    cudaMalloc((void **)&d_B, widthA * widthB * sizeof(float));
    cudaMalloc((void **)&d_C, heightA * widthB * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((widthB + blockDim.x - 1) / blockDim.x, (heightA + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, widthA, widthB, heightA);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy results from device to host
    cudaMemcpy(h_C, d_C, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results (optional - for small matrices only)
    //for (int i = 0; i < heightA; ++i) {
    //    for (int j = 0; j < widthB; ++j) {
    //        printf("%f ", h_C[i * widthB + j]);
    //    }
    //    printf("\n");
    //}

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}