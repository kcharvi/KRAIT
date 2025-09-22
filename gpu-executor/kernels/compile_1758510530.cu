// COMPILATION REQUEST
    // Hardware: NVIDIA T4
    // Backend: CUDA
    // Timestamp: 1758510530
    // Type: compile_only

    #include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

//Optimized Kernel for Matrix Multiplication
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int widthA, int widthB, int heightA) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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
    float *h_A, *h_B, *h_C;
    cudaMallocHost((void **)&h_A, heightA * widthA * sizeof(float));
    cudaMallocHost((void **)&h_B, widthA * widthB * sizeof(float));
    cudaMallocHost((void **)&h_C, heightA * widthB * sizeof(float));


    // Initialize host matrices (Example: fill with random values)
    for (int i = 0; i < heightA * widthA; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < widthA * widthB; ++i) h_B[i] = (float)rand() / RAND_MAX;


    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, heightA * widthA * sizeof(float));
    cudaMalloc((void **)&d_B, widthA * widthB * sizeof(float));
    cudaMalloc((void **)&d_C, heightA * widthB * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);


    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((widthB + BLOCK_SIZE - 1) / BLOCK_SIZE, (heightA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, widthA, widthB, heightA);


    // Copy results from device to host
    cudaMemcpy(h_C, d_C, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);


    //Print a small portion of the result for verification.  Avoid printing the entire matrix for large sizes.
    printf("Result (a small portion):\n");
    for(int i=0; i<10; ++i){
        for(int j=0; j<10; ++j){
            printf("%f ", h_C[i*widthB + j]);
        }
        printf("\n");
    }


    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    //Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}