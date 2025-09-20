// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758389561
// Type: compile_only

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 256

//Structure to hold reduction results
struct ReductionResult {
    float sum;
    float mean;
    float max;
    float min;
};


__global__ void reductionKernel(const float* input, float* output, int n) {
    __shared__ float shared_data[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    float max_val = -FLT_MAX;
    float min_val = FLT_MAX;

    //Load data into shared memory.  Handle cases where n < blockDim.x
    if (i < n) {
        sum = input[i];
        max_val = input[i];
        min_val = input[i];
    } else {
        sum = 0.0f;
        max_val = -FLT_MAX;
        min_val = FLT_MAX;
    }

    shared_data[tid] = sum;
    __syncthreads();


    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
            max_val = fmaxf(max_val, shared_data[tid + s]);
            min_val = fminf(min_val, shared_data[tid + s]);

        }
        __syncthreads();
    }

    //Write results to global memory for final reduction
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
        //Store max and min separately as they can't be efficiently reduced in parallel in this way.
        //A second kernel pass is an alternative, but significantly increases complexity for marginal gain.
    }
}


int main() {
    int n = 1024 * 1024 * 64; // Adjust array size as needed.
    float *h_data, *d_data, *h_output, *d_output;

    // Allocate host memory
    h_data = (float*)malloc(n * sizeof(float));
    h_output = (float*)malloc( (n + BLOCK_SIZE -1)/BLOCK_SIZE * sizeof(float));

    // Initialize host data (replace with your actual data)
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_data, n * sizeof(float));
    cudaMalloc((void**)&d_output, ((n + BLOCK_SIZE -1)/BLOCK_SIZE) * sizeof(float));


    // Copy data from host to device
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reductionKernel<<<blocks, BLOCK_SIZE>>>(d_data, d_output, n);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, ((n + BLOCK_SIZE - 1)/BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);


    //Final reduction on host (for sum)
    float totalSum = 0;
    for(int i = 0; i < blocks; ++i){
        totalSum += h_output[i];
    }

    float totalMean = totalSum / n;

    //Find max and min from initial data.  Alternative is a second kernel pass.
    float totalMax = h_data[0];
    float totalMin = h_data[0];
    for(int i = 1; i < n; ++i){
        totalMax = fmaxf(totalMax, h_data[i]);
        totalMin = fminf(totalMin, h_data[i]);
    }



    printf("Sum: %f\n", totalSum);
    printf("Mean: %f\n", totalMean);
    printf("Max: %f\n", totalMax);
    printf("Min: %f\n", totalMin);

    // Free memory
    free(h_data);
    free(h_output);
    cudaFree(d_data);
    cudaFree(d_output);

    cudaDeviceReset(); //Clean up CUDA resources

    return 0;
}