// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758336553
// Type: compile_only

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

// Kernel function for 2D convolution
__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int in_channels, int out_channels, int in_height, int in_width, int kernel_size) {
    int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = blockIdx.y * blockDim.y + threadIdx.y;
    int out_width = blockIdx.z * blockDim.z + threadIdx.z;


    if (out_channel < out_channels && out_height < in_height && out_width < in_width) {
        float sum = bias[out_channel];
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int in_h = out_height + k_h - kernel_size / 2;
                    int in_w = out_width + k_w - kernel_size / 2;

                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                        int input_index = in_channel * in_height * in_width + in_h * in_width + in_w;
                        int weight_index = out_channel * in_channels * kernel_size * kernel_size + in_channel * kernel_size * kernel_size + k_h * kernel_size + k_w;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
        int output_index = out_channel * in_height * in_width + out_height * in_width + out_width;
        output[output_index] = sum;
    }
}


int main() {
    // Example parameters
    int in_channels = 3;
    int out_channels = 16;
    int in_height = 256;
    int in_width = 256;
    int kernel_size = 3;

    // Allocate memory on host
    size_t input_size = in_channels * in_height * in_width * sizeof(float);
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    size_t output_size = out_channels * in_height * in_width * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_weight = (float*)malloc(weight_size);
    float* h_bias = (float*)malloc(bias_size);
    float* h_output = (float*)malloc(output_size);

    // Initialize data on host (replace with your actual data)
    for (int i = 0; i < input_size / sizeof(float); ++i) h_input[i] = i;
    for (int i = 0; i < weight_size / sizeof(float); ++i) h_weight[i] = i;
    for (int i = 0; i < bias_size / sizeof(float); ++i) h_bias[i] = i;


    // Allocate memory on device
    float* d_input;
    float* d_weight;
    float* d_bias;
    float* d_output;

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weight, weight_size);
    cudaMalloc((void**)&d_bias, bias_size);
    cudaMalloc((void**)&d_output, output_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((out_channels + blockDim.x - 1) / blockDim.x, (in_height + blockDim.y - 1) / blockDim.y, (in_width + blockDim.z - 1) / blockDim.z);

    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_weight, d_bias, d_output, in_channels, out_channels, in_height, in_width, kernel_size);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Print results (optional)
    //printf("Output:\n");
    //for (int i = 0; i < 10; ++i) printf("%f ", h_output[i]);
    //printf("\n");


    // Clean up memory
    free(h_input);
    free(h_weight);
    free(h_bias);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}