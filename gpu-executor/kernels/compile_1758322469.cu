// COMPILATION REQUEST
// Hardware: NVIDIA H100
// Backend: CUDA
// Timestamp: 1758322469
// Type: compile_only

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16

// Kernel function for 2D convolution
__global__ void conv2d_kernel(const float* input, const float* weights, const float* bias, float* output,
                              int in_channels, int out_channels, int in_height, int in_width, int kernel_size) {
    int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = blockIdx.y * blockDim.y + threadIdx.y;
    int out_width = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_channel < out_channels && out_height < in_height && out_width < in_width) {
        float sum = bias[out_channel];
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int in_height_index = out_height + i - kernel_size / 2;
                    int in_width_index = out_width + j - kernel_size / 2;

                    if (in_height_index >= 0 && in_height_index < in_height &&
                        in_width_index >= 0 && in_width_index < in_width) {
                        int input_index = in_channel * in_height * in_width + in_height_index * in_width + in_width_index;
                        int weight_index = out_channel * in_channels * kernel_size * kernel_size +
                                           in_channel * kernel_size * kernel_size + i * kernel_size + j;
                        sum += input[input_index] * weights[weight_index];
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
    size_t weights_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    size_t output_size = out_channels * in_height * in_width * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_weights = (float*)malloc(weights_size);
    float* h_bias = (float*)malloc(bias_size);
    float* h_output = (float*)malloc(output_size);

    // Initialize input, weights, and bias (replace with your actual data)
    for (int i = 0; i < input_size / sizeof(float); ++i) h_input[i] = i;
    for (int i = 0; i < weights_size / sizeof(float); ++i) h_weights[i] = i;
    for (int i = 0; i < bias_size / sizeof(float); ++i) h_bias[i] = i;

    // Allocate memory on device
    float* d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_bias, bias_size);
    cudaMalloc(&d_output, output_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);


    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((out_channels + blockDim.x - 1) / blockDim.x, (in_height + blockDim.y - 1) / blockDim.y, (in_width + blockDim.z - 1) / blockDim.z);
    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_weights, d_bias, d_output, in_channels, out_channels, in_height, in_width, kernel_size);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Print results (optional - for verification)
    //printf("Output:\n");
    //for (int i = 0; i < 10; ++i) printf("%f ", h_output[i]);
    //printf("\n");

    // Clean up memory
    free(h_input);
    free(h_weights);
    free(h_bias);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}