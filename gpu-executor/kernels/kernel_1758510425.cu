// EXECUTION REQUEST
    // Hardware: NVIDIA T4
    // Backend: CUDA
    // Timestamp: 1758510425
    // Type: execute

    #define BLOCK_SIZE 16
    #include <cuda_runtime.h>
#include <stdio.h>


__global__ void conv2d_kernel(const float *input, const float *weight, const float *bias, float *output,
                              int in_channels, int out_channels, int in_height, int in_width, int kernel_size) {
    int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_channel >= out_channels || out_height >= in_height - kernel_size + 1) return;

    int out_width;
    for (out_width = 0; out_width < in_width - kernel_size + 1; ++out_width) {
        float sum = bias[out_channel];
        for (int k = 0; k < kernel_size; ++k) {
            for (int l = 0; l < kernel_size; ++l) {
                for (int c = 0; c < in_channels; ++c) {
                    int input_index = c * in_height * in_width + (out_height + k) * in_width + (out_width + l);
                    int weight_index = out_channel * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + k * kernel_size + l;
                    sum += input[input_index] * weight[weight_index];
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
    int in_height = 224;
    int in_width = 224;
    int kernel_size = 3;

    //Input, weights, bias and output sizes
    size_t input_size = in_channels * in_height * in_width * sizeof(float);
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    size_t output_size = out_channels * in_height * in_width * sizeof(float);


    // Host memory allocation
    float *h_input = (float *)malloc(input_size);
    float *h_weight = (float *)malloc(weight_size);
    float *h_bias = (float *)malloc(bias_size);
    float *h_output = (float *)malloc(output_size);

    // Initialize host memory (replace with your actual data)
    for (size_t i = 0; i < input_size / sizeof(float); ++i) h_input[i] = i;
    for (size_t i = 0; i < weight_size / sizeof(float); ++i) h_weight[i] = i;
    for (size_t i = 0; i < bias_size / sizeof(float); ++i) h_bias[i] = i;


    // Device memory allocation
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_weight, weight_size);
    cudaMalloc((void **)&d_bias, bias_size);
    cudaMalloc((void **)&d_output, output_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((out_channels + blockDim.x - 1) / blockDim.x, (in_height + blockDim.y -1) / blockDim.y);

    // Kernel launch
    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_weight, d_bias, d_output, in_channels, out_channels, in_height, in_width, kernel_size);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    //Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print results (optional - for verification)
    //printf("First 10 output values: ");
    //for (int i = 0; i < 10; ++i) printf("%f ", h_output[i]);
    //printf("\n");

    // Free memory
    free(h_input);
    free(h_weight);
    free(h_bias);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}