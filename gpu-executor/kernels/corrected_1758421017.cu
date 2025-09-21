#define BLOCK_SIZE 16
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void conv2d_kernel(const float *input, const float *weight, const float *bias, float *output,
                             int in_channels, int out_channels, int in_height, int in_width, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < in_width && y < in_height && out_channel < out_channels) {
        float sum = bias[out_channel];
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int input_x = x + j - kernel_size / 2;
                    int input_y = y + i - kernel_size / 2;

                    if (input_x >= 0 && input_x < in_width && input_y >= 0 && input_y < in_height) {
                        sum += input[input_y * in_width * in_channels + input_x * in_channels + c] *
                               weight[out_channel * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + i * kernel_size + j];
                    }
                }
            }
        }
        output[y * in_width * out_channels + x * out_channels + out_channel] = sum;
    }
}


int main() {
    // Example parameters
    int in_channels = 3;
    int out_channels = 16;
    int in_height = 224;
    int in_width = 224;
    int kernel_size = 3;

    // Input/output sizes
    size_t input_size = in_height * in_width * in_channels * sizeof(float);
    size_t output_size = in_height * in_width * out_channels * sizeof(float);
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);


    // Allocate host memory
    float *h_input = (float *)malloc(input_size);
    float *h_output = (float *)malloc(output_size);
    float *h_weight = (float *)malloc(weight_size);
    float *h_bias = (float *)malloc(bias_size);

    // Initialize host memory (replace with your actual data)
    for (size_t i = 0; i < input_size / sizeof(float); ++i) h_input[i] = i;
    for (size_t i = 0; i < weight_size / sizeof(float); ++i) h_weight[i] = i;
    for (size_t i = 0; i < bias_size / sizeof(float); ++i) h_bias[i] = i;


    // Allocate device memory
    float *d_input, *d_output, *d_weight, *d_bias;
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, output_size);
    cudaMalloc((void **)&d_weight, weight_size);
    cudaMalloc((void **)&d_bias, bias_size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((in_width + blockDim.x - 1) / blockDim.x, (in_height + blockDim.y - 1) / blockDim.y, (out_channels + blockDim.z -1) / blockDim.z);

    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_weight, d_bias, d_output, in_channels, out_channels, in_height, in_width, kernel_size);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);


    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print some results (optional)
    printf("First 10 output values: ");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Free memory
    free(h_input);
    free(h_output);
    free(h_weight);
    free(h_bias);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_bias);

    return 0;
}