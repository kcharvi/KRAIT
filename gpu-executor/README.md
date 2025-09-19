# KRAIT GPU Executor

This directory contains the GPU execution infrastructure for KRAIT, enabling real-time CUDA kernel execution and performance analysis using Google Colab.

## Directory Structure

```
gpu-executor/
├── kernels/               # Directory for incoming kernel files (.cu)
├── results/               # Directory for execution results (.json)
├── executor.ipynb         # Main Colab notebook for kernel execution
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

1. **Kernel Submission**: The KRAIT backend uploads CUDA kernel code to `kernels/` directory in the GitHub repository
2. **Colab Monitoring**: The `executor.ipynb` notebook runs on Google Colab and monitors the `kernels/` directory for new files
3. **Kernel Execution**: When a new `.cu` file is detected, Colab compiles and executes it with profiling
4. **Results Collection**: Execution metrics are saved to `results/` directory as JSON files
5. **Cleanup**: Processed kernel files are removed, and results are retrieved by the backend

## Setup Instructions

### 1. Configure Repository URL

Edit `executor.ipynb` and update the `REPO_URL` variable with your actual GitHub repository URL:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/krait.git"
```

### 2. Set Up Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Open the `executor.ipynb` notebook from your repository
3. Enable GPU in Runtime settings:
   - Runtime → Change runtime type → Hardware accelerator → GPU
4. Run all cells to start monitoring

### 3. Backend Integration

The KRAIT backend automatically handles:
- Uploading kernel files to `kernels/` directory
- Monitoring `results/` directory for completion
- Retrieving and parsing execution metrics
- Cleaning up result files after processing

## Supported Metrics

The GPU executor collects the following performance metrics:

- **Execution Time**: Kernel runtime in milliseconds
- **GPU Utilization**: Percentage of GPU usage during execution
- **Memory Usage**: GPU memory consumption in MB
- **Throughput**: Operations per second
- **Hardware Info**: GPU type and specifications
- **Status**: Execution status (completed, failed, etc.)

## Error Handling

The executor includes robust error handling for:
- Compilation failures
- Profiling errors
- Network connectivity issues
- File system errors

When errors occur, basic metrics are still provided with appropriate error messages.

## Security Considerations

- Kernel files are automatically cleaned up after processing
- Result files are removed after being retrieved by the backend
- No sensitive data is stored permanently in the repository
- All execution happens in isolated Colab environments

## Troubleshooting

### Common Issues

1. **Colab GPU Not Available**
   - Ensure GPU is enabled in Runtime settings
   - Check if you've exceeded free GPU quota

2. **Repository Access Issues**
   - Verify the repository URL is correct
   - Ensure the repository is public or Colab has access

3. **Compilation Failures**
   - Check CUDA kernel syntax
   - Verify all required headers are included

4. **Profiling Errors**
   - Some kernels may not be compatible with nsys profiling
   - Basic metrics will still be provided

### Monitoring

The Colab notebook provides real-time feedback:
- Dots (.) indicate active monitoring
- Detailed logs for each kernel execution
- Error messages for failed executions
- Success confirmations with metrics

## Development

To extend the GPU executor:

1. **Add New Metrics**: Modify the `extract_metrics()` function
2. **Support New Backends**: Add compilation commands for other GPU backends
3. **Improve Profiling**: Enhance the nsys profile parsing logic
4. **Add Validation**: Include kernel validation before execution

## Dependencies

See `requirements.txt` for the complete list of Python dependencies required for the Colab notebook.
