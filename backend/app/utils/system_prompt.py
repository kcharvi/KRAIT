"""
KRAIT - Kernel Review, Analysis, and Intelligent Tuning
System prompt for kernel generation
"""

def create_kernel_generation_prompt(backend: str, hardware: str, code: str, user_prompt: str, problem_name: str = None) -> str:
    """
    Create a comprehensive system prompt for kernel generation.
    
    Args:
        backend: The backend framework (Triton, CUDA, OpenCL)
        hardware: The target hardware (AMD MI300X, NVIDIA A100, etc.)
        code: The base code to optimize
        user_prompt: User's specific requirements
        problem_name: Name of the problem type (optional)
    
    Returns:
        Formatted system prompt string
    """
    
    # Determine the appropriate programming language based on backend
    language_map = {
        "CUDA": "cuda",
        "PYTORCH_CUDA_EXTENSION": "python",
        "Triton": "python", 
        "OpenCL": "c",
        "C++": "cpp",
        "C": "c"
    }
    language = language_map.get(backend, "python")
    
    if backend.upper() == "CUDA":
        prompt = f"""Generate a complete CUDA program for {hardware} with the following requirements:

Base code:
```cuda
{code}
```

Requirements: {user_prompt}

IMPORTANT: Generate a COMPLETE CUDA program that includes:
1. All necessary #include statements (#include <cuda_runtime.h>, #include <stdio.h>, etc.)
2. All #define statements at the TOP of the file
3. Complete kernel functions with proper __global__ declarations
4. A main() function that:
   - Allocates memory on host and device
   - Launches the kernel with proper grid/block dimensions
   - Copies results back to host
   - Prints results and cleans up memory
5. Proper error checking for CUDA operations

Return ONLY the complete CUDA program in ```cuda``` blocks:"""
    elif backend.upper() == "PYTORCH_CUDA_EXTENSION":
        prompt = f"""Generate a Python script that defines a custom CUDA kernel for {hardware} using PyTorch's `load_inline` function.

Base code:
```python
{code}
```

Requirements: {user_prompt}

The script should include:
1. The CUDA C++ kernel code (e.g., `matmul_kernel`) as a multi-line string
2. Python code to load this kernel using `torch.utils.cpp_extension.load_inline`
3. A Python function that wraps the loaded kernel and handles tensor inputs/outputs
4. Example usage demonstrating how to call the Python wrapper with PyTorch tensors

IMPORTANT: Generate a COMPLETE Python script that includes:
- All necessary imports (torch, torch.utils.cpp_extension, etc.)
- CUDA C++ kernel code embedded as a string
- Python wrapper function for the kernel
- Example usage with proper tensor creation and function calls
- Error handling for CUDA operations

Return ONLY the complete Python script in ```python``` blocks, with the CUDA C++ kernel embedded within it in a string."""
    else:
        prompt = f"""Generate optimized {backend} kernel for {hardware}.

Base code:
```{language}
{code}
```

Requirements: {user_prompt}

Return ONLY the optimized code in ```{language}``` blocks:"""
    
    if problem_name:
        prompt += f"\nProblem: {problem_name}"
    
    if backend.upper() == "CUDA":
        prompt += f"""

Generate a complete, compilable CUDA program optimized for {hardware}. Include all necessary includes, defines, kernel functions, and main() function. Return ONLY the complete code in ```cuda``` blocks."""
    elif backend.upper() == "PYTORCH_CUDA_EXTENSION":
        prompt = f"""Generate a Python script that defines a custom CUDA kernel for {hardware} using PyTorch's `load_inline` function.

    Base code:
    ```python
    {code}
    ```

    Requirements: {user_prompt}

    The script should include:
    1. All necessary imports (torch, torch.utils.cpp_extension, etc.)
    2. CUDA C++ kernel code with proper structure:
    - All #define statements at the top
    - Proper includes: #include <torch/extension.h>
    - CUDA kernel function with __global__ keyword
    - C++ wrapper function that handles tensor operations
    - Python module binding with PYBIND11_MODULE
    3. Python code to load this kernel using `torch.utils.cpp_extension.load_inline`
    4. A Python function that wraps the loaded kernel and handles tensor inputs/outputs
    5. Example usage demonstrating how to call the Python wrapper with PyTorch tensors
    6. Proper error handling for CUDA operations

    IMPORTANT: Generate a COMPLETE Python script that includes:
    - All necessary imports
    - CUDA C++ kernel code embedded as a string with proper structure
    - Python wrapper function for the kernel
    - Example usage with proper tensor creation and function calls
    - Error handling for CUDA operations
    - Proper #define placement at the top of the CUDA code
    - PYBIND11_MODULE for Python binding

    CUDA Code Structure:
    ```cpp
    #include <torch/extension.h>

    #define BLOCK_SIZE 32  // Define constants at the top

    __global__ void your_kernel_name(/* parameters */) {{
        // Kernel implementation
    }}

    std::vector<torch::Tensor> your_wrapper_name(torch::Tensor A, torch::Tensor B) {{
        // Get dimensions
        int M = A.size(0);
        int K = A.size(1);
        int N = B.size(1);
        
        // Create output tensor
        auto C = torch::zeros({{M, N}}, A.options());
        
        // Configure grid and block dimensions
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Launch kernel
        your_kernel_name<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
        
        cudaDeviceSynchronize();
        return {{C}};
    }}

    // Python module binding - REQUIRED
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
        m.def("your_wrapper_name", &your_wrapper_name, "Description");
    }}
    ```

    Return ONLY the complete Python script in ```python``` blocks, with the CUDA C++ kernel embedded within it in a string."""
    else:
        prompt += f"""

Optimize for {hardware} using {backend} best practices. Return ONLY code in ```{language}``` blocks."""
    
    return prompt
