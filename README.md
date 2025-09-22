# KRAIT - Kernel Review, Analysis, and Intelligent Tuning

An advanced GPU kernel analysis and optimization platform that combines static code analysis with AI-powered insights to help developers write better, more efficient GPU kernels. KRAIT provides comprehensive correctness checking, performance analysis, and intelligent suggestions for CUDA, Triton, and OpenCL kernels.

## 🚀 Features

### Core Analysis Capabilities
- **Correctness Analysis**: Comprehensive bounds checking, memory safety, type safety, and synchronization analysis
- **Performance Analysis**: FLOPs calculation, memory usage estimation, runtime prediction, and efficiency scoring
- **AI-Powered Code Review**: Advanced LLM-based analysis using Google Gemini for complex code patterns
- **Multi-Backend Support**: Full support for CUDA, Triton, and OpenCL kernels
- **Hardware-Aware Optimization**: Tailored suggestions for different GPU architectures (NVIDIA A100/H100, AMD MI300X)

### Advanced Features
- **Kernel Code Generation**: Generate optimized GPU kernels with AI assistance
- **Real-time GPU Execution**: Execute kernels on actual GPU hardware via Google Colab integration
- **Interactive Workbench**: Web-based interface for kernel development and analysis
- **Performance Metrics Dashboard**: Visualize kernel performance across different hardware configurations
- **Automated Code Fixing**: AI-powered suggestions to fix compilation errors and improve performance
- **Batch Analysis**: Analyze multiple kernels simultaneously

### Analysis Modules
- **Bounds Checker**: Detects unsafe memory accesses and suggests proper bounds checking
- **Memory Safety Checker**: Identifies memory leaks, buffer overflows, and unsafe memory patterns
- **Synchronization Checker**: Ensures proper thread synchronization and race condition prevention
- **Type Safety Checker**: Validates type usage and prevents type-related bugs
- **Performance Checker**: Analyzes computational efficiency and memory utilization
- **Vectorization Checker**: Identifies vectorization opportunities
- **Tiling Detector**: Detects and analyzes memory tiling patterns

## 🏗️ Architecture

### Frontend (Next.js + TypeScript)
- **Kernel Workbench**: Interactive code editor with syntax highlighting
- **Critic Panel**: Real-time analysis results and suggestions
- **Metrics Dashboard**: Performance visualization and historical data
- **Chat Interface**: AI-powered assistance for kernel development

### Backend (FastAPI + Python)
- **Analysis Engine**: Modular checker system for kernel correctness
- **LLM Integration**: Google Gemini and Hugging Face model support
- **GPU Executor**: GitHub-Colab integration for real kernel execution
- **RESTful API**: Comprehensive API for all analysis features

### GPU Execution Pipeline
- **GitHub Integration**: Automatic kernel deployment to Google Colab
- **Real-time Monitoring**: Live execution status and result collection
- **Performance Metrics**: Actual GPU utilization and timing data
- **Error Handling**: Comprehensive error detection and recovery

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.11+
- **Git** for version control
- **Google Colab** account (for GPU execution)
- **GitHub** account (for kernel storage and execution)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kcharvi/KRAIT.git
cd KRAIT
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys and configuration
```

**Required Environment Variables:**
```env
# Gemini API (for AI analysis)
GEMINI_API_KEY=your_gemini_api_key

# GitHub Integration (for GPU execution)
GITHUB_TOKEN=your_github_token
GITHUB_OWNER=your_github_username
GITHUB_REPO_NAME=KRAIT

# Optional: Hugging Face (alternative AI provider)
HUGGINGFACE_API_KEY=your_huggingface_token
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp env.local.example .env.local
# Edit .env.local with your backend URL
```

**Frontend Environment Variables:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. GPU Executor Setup (Optional)
```bash
cd gpu-executor

# Install Python dependencies
pip install -r requirements.txt

# Follow SETUP.md for detailed Colab configuration
```

## 🚀 Getting Started

### 1. Start the Backend
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 Usage Guide

### Basic Kernel Analysis
1. **Open the Kernel Workbench** in your browser
2. **Select your backend** (CUDA, Triton, or OpenCL)
3. **Choose target hardware** (NVIDIA T4, A100, H100, etc.)
4. **Write or paste your kernel code**
5. **Click "Analyze"** to run comprehensive analysis
6. **Review results** in the Critic Panel

### AI-Powered Code Generation
1. **Navigate to the Kernel Generation tab**
2. **Describe your problem** (e.g., "matrix multiplication for 1024x1024 matrices")
3. **Select backend and hardware**
4. **Generate optimized kernel** with AI assistance
5. **Analyze and refine** the generated code

### Real GPU Execution
1. **Ensure GitHub integration is configured**
2. **Write your kernel code**
3. **Click "Execute on GPU"** to run on actual hardware
4. **Monitor execution** in real-time
5. **View performance metrics** and results

### Performance Analysis
1. **Access the Metrics Dashboard**
2. **View historical performance data**
3. **Compare kernels across different hardware**
4. **Export data** for further analysis

## 🔧 API Usage

### Analyze a Kernel
```bash
curl -X POST "http://localhost:8000/critic/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_code": "__global__ void matrixMul(float* A, float* B, float* C, int N) { ... }",
    "hardware": "NVIDIA A100",
    "backend": "CUDA"
  }'
```

### Generate a Kernel
```bash
curl -X POST "http://localhost:8000/kernel/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "CUDA",
    "hardware": "NVIDIA A100",
    "code": "// Matrix multiplication kernel",
    "user_prompt": "Optimize for memory bandwidth",
    "problem_name": "Matrix Multiplication"
  }'
```

### Execute on GPU
```bash
curl -X POST "http://localhost:8000/gpu/execute-kernel" \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_code": "__global__ void matrixMul(...) { ... }",
    "hardware": "NVIDIA T4",
    "provider": "github_colab"
  }'
```

## 📊 Analysis Results

### Correctness Analysis
- **Bounds Checking**: Identifies unsafe array accesses
- **Memory Safety**: Detects memory leaks and buffer overflows
- **Synchronization**: Ensures proper thread coordination
- **Type Safety**: Validates type usage and conversions

### Performance Analysis
- **FLOPs Calculation**: Theoretical floating-point operations
- **Memory Usage**: Estimated memory consumption
- **Runtime Prediction**: Expected execution time
- **Efficiency Score**: Overall performance rating (0-100)

### Suggestions
- **Critical**: Fixes for correctness issues
- **High**: Important performance optimizations
- **Medium**: General improvements
- **Low**: Minor optimizations

## 🎯 Supported Hardware

### NVIDIA GPUs
- **Tesla T4**: Entry-level development and testing
- **A100**: High-performance computing
- **H100**: Latest generation with advanced features
- **RTX Series**: Consumer and professional GPUs

### AMD GPUs
- **MI300X**: High-performance data center GPU
- **Radeon Series**: Consumer and professional GPUs

## 🔌 Supported Backends

### CUDA
- **CUDA C++**: Traditional CUDA kernel development
- **CUDA Templates**: Generic programming support
- **Tensor Cores**: Mixed-precision acceleration

### Triton
- **Python-based**: High-level kernel development
- **Automatic optimization**: Compiler-driven optimizations
- **Multi-GPU support**: Distributed computing

### OpenCL
- **Cross-platform**: Multi-vendor GPU support
- **C/C++ kernels**: Traditional OpenCL development
- **Heterogeneous computing**: CPU and GPU integration

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📝 Development

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Quality
```bash
# Backend linting
cd backend
black .
flake8 .

# Frontend linting
cd frontend
npm run lint
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Colab** for GPU execution infrastructure
- **Google Gemini** for AI-powered analysis
- **Hugging Face** for alternative AI models
- **NVIDIA** for CUDA development tools
- **OpenCL** community for cross-platform support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kcharvi/KRAIT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kcharvi/KRAIT/discussions)
- **Documentation**: [Wiki](https://github.com/kcharvi/KRAIT/wiki)

## 🔄 Changelog

### v1.0.0
- Initial release with core analysis capabilities
- CUDA, Triton, and OpenCL support
- AI-powered code generation and analysis
- Real-time GPU execution via Colab integration
- Comprehensive web interface

---

**KRAIT** - Making GPU kernel development smarter, safer, and more efficient! 🚀

