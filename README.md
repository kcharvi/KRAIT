# KRAIT - Kernel Review, Analysis, and Intelligent Tuning

An advanced GPU kernel analysis and optimization platform that combines static code analysis with AI-powered insights to help developers write better, more efficient GPU kernels.

## Features

- **Kernel Code Generation**: Generate optimized GPU kernels for various backends (CUDA, Triton, OpenCL)
- **Intelligent Analysis**: AI-powered code review with bounds checking, memory safety, type safety, and synchronization analysis
- **Performance Metrics**: Comprehensive performance analysis with FLOPs calculation, memory usage, and runtime estimation
- **Multi-Backend Support**: Support for CUDA, Triton, and OpenCL backends
- **Hardware-Aware Optimization**: Tailored suggestions for different GPU architectures (NVIDIA A100/H100, AMD MI300X)

## Architecture

- **Frontend**: Next.js with TypeScript and Tailwind CSS
- **Backend**: FastAPI with Python
- **AI Integration**: Google Gemini for advanced code analysis
- **Analysis Engine**: Modular checker system for different aspects of kernel correctness

## Quick Start

1. Clone the repository
2. Install dependencies:
   ```bash
   # Frontend
   cd frontend && npm install
   
   # Backend
   cd backend && pip install -r requirements.txt
   ```
3. Set up environment variables (see `.env.example`)
4. Run the application:
   ```bash
   # Frontend
   cd frontend && npm run dev
   
   # Backend
   cd backend && python -m uvicorn app.main:app --reload
   ```

