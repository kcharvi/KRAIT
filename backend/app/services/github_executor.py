"""
GitHub-based GPU executor service for KRAIT.

This service integrates with Google Colab via GitHub repository monitoring
to execute CUDA kernels and collect real-time performance metrics.
"""

import requests
import json
import time
import asyncio
from typing import Dict, Any, Optional
from github import Github
from github.GithubException import GithubException
import logging

logger = logging.getLogger(__name__)


class GitHubExecutor:
    """GitHub-based GPU executor using Colab integration."""
    
    def __init__(self, github_token: str, repo_name: str, owner: str):
        """
        Initialize the GitHub executor.
        
        Args:
            github_token: GitHub personal access token
            repo_name: Name of the GitHub repository
            owner: GitHub username or organization name
        """
        self.github = Github(github_token)
        self.repo = self.github.get_repo(f"{owner}/{repo_name}")
        self.owner = owner
        self.repo_name = repo_name
        self.kernels_path = "gpu-executor/kernels"
        self.results_path = "gpu-executor/results"
        
        logger.info(f"GitHub executor initialized for {owner}/{repo_name}")
    
    def ensure_results_directory(self) -> bool:
        """
        Ensure the results directory exists in the GitHub repository.
        Creates the directory by uploading a .gitkeep file if it doesn't exist.
        
        Returns:
            True if directory exists or was created successfully, False otherwise
        """
        try:
            # Check if results directory exists
            try:
                self.repo.get_contents(self.results_path, ref="main")
                logger.info(f"Results directory already exists: {self.results_path}")
                return True
            except GithubException as e:
                if e.status == 404:
                    # Directory doesn't exist, create it
                    logger.info(f"Creating results directory: {self.results_path}")
                    
                    # Create .gitkeep file to establish the directory
                    gitkeep_content = f"# {self.results_path} directory\n# This file ensures the directory exists for KRAIT GPU execution results"
                    gitkeep_path = f"{self.results_path}/.gitkeep"
                    
                    self.repo.create_file(
                        gitkeep_path,
                        f"Create {self.results_path} directory",
                        gitkeep_content,
                        branch="main"
                    )
                    
                    logger.info(f"Results directory created successfully: {self.results_path}")
                    return True
                else:
                    logger.error(f"Error checking results directory: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to ensure results directory: {e}")
            return False
    
    def clean_old_kernel_files(self) -> bool:
        """
        Clean up old kernel files from the kernels directory.
        Removes files older than 1 hour to prevent accumulation.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            logger.info("Cleaning up old kernel files...")
            
            # Get all files in kernels directory
            kernels_contents = self.repo.get_contents(self.kernels_path, ref="main")
            current_time = int(time.time())
            cleaned_count = 0
            
            for file_info in kernels_contents:
                filename = file_info.name
                # Clean up old kernel files (older than 1 hour)
                if filename.endswith('.cu') and ('compile_' in filename or 'kernel_' in filename):
                    try:
                        # Extract timestamp from filename
                        if 'compile_' in filename:
                            timestamp_str = filename.replace('compile_', '').replace('.cu', '')
                        elif 'kernel_' in filename:
                            timestamp_str = filename.replace('kernel_', '').replace('.cu', '')
                        else:
                            continue
                        
                        file_timestamp = int(timestamp_str)
                        # If file is older than 1 hour (3600 seconds), delete it
                        if current_time - file_timestamp > 3600:
                            self.repo.delete_file(
                                f"{self.kernels_path}/{filename}",
                                f"Clean up old kernel file {filename}",
                                file_info.sha,
                                branch="main"
                            )
                            logger.info(f"Deleted old kernel file: {filename}")
                            cleaned_count += 1
                            
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error processing {filename}: {e}")
                        continue
            
            logger.info(f"Cleaned up {cleaned_count} old kernel files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clean up old kernel files: {e}")
            return False
    
    async def execute_cuda_kernel(
        self, 
        kernel_code: str, 
        hardware: str = "NVIDIA T4",
        timeout: int = 600  # Increased to 10 minutes for GitHub-Colab cycle
    ) -> Dict[str, Any]:
        """
        Execute CUDA kernel via GitHub + Colab integration.
        
        Args:
            kernel_code: CUDA kernel code to execute
            hardware: Target hardware specification
            timeout: Maximum wait time in seconds
            
        Returns:
            Dictionary containing execution metrics or error information
        """
        # Ensure results directory exists before proceeding
        if not self.ensure_results_directory():
            error_msg = "Failed to ensure results directory exists"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Clean up old kernel files before uploading new one
        self.clean_old_kernel_files()
        
        # Create unique filename
        timestamp = int(time.time())
        filename = f"{self.kernels_path}/kernel_{timestamp}.cu"
        
        try:
            logger.info(f"Uploading kernel to GitHub: {filename}")
            
            # Upload kernel code to GitHub with execution metadata
            metadata = f"""// EXECUTION REQUEST
// Hardware: {hardware}
// Backend: CUDA
// Timestamp: {timestamp}
// Type: execute

{kernel_code}"""
            
            self.repo.create_file(
                filename,
                f"Kernel execution request {timestamp}",
                metadata,
                branch="main"
            )
            
            logger.info(f"Kernel uploaded successfully: {filename}")
            
            # Wait for Colab to process and return result
            result = await self._wait_for_result(timestamp, timeout)
            
            # Clean up kernel file
            try:
                kernel_file = self.repo.get_contents(filename, ref="main")
                self.repo.delete_file(
                    filename,
                    f"Cleanup kernel file {timestamp}",
                    kernel_file.sha,
                    branch="main"
                )
                logger.info(f"Kernel file cleaned up: {filename}")
            except Exception as e:
                logger.warning(f"Failed to clean up kernel file: {e}")
            
            return result
            
        except GithubException as e:
            error_msg = f"GitHub API error: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def compile_kernel_on_colab(
        self, 
        kernel_code: str, 
        hardware: str = "NVIDIA T4",
        backend: str = "CUDA",
        timeout: int = 300  # 5 minutes for compilation only
    ) -> Dict[str, Any]:
        """
        Compile kernel via GitHub + Colab integration (compilation only).
        
        Args:
            kernel_code: Kernel code to compile
            hardware: Target hardware specification
            backend: Backend type (CUDA/Triton)
            timeout: Maximum wait time in seconds
            
        Returns:
            Dictionary containing compilation results or error information
        """
        # Ensure results directory exists before proceeding
        if not self.ensure_results_directory():
            error_msg = "Failed to ensure results directory exists"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Clean up old kernel files before uploading new one
        self.clean_old_kernel_files()
        
        # Create unique filename with compilation prefix
        timestamp = int(time.time())
        filename = f"{self.kernels_path}/compile_{timestamp}.cu"
        
        try:
            logger.info(f"Uploading kernel for compilation to GitHub: {filename}")
            
            # Upload kernel code to GitHub with compilation metadata
            metadata = f"""// COMPILATION REQUEST
// Hardware: {hardware}
// Backend: {backend}
// Timestamp: {timestamp}
// Type: compile_only

{kernel_code}"""
            
            self.repo.create_file(
                filename,
                f"Kernel compilation request {timestamp}",
                metadata,
                branch="main"
            )
            
            logger.info(f"Kernel uploaded for compilation: {filename}")
            
            # Wait for Colab to process and return compilation result
            result = await self._wait_for_result(timestamp, timeout, result_type="compile")
            
            # Clean up kernel file
            try:
                kernel_file = self.repo.get_contents(filename, ref="main")
                self.repo.delete_file(
                    filename,
                    f"Cleanup compilation file {timestamp}",
                    kernel_file.sha,
                    branch="main"
                )
                logger.info(f"Compilation file cleaned up: {filename}")
            except Exception as e:
                logger.warning(f"Failed to clean up compilation file: {e}")
            
            return result
            
        except GithubException as e:
            error_msg = f"GitHub API error: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Compilation failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def _wait_for_result(
        self, 
        timestamp: int, 
        timeout: int = 600,  # Increased to 10 minutes for GitHub-Colab cycle
        result_type: str = "kernel"  # "kernel" for execution, "compile" for compilation
    ) -> Dict[str, Any]:
        """
        Wait for Colab to process kernel and return result.
        
        Args:
            timestamp: Timestamp used for kernel filename
            timeout: Maximum wait time in seconds
            result_type: Type of result to wait for ("kernel" or "compile")
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        if result_type == "compile":
            result_filename = f"{self.results_path}/compile_{timestamp}_result.json"
        else:
            result_filename = f"{self.results_path}/kernel_{timestamp}_result.json"
        
        logger.info(f"Waiting for result: {result_filename}")
        logger.info(f"Timeout: {timeout} seconds")
        logger.info(f"Looking for result file with timestamp: {timestamp}")
        logger.info(f"Result type: {result_type}")
        
        while time.time() - start_time < timeout:
            try:
                # Check if result file exists
                logger.info(f"Checking for result file: {result_filename}")
                result_file = self.repo.get_contents(result_filename, ref="main")
                
                if result_file:
                    # Download and parse result
                    result_content = result_file.decoded_content.decode('utf-8')
                    result = json.loads(result_content)
                    
                    logger.info(f"Result retrieved successfully: {result_filename}")
                    
                    # Delete result file after reading
                    try:
                        self.repo.delete_file(
                            result_filename,
                            f"Cleanup result file {timestamp}",
                            result_file.sha,
                            branch="main"
                        )
                        logger.info(f"Result file cleaned up: {result_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up result file: {e}")
                    
                    return result
                    
            except GithubException as e:
                if e.status == 404:
                    # File doesn't exist yet, continue waiting
                    # Debug: List available result files
                    try:
                        results_contents = self.repo.get_contents(self.results_path, ref="main")
                        available_files = [item.name for item in results_contents if item.name.endswith('.json')]
                        logger.info(f"Available result files: {available_files}")
                        logger.info(f"Looking for: {result_filename}")
                        logger.info(f"Files matching pattern: {[f for f in available_files if str(timestamp) in f]}")
                    except Exception as debug_e:
                        logger.warning(f"Debug listing failed: {debug_e}")
                        pass
                else:
                    logger.error(f"GitHub API error while waiting: {e}")
                    return {"error": f"GitHub API error: {e}"}
            except Exception as e:
                logger.error(f"Error while waiting for result: {e}")
                return {"error": f"Error waiting for result: {e}"}
            
            # Wait before checking again
            await asyncio.sleep(10)
        
        # Timeout reached
        error_msg = f"Timeout waiting for result after {timeout} seconds"
        logger.error(error_msg)
        return {"error": error_msg}
    
    async def get_execution_status(self) -> Dict[str, Any]:
        """
        Get current execution status and queue information.
        
        Returns:
            Dictionary containing status information
        """
        try:
            # Check kernels directory for pending files
            kernels_contents = self.repo.get_contents(self.kernels_path, ref="main")
            pending_kernels = [item.name for item in kernels_contents if item.name.endswith('.cu')]
            
            # Check results directory for recent results
            results_contents = self.repo.get_contents(self.results_path, ref="main")
            recent_results = [item.name for item in results_contents if item.name.endswith('.json')]
            
            return {
                "status": "available",
                "provider": "github_colab",
                "pending_kernels": len(pending_kernels),
                "recent_results": len(recent_results),
                "kernels_path": self.kernels_path,
                "results_path": self.results_path
            }
            
        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test GitHub connection and repository access.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to access repository
            repo_info = self.repo.get_contents("")
            logger.info(f"GitHub connection test successful for {self.owner}/{self.repo_name}")
            return True
        except Exception as e:
            logger.error(f"GitHub connection test failed: {e}")
            return False
