# KRAIT GPU Executor Setup Guide

This guide will help you set up the GPU execution functionality for KRAIT using GitHub + Google Colab integration.

## Prerequisites

1. **GitHub Account**: You need a GitHub account with a repository
2. **Google Account**: For accessing Google Colab
3. **GitHub Personal Access Token**: For API access

## Step 1: Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "KRAIT GPU Executor"
4. Select the following scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
5. Click "Generate token"
6. **Copy the token immediately** - you won't be able to see it again

## Step 2: Configure Backend Environment

Create a `.env` file in the `backend/` directory with the following content:

```env
# GitHub Integration for GPU Execution
GITHUB_TOKEN=your_github_personal_access_token_here
GITHUB_OWNER=your_github_username_here
GITHUB_REPO_NAME=krait

# Other existing environment variables...
GEMINI_API_KEY=your_gemini_api_key_here
# ... etc
```

Replace:
- `your_github_personal_access_token_here` with your actual GitHub token
- `your_github_username_here` with your GitHub username
- `krait` with your actual repository name if different

## Step 3: Update Colab Notebook

1. Open `gpu-executor/executor.ipynb` in your repository
2. Update the `REPO_URL` variable with your actual repository URL:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/krait.git"
```

Replace `YOUR_USERNAME` and `krait` with your actual GitHub username and repository name.

## Step 4: Set Up Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Open the `executor.ipynb` notebook from your repository:
   - File → Open notebook → GitHub
   - Enter your repository URL
   - Navigate to `gpu-executor/executor.ipynb`
3. Enable GPU in Runtime settings:
   - Runtime → Change runtime type → Hardware accelerator → GPU
4. Run all cells in the notebook to start monitoring

## Step 5: Test the Setup

1. Start your KRAIT backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   python -m uvicorn app.main:app --reload
   ```

2. Start your KRAIT frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Test GPU execution:
   - Generate a kernel in the frontend
   - Click "Analyze Kernel" to run the critic analysis
   - Expand the "Real GPU Execution" section
   - Click "Run on GPU" to execute on Colab

## Step 6: Verify Integration

1. **Check Backend Logs**: Look for successful GitHub API calls
2. **Check Colab Output**: The notebook should show monitoring activity
3. **Check Frontend**: Real metrics should appear after execution

## Troubleshooting

### Common Issues

1. **"GitHub integration not configured"**
   - Check that `GITHUB_TOKEN` and `GITHUB_OWNER` are set in `.env`
   - Restart the backend after updating `.env`

2. **"Failed to connect to GitHub repository"**
   - Verify your GitHub token has the correct permissions
   - Check that the repository name is correct
   - Ensure the repository is public or the token has access

3. **"Timeout waiting for result"**
   - Check that the Colab notebook is running
   - Verify the repository URL in the notebook is correct
   - Check Colab GPU availability

4. **"Compilation failed"**
   - The kernel code may have syntax errors
   - Check the Colab output for detailed error messages

5. **Colab notebook not detecting files**
   - Ensure the notebook is running and monitoring
   - Check that files are being uploaded to the correct directory
   - Verify GitHub repository permissions

### Debug Steps

1. **Check Backend Status**:
   ```bash
   curl http://localhost:8000/api/v1/gpu/status
   ```

2. **Test GitHub Connection**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/gpu/test-connection
   ```

3. **Check Colab Logs**: Look at the Colab notebook output for processing messages

4. **Check GitHub Repository**: Verify files are being created in `gpu-executor/kernels/` and `gpu-executor/results/`

## Security Notes

- Keep your GitHub token secure and never commit it to version control
- The `.env` file should be in `.gitignore`
- Consider using GitHub Apps for production instead of personal access tokens
- Kernel files are automatically cleaned up after processing

## Performance Tips

- Colab has free GPU limits - monitor your usage
- Large kernels may take longer to compile and execute
- Consider using Colab Pro for better performance and longer sessions
- The system includes automatic retry logic for failed executions

## Next Steps

Once the basic setup is working:

1. **Enhance Metrics**: Improve the `extract_metrics()` function in the Colab notebook
2. **Add More Providers**: Implement additional GPU execution providers
3. **Improve Error Handling**: Add more robust error recovery
4. **Add Monitoring**: Implement execution queue monitoring
5. **Optimize Performance**: Add caching and optimization features
