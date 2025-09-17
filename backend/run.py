#!/usr/bin/env python3
"""
Simple script to run the FastAPI backend server
"""

import uvicorn
from app.config import settings
from pydantic import Field
from pydantic_settings import BaseSettings

if __name__ == "__main__":
    print(f"Starting {settings.app_name} v{settings.version}")
    print(f"Server will be available at: http://{settings.host}:{settings.port}")
    print(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
