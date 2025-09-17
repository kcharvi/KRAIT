from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class ModelProvider(str, Enum):
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(..., description="Model name to use")
    provider: ModelProvider = Field(..., description="Model provider (gemini or huggingface)")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1000, gt=0, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    content: str = Field(..., description="Generated response content")
    model: str = Field(..., description="Model used for generation")
    provider: str = Field(..., description="Provider used for generation")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage information")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider")
    description: Optional[str] = Field(default=None, description="Model description")
    max_tokens: Optional[int] = Field(default=None, description="Maximum context length")
    supports_streaming: bool = Field(default=False, description="Whether model supports streaming")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    available_providers: List[str] = Field(..., description="Available model providers")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
