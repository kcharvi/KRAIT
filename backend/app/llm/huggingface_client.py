import os
import torch
from typing import List, Dict, Any, Optional, AsyncGenerator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    TextGenerationPipeline
)
from concurrent.futures import ThreadPoolExecutor
import asyncio
import gc

from ..config import settings
from ..models import ChatMessage, ChatResponse


class HuggingFaceClient:
    def __init__(self):
        self.api_key = settings.huggingface_api_key
        self.model_name = settings.huggingface_model
        self.device = self._get_device()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._model_loaded = False
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _get_device(self) -> str:
        """Determine the best device to use"""
        if settings.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return settings.device
    
    async def _load_model(self):
        """Load the model and tokenizer asynchronously"""
        if self._model_loaded:
            return
        
        def _load():
            try:
                print(f"Loading Hugging Face model: {self.model_name}")
                print(f"Using device: {self.device}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.api_key,
                    trust_remote_code=True
                )
                
                # Add padding token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=self.api_key,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Create pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    return_full_text=False,
                    do_sample=True,
                    temperature=settings.temperature,
                    max_new_tokens=settings.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                self._model_loaded = True
                print(f"Model {self.model_name} loaded successfully on {self.device}")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise e
        
        # Load model in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _load)
    
    def _format_messages(self, messages: List) -> str:
        """Format messages for the model"""
        formatted_text = ""
        
        for msg in messages:
            # Handle both ChatMessage objects and plain dictionaries
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content
            else:
                role = msg.get('role')
                content = msg.get('content')
            
            if role == "system":
                formatted_text += f"System: {content}\n"
            elif role == "user":
                formatted_text += f"Human: {content}\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n"
        
        # Add prompt for response
        formatted_text += "Assistant: "
        return formatted_text
    
    async def generate_response(
        self, 
        messages: List, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> ChatResponse:
        """Generate a response using Hugging Face model"""
        try:
            # Ensure model is loaded
            await self._load_model()
            
            # Format messages
            formatted_text = self._format_messages(messages)
            
            # Update pipeline parameters if provided
            if temperature is not None:
                self.pipeline.generation_config.temperature = temperature
            if max_tokens is not None:
                self.pipeline.generation_config.max_new_tokens = max_tokens
            
            def _generate():
                try:
                    # Generate response
                    result = self.pipeline(
                        formatted_text,
                        max_new_tokens=max_tokens or settings.max_tokens,
                        temperature=temperature or settings.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_full_text=False
                    )
                    
                    # Extract generated text
                    generated_text = result[0]['generated_text'].strip()
                    
                    # Calculate token usage
                    input_tokens = len(self.tokenizer.encode(formatted_text))
                    output_tokens = len(self.tokenizer.encode(generated_text))
                    
                    return {
                        'text': generated_text,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens
                    }
                    
                except Exception as e:
                    print(f"Generation error: {str(e)}")
                    raise e
            
            # Run generation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _generate)
            
            return ChatResponse(
                content=result['text'],
                model=self.model_name,
                provider="huggingface",
                usage={
                    "prompt_tokens": result['input_tokens'],
                    "completion_tokens": result['output_tokens'],
                    "total_tokens": result['total_tokens']
                }
            )
            
        except Exception as e:
            raise Exception(f"Hugging Face model error: {str(e)}")
    
    async def generate_stream_response(
        self, 
        messages: List, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Hugging Face model"""
        try:
            # For now, we'll generate the full response and simulate streaming
            # True streaming would require more complex implementation
            response = await self.generate_response(
                messages, temperature, max_tokens, stream=False
            )
            
            # Simulate streaming by yielding chunks
            content = response.content
            chunk_size = 20  # Words per chunk
            words = content.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield chunk
                await asyncio.sleep(0.1)  # Delay to simulate streaming
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Hugging Face models"""
        return [
            {
                "name": "facebook/KernelLLM",
                "provider": "huggingface", 
                "description": "Facebook KernelLLM model for text generation",
                "max_tokens": 2048,  # Adjust based on model capabilities
                "supports_streaming": True
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "provider": "huggingface",
                "description": "DialoGPT medium model for conversational AI",
                "max_tokens": 1024,
                "supports_streaming": True
            },
            {
                "name": "microsoft/DialoGPT-large",
                "provider": "huggingface",
                "description": "DialoGPT large model for conversational AI",
                "max_tokens": 1024,
                "supports_streaming": True
            },
        ]
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.pipeline is not None:
            del self.pipeline
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        self._model_loaded = False
