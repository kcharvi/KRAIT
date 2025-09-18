import os
from typing import List, Dict, Any, Optional, AsyncGenerator
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config import settings
from ..models import ChatMessage, ChatResponse


class GeminiClient:
    def __init__(self):
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini integration")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize LangChain client
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=self.api_key,
            temperature=settings.temperature,
            max_output_tokens=12000,  # Set higher limit directly
        )
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _convert_messages(self, messages: List) -> List:
        """Convert ChatMessage objects or dictionaries to LangChain message objects"""
        langchain_messages = []
        
        for msg in messages:
            # Handle both ChatMessage objects and plain dictionaries
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content
            else:
                role = msg.get('role')
                content = msg.get('content')
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        
        return langchain_messages
    
    async def generate_response(
        self, 
        messages: List, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> ChatResponse:
        """Generate a response using Gemini"""
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)
            
            # Update model parameters if provided
            if temperature is not None:
                self.llm.temperature = temperature
            if max_tokens is not None:
                self.llm.max_output_tokens = max_tokens
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.llm.invoke(langchain_messages)
            )
            
            # Extract content and usage info
            # Try different ways to get the content
            content = ""
            
            # Method 1: Try response.content
            if hasattr(response, 'content') and response.content:
                content = response.content
            # Method 2: Try response.text (method)
            elif hasattr(response, 'text'):
                if callable(response.text):
                    content = response.text()
                else:
                    content = response.text
            # Method 3: Try to get text from the response object
            elif hasattr(response, 'text') and callable(response.text):
                content = response.text()
            # Method 4: Convert to string
            else:
                content = str(response)
            
            # Ensure content is a string
            if callable(content):
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)
            
            print(f"DEBUG - Content extracted: '{content[:200]}...'")
            print(f"DEBUG - Content type: {type(content)}")
            print(f"DEBUG - Content length: {len(content)}")
            print(f"DEBUG - Response finish_reason: {getattr(response, 'response_metadata', {}).get('finish_reason', 'unknown')}")
            print(f"DEBUG - Usage metadata: {getattr(response, 'usage_metadata', {})}")
            
            # Debug: Uncomment to see response details
            # print(f"Gemini response content: '{content}'")
            # print(f"Gemini response type: {type(response)}")
            
            # Calculate usage info safely
            prompt_text = " ".join([msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '') for msg in messages])
            content_words = content.split() if content else []
            
            usage_info = {
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": len(content_words),
                "total_tokens": len(prompt_text.split()) + len(content_words)
            }
            
            return ChatResponse(
                content=content,
                model=settings.gemini_model,
                provider="gemini",
                usage=usage_info
            )
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def generate_stream_response(
        self, 
        messages: List, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Gemini"""
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)
            
            # Update model parameters if provided
            if temperature is not None:
                self.llm.temperature = temperature
            if max_tokens is not None:
                self.llm.max_output_tokens = max_tokens
            
            # For streaming, we'll simulate it by yielding chunks
            # Note: LangChain's Gemini integration doesn't support true streaming yet
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.llm.invoke(langchain_messages)
            )
            
            # Simulate streaming by yielding chunks
            content = response.content
            chunk_size = 10  # Smaller chunks for better streaming effect
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.1)  # Slightly longer delay for visible streaming
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def generate_content_async(self, prompt: str) -> str:
        """Generate content from a simple string prompt"""
        try:
            print(f"🔍 generate_content_async - Input prompt length: {len(prompt)}")
            # Convert string prompt to message format
            messages = [{"role": "user", "content": prompt}]
            print(f"🔍 generate_content_async - Messages: {messages}")
            
            response = await self.generate_response(messages)
            print(f"🔍 generate_content_async - Response type: {type(response)}")
            print(f"🔍 generate_content_async - Response has content: {hasattr(response, 'content')}")
            
            if hasattr(response, 'content'):
                print(f"🔍 generate_content_async - Content type: {type(response.content)}")
                print(f"🔍 generate_content_async - Content preview: {str(response.content)[:200]}...")
                return response.content
            else:
                print(f"🔍 generate_content_async - No content attr, converting to string")
                return str(response)
        except Exception as e:
            print(f"❌ generate_content_async - Exception: {str(e)}")
            print(f"❌ generate_content_async - Exception type: {type(e)}")
            import traceback
            print(f"❌ generate_content_async - Traceback: {traceback.format_exc()}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Gemini models"""
        return [
            {
                "name": "gemini-2.5-pro",
                "provider": "gemini",
                "description": "Gemini 2.5 Pro model for code generation",
                "max_tokens": 32768,
                "supports_streaming": False
            }
        ]
