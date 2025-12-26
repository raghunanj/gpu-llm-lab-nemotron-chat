"""
Nemotron 3 Client

Provides a clean interface to NVIDIA's Nemotron 3 models via their NIM API.
"""

import os
from typing import Generator, Optional, Any
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class ChatResponse:
    """Represents a chat completion response."""
    content: str
    model: str
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None


class NemotronClient:
    """
    Client for NVIDIA Nemotron 3 models.
    
    Uses OpenAI-compatible API for easy integration.
    """
    
    DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b"
    API_BASE = "https://integrate.api.nvidia.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize the Nemotron client.
        
        Args:
            api_key: NVIDIA API key (falls back to NVIDIA_API_KEY env var)
            model: Model identifier (defaults to Nemotron 3 Nano)
            api_base: API base URL (for custom deployments)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.api_base = api_base or self.API_BASE
        self._client = None
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    @property
    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        return self._client is not None and self.api_key is not None
    
    def chat(
        self,
        messages: list[dict | ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[list] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse | Generator[str, None, None]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of messages in conversation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            tools: Optional list of tools for function calling
            stream: Whether to stream the response
            **kwargs: Additional parameters passed to the API
            
        Returns:
            ChatResponse or generator of strings if streaming
        """
        if not self.is_available:
            raise RuntimeError("Nemotron client not configured. Set NVIDIA_API_KEY.")
        
        # Convert ChatMessage objects to dicts
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {})
                })
            else:
                formatted_messages.append(msg)
        
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        if stream:
            return self._stream_chat(params)
        else:
            return self._complete_chat(params)
    
    def _complete_chat(self, params: dict) -> ChatResponse:
        """Execute a non-streaming chat completion."""
        response = self._client.chat.completions.create(**params)
        
        return ChatResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def _stream_chat(self, params: dict) -> Generator[str, None, None]:
        """Execute a streaming chat completion."""
        stream = self._client.chat.completions.create(**params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple single-turn chat.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            
        Returns:
            The model's response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        response = self.chat(messages)
        return response.content
    
    def get_models(self) -> list[str]:
        """Get available Nemotron models."""
        return [
            "nvidia/nemotron-3-nano-30b",
            "nvidia/nemotron-3-super-100b",  # If available
            "nvidia/nemotron-3-ultra-500b",  # If available
        ]


# Convenience function
def create_nemotron_client(**kwargs) -> NemotronClient:
    """Create a Nemotron client with optional configuration."""
    return NemotronClient(**kwargs)


