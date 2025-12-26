"""
LLM Provider abstraction layer.

Allows easy switching between different LLM providers (NVIDIA, OpenAI, local models).
"""

from enum import Enum
from typing import Protocol, Optional, Generator
from dataclasses import dataclass

from .nemotron import NemotronClient, ChatResponse


class LLMProvider(Enum):
    """Supported LLM providers."""
    NVIDIA = "nvidia"      # NVIDIA NIM API (cloud)
    OPENAI = "openai"      # OpenAI API
    LOCAL = "local"        # Local models via transformers + bitsandbytes
    OLLAMA = "ollama"      # Ollama for easy local inference


class LLMClientProtocol(Protocol):
    """Protocol defining the LLM client interface."""
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse | Generator[str, None, None]:
        """Send a chat completion request."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        ...


class MockLLMClient:
    """
    Mock LLM client for testing or when no API is available.
    """
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
    
    @property
    def is_available(self) -> bool:
        return True
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse | Generator[str, None, None]:
        """Generate a mock response."""
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        response_text = self._generate_mock_response(user_message)
        
        if stream:
            def stream_gen():
                for word in response_text.split():
                    yield word + " "
            return stream_gen()
        
        return ChatResponse(
            content=response_text,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
            finish_reason="stop"
        )
    
    def _generate_mock_response(self, user_message: str) -> str:
        """Generate a context-aware mock response."""
        message_lower = user_message.lower()
        
        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm the AI Assistant. How can I help you today?"
        
        if "help" in message_lower:
            return """I can help you with:
1. Customer support queries
2. Code review and debugging
3. General questions

What would you like to know?"""
        
        return f"""I understand you're asking about: "{user_message}"

As this is a mock response (API not configured), I can provide general guidance:
1. Configure your NVIDIA API key for full functionality
2. Visit build.nvidia.com to get an API key
3. Set NVIDIA_API_KEY environment variable

How else can I help you?"""
    
    def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Simple single-turn chat."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        response = self.chat(messages)
        return response.content


class LocalLLMClient:
    """
    Client for local model inference using transformers + bitsandbytes.
    """
    
    def __init__(
        self, 
        model_key: str = "nemotron-mini-4b",
        load_in_4bit: bool = True
    ):
        """
        Initialize local model.
        
        Args:
            model_key: Model identifier (nemotron-mini-4b, minitron-8b, nemotron-nano-30b)
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_key = model_key
        self.load_in_4bit = load_in_4bit
        self._client = None
        self._init_attempted = False
    
    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._init_attempted:
            return
        self._init_attempted = True
        
        try:
            # Import from local_inference module
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exploration" / "nemotron_examples"))
            from local_inference import LocalNemotronClient
            
            print(f"🔄 Loading local model: {self.model_key}")
            self._client = LocalNemotronClient(self.model_key, self.load_in_4bit)
            print("✅ Local model ready!")
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        self._lazy_init()
        return self._client is not None
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> ChatResponse | Generator[str, None, None]:
        """Chat with the local model."""
        self._lazy_init()
        
        if not self._client:
            return ChatResponse(
                content="Local model not available.",
                model="local-unavailable",
                usage=None,
                finish_reason="error"
            )
        
        response = self._client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return response
        
        return ChatResponse(
            content=response if isinstance(response, str) else str(response),
            model=self.model_key,
            usage=None,
            finish_reason="stop"
        )
    
    def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Simple single-turn chat."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        response = self.chat(messages)
        return response.content if hasattr(response, 'content') else str(response)


def get_llm_client(
    provider: LLMProvider = LLMProvider.NVIDIA,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMClientProtocol:
    """
    Factory function to get an LLM client.
    
    Args:
        provider: Which LLM provider to use
        api_key: API key for the provider
        model: Model identifier
        **kwargs: Additional provider-specific arguments
        
    Returns:
        An LLM client instance
    """
    if provider == LLMProvider.NVIDIA:
        client = NemotronClient(api_key=api_key, model=model, **kwargs)
        if client.is_available:
            return client
        else:
            print("⚠️  NVIDIA API not configured, falling back to mock client")
            return MockLLMClient(model=model or "nemotron-mock")
    
    elif provider == LLMProvider.OPENAI:
        # OpenAI implementation would go here
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    elif provider == LLMProvider.LOCAL:
        # Local model using transformers + bitsandbytes
        model_key = kwargs.get("model_key", "nemotron-mini-4b")
        load_in_4bit = kwargs.get("load_in_4bit", True)
        return LocalLLMClient(model_key=model_key, load_in_4bit=load_in_4bit)
    
    elif provider == LLMProvider.OLLAMA:
        # Ollama would need separate implementation
        raise NotImplementedError("Ollama provider not yet implemented. Use LOCAL instead.")
    
    else:
        return MockLLMClient()
