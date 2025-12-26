"""
Base Agent Implementation

Inspired by Sierra AI's approach to building trustworthy AI agents.
Key principles:
1. Clear boundaries and capabilities
2. Conversation memory for context
3. Tool integration for extended functionality
4. Honest about limitations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
import json

from ..llm.providers import get_llm_client, LLMProvider


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "user", "assistant", "system", or "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentMemory:
    """
    Manages conversation history and context.
    
    Features:
    - Automatic context window management
    - Important information extraction
    - Summary generation for long conversations
    """
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 100
    important_facts: list[str] = field(default_factory=list)
    
    def add(self, role: str, content: str, **metadata):
        """Add a message to memory."""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata
        ))
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Trim old messages if we exceed the limit."""
        if len(self.messages) > self.max_messages:
            # Keep system messages and recent messages
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            
            keep_count = self.max_messages - len(system_msgs)
            self.messages = system_msgs + other_msgs[-keep_count:]
    
    def to_api_format(self) -> list[dict]:
        """Convert to API-compatible format."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
    
    def get_context_summary(self) -> str:
        """Get a summary of important context."""
        if not self.important_facts:
            return ""
        return "Important context:\n" + "\n".join(f"- {fact}" for fact in self.important_facts)
    
    def add_important_fact(self, fact: str):
        """Record an important fact from the conversation."""
        if fact not in self.important_facts:
            self.important_facts.append(fact)
    
    def clear(self):
        """Clear all messages but keep important facts."""
        self.messages = []
    
    def full_reset(self):
        """Reset everything."""
        self.messages = []
        self.important_facts = []


@dataclass
class Tool:
    """Represents a tool that the agent can use."""
    name: str
    description: str
    parameters: dict
    function: Callable
    
    def to_api_format(self) -> dict:
        """Convert to OpenAI-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        return self.function(**kwargs)


class BaseAgent(ABC):
    """
    Base class for AI agents.
    
    All agents should:
    1. Have a clear purpose (defined in system prompt)
    2. Manage conversation memory
    3. Support tool integration
    4. Be transparent about limitations
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.NVIDIA,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_memory: int = 100
    ):
        """
        Initialize the agent.
        
        Args:
            provider: LLM provider to use
            api_key: API key for the provider
            model: Model identifier
            max_memory: Maximum messages to keep in memory
        """
        self.llm = get_llm_client(provider=provider, api_key=api_key, model=model)
        self.memory = AgentMemory(max_messages=max_memory)
        self.tools: list[Tool] = []
        
        # Initialize with system prompt
        self._init_agent()
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Define the agent's personality and capabilities."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The agent's display name."""
        pass
    
    def _init_agent(self):
        """Initialize the agent with system prompt."""
        self.memory.add("system", self.system_prompt)
        self._register_tools()
    
    def _register_tools(self):
        """Override to register tools the agent can use."""
        pass
    
    def add_tool(self, tool: Tool):
        """Register a tool for the agent."""
        self.tools.append(tool)
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: The user's input
            
        Returns:
            The agent's response
        """
        # Add user message to memory
        self.memory.add("user", user_message)
        
        # Get response from LLM
        try:
            tools_api = [t.to_api_format() for t in self.tools] if self.tools else None
            
            response = self.llm.chat(
                messages=self.memory.to_api_format(),
                tools=tools_api
            )
            
            response_text = response.content
            
            # Post-process response
            response_text = self._post_process(response_text)
            
            # Add to memory
            self.memory.add("assistant", response_text)
            
            return response_text
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            self.memory.add("assistant", error_msg)
            return error_msg
    
    def _post_process(self, response: str) -> str:
        """
        Post-process the response.
        Override for custom processing.
        """
        return response
    
    def stream_chat(self, user_message: str):
        """
        Stream a response to a user message.
        
        Args:
            user_message: The user's input
            
        Yields:
            Response chunks as they arrive
        """
        self.memory.add("user", user_message)
        
        try:
            full_response = ""
            for chunk in self.llm.chat(
                messages=self.memory.to_api_format(),
                stream=True
            ):
                full_response += chunk
                yield chunk
            
            self.memory.add("assistant", full_response)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.memory.add("assistant", error_msg)
            yield error_msg
    
    def reset(self):
        """Reset conversation memory."""
        self.memory.clear()
        self.memory.add("system", self.system_prompt)
    
    def full_reset(self):
        """Fully reset the agent."""
        self.memory.full_reset()
        self._init_agent()
    
    def get_info(self) -> dict:
        """Get information about this agent."""
        return {
            "name": self.name,
            "tools": [t.name for t in self.tools],
            "memory_size": len(self.memory.messages),
            "important_facts": self.memory.important_facts
        }


