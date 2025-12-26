"""
Coding Assistant Agent

An AI agent specialized for helping developers with coding tasks.
"""

from typing import Optional
from .base_agent import BaseAgent, Tool
from ..llm.providers import LLMProvider


class CodingAssistantAgent(BaseAgent):
    """
    Coding Assistant for developers.
    
    Helps with:
    - Code explanation and review
    - Bug fixing and debugging
    - Writing new code
    - Documentation
    - Best practices
    """
    
    @property
    def name(self) -> str:
        return "Coding Assistant"
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert coding assistant.

PERSONALITY:
- Patient and educational
- Explains concepts clearly
- Provides practical, production-ready code

EXPERTISE:
- Python, JavaScript/TypeScript, Java, Go
- Web development (React, Node.js, Django, FastAPI)
- Database design (PostgreSQL, MySQL, MongoDB)
- Cloud basics (AWS, GCP, Azure)
- DevOps fundamentals (Docker, CI/CD)
- API design and integration

GUIDELINES:
1. Always explain your code with comments
2. Follow language-specific best practices
3. Consider security implications
4. Suggest error handling
5. Recommend testing approaches
6. Mention when something needs more expertise

CODE STYLE:
- Clean, readable code over clever tricks
- Meaningful variable names
- Appropriate comments
- Error handling included
- Type hints where applicable

FORMAT RESPONSES:
- Use markdown code blocks with language tags
- Explain before showing code
- Break down complex solutions
- Provide examples for concepts

LIMITATIONS:
- Cannot run or test code
- Cannot access external systems
- For complex architecture decisions, recommend consulting a senior developer
- Security-critical code should be reviewed by an expert"""
    
    def _register_tools(self):
        """Register coding assistant tools."""
        
        # Explain Code
        self.add_tool(Tool(
            name="explain_code",
            description="Provide a detailed explanation of a code snippet",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code snippet to explain"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language of the code"
                    }
                },
                "required": ["code"]
            },
            function=self._explain_code
        ))
        
        # Generate Documentation
        self.add_tool(Tool(
            name="generate_docs",
            description="Generate documentation for code",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to document"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["docstring", "markdown", "jsdoc"],
                        "description": "Documentation style"
                    }
                },
                "required": ["code"]
            },
            function=self._generate_docs
        ))
        
        # Suggest Tests
        self.add_tool(Tool(
            name="suggest_tests",
            description="Suggest test cases for a function or class",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to test"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Testing framework (pytest, jest, junit)"
                    }
                },
                "required": ["code"]
            },
            function=self._suggest_tests
        ))
        
        # Code Review
        self.add_tool(Tool(
            name="review_code",
            description="Perform a code review with suggestions",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to review"
                    },
                    "focus": {
                        "type": "string",
                        "description": "What to focus on (security, performance, readability)"
                    }
                },
                "required": ["code"]
            },
            function=self._review_code
        ))
    
    def _explain_code(self, code: str, language: str = "auto") -> str:
        """Provide code explanation."""
        return f"""Code Explanation Request:
Language: {language}
Code:
```
{code[:500]}{'...' if len(code) > 500 else ''}
```

I'll analyze this code and provide:
1. Overall purpose
2. Step-by-step breakdown
3. Key concepts used
4. Potential improvements"""
    
    def _generate_docs(self, code: str, style: str = "docstring") -> str:
        """Generate documentation for code."""
        return f"""Documentation Generation:
Style: {style}
Code to document:
```
{code[:300]}{'...' if len(code) > 300 else ''}
```

Generating {style}-style documentation..."""
    
    def _suggest_tests(self, code: str, framework: str = "pytest") -> str:
        """Suggest test cases."""
        return f"""Test Suggestions:
Framework: {framework}
Code to test:
```
{code[:300]}{'...' if len(code) > 300 else ''}
```

Test cases to consider:
1. Happy path testing
2. Edge cases
3. Error handling
4. Boundary conditions"""
    
    def _review_code(self, code: str, focus: str = "general") -> str:
        """Perform code review."""
        return f"""Code Review:
Focus: {focus}
Code:
```
{code[:500]}{'...' if len(code) > 500 else ''}
```

Review checklist:
- [ ] Code clarity and readability
- [ ] Error handling
- [ ] Security considerations
- [ ] Performance implications
- [ ] Best practices adherence"""
    
    def _post_process(self, response: str) -> str:
        """Ensure code blocks are properly formatted."""
        return response


def create_coding_assistant(
    provider: LLMProvider = LLMProvider.NVIDIA,
    api_key: Optional[str] = None
) -> CodingAssistantAgent:
    """Factory function to create a coding assistant."""
    return CodingAssistantAgent(provider=provider, api_key=api_key)
