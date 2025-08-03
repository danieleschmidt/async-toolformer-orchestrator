"""LLM integration for tool calling decisions."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from an LLM."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
    metadata: Dict[str, Any] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def get_tool_calls(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[ToolCall]:
        """Get tool calls from the LLM."""
        pass
    
    @abstractmethod
    def format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for the specific LLM provider."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider for tool calling."""
    
    def __init__(self, client, default_model: str = "gpt-4o"):
        """
        Initialize OpenAI provider.
        
        Args:
            client: OpenAI client instance
            default_model: Default model to use
        """
        self.client = client
        self.default_model = default_model
    
    async def get_tool_calls(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[ToolCall]:
        """Get tool calls from OpenAI."""
        try:
            # Format tools for OpenAI
            formatted_tools = self.format_tools_for_llm(tools)
            
            # Create chat completion with tools
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=formatted_tools,
                tool_choice="auto",  # Let the model decide
                **kwargs
            )
            
            # Extract tool calls
            tool_calls = []
            message = response.choices[0].message
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse arguments: {tc.function.arguments}")
                        arguments = {}
                    
                    tool_call = ToolCall(
                        name=tc.function.name,
                        arguments=arguments,
                        id=tc.id,
                        metadata={"model": model or self.default_model}
                    )
                    tool_calls.append(tool_call)
            
            return tool_calls
            
        except Exception as e:
            logger.error(f"OpenAI tool call failed: {e}")
            return []
    
    def format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI function calling."""
        formatted = []
        
        for tool in tools:
            # OpenAI function format
            function_def = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            formatted.append(function_def)
        
        return formatted


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider for tool calling."""
    
    def __init__(self, client, default_model: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic provider.
        
        Args:
            client: Anthropic client instance
            default_model: Default model to use
        """
        self.client = client
        self.default_model = default_model
    
    async def get_tool_calls(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[ToolCall]:
        """Get tool calls from Anthropic."""
        try:
            # Format tools for Anthropic
            formatted_tools = self.format_tools_for_llm(tools)
            
            # Create message with tools
            response = await self.client.messages.create(
                model=model or self.default_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=formatted_tools,
                max_tokens=4096,
                **kwargs
            )
            
            # Extract tool calls
            tool_calls = []
            
            for content in response.content:
                if content.type == "tool_use":
                    tool_call = ToolCall(
                        name=content.name,
                        arguments=content.input,
                        id=content.id,
                        metadata={"model": model or self.default_model}
                    )
                    tool_calls.append(tool_call)
            
            return tool_calls
            
        except Exception as e:
            logger.error(f"Anthropic tool call failed: {e}")
            return []
    
    def format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic tool use."""
        formatted = []
        
        for tool in tools:
            # Anthropic tool format
            tool_def = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
            formatted.append(tool_def)
        
        return formatted


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, mock_responses: Optional[List[ToolCall]] = None):
        """
        Initialize mock provider.
        
        Args:
            mock_responses: Predefined tool calls to return
        """
        self.mock_responses = mock_responses or []
    
    async def get_tool_calls(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> List[ToolCall]:
        """Return mock tool calls."""
        if self.mock_responses:
            return self.mock_responses
        
        # Generate mock calls based on available tools
        tool_calls = []
        for i, tool in enumerate(tools[:3]):  # Mock: call first 3 tools
            tool_call = ToolCall(
                name=tool["name"],
                arguments={},
                id=f"mock_{i}",
                metadata={"mock": True}
            )
            tool_calls.append(tool_call)
        
        return tool_calls
    
    def format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return tools as-is for mock provider."""
        return tools


class LLMIntegration:
    """
    Main LLM integration class for managing multiple providers.
    """
    
    def __init__(self, default_provider: Optional[str] = None):
        """
        Initialize LLM integration.
        
        Args:
            default_provider: Default provider name
        """
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = default_provider
        
        # Metrics
        self._metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "tools_called": 0,
        }
    
    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register an LLM provider."""
        self.providers[name] = provider
        
        if not self.default_provider:
            self.default_provider = name
        
        logger.info(f"Registered LLM provider: {name}")
    
    async def get_tool_calls(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> List[ToolCall]:
        """
        Get tool calls from an LLM provider.
        
        Args:
            prompt: Input prompt
            tools: Available tools
            provider: Provider name (uses default if not specified)
            model: Model to use
            **kwargs: Additional provider-specific arguments
            
        Returns:
            List of tool calls
        """
        provider_name = provider or self.default_provider
        
        if not provider_name:
            logger.error("No LLM provider specified or configured")
            self._metrics["failed_calls"] += 1
            return []
        
        if provider_name not in self.providers:
            logger.error(f"Unknown provider: {provider_name}")
            self._metrics["failed_calls"] += 1
            return []
        
        self._metrics["total_calls"] += 1
        
        try:
            provider_instance = self.providers[provider_name]
            tool_calls = await provider_instance.get_tool_calls(
                prompt, tools, model, **kwargs
            )
            
            self._metrics["successful_calls"] += 1
            self._metrics["tools_called"] += len(tool_calls)
            
            logger.info(
                f"Got {len(tool_calls)} tool calls from {provider_name}"
            )
            
            return tool_calls
            
        except Exception as e:
            logger.error(f"Failed to get tool calls from {provider_name}: {e}")
            self._metrics["failed_calls"] += 1
            return []
    
    def format_tools_for_provider(
        self,
        tools: List[Dict[str, Any]],
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Format tools for a specific provider."""
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            return tools
        
        return self.providers[provider_name].format_tools_for_llm(tools)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM integration metrics."""
        success_rate = 0.0
        if self._metrics["total_calls"] > 0:
            success_rate = (
                self._metrics["successful_calls"] /
                self._metrics["total_calls"]
            )
        
        return {
            **self._metrics,
            "success_rate": success_rate,
            "registered_providers": list(self.providers.keys()),
            "default_provider": self.default_provider,
        }


def create_llm_integration(
    openai_client=None,
    anthropic_client=None,
    default_provider: Optional[str] = None,
    use_mock: bool = False,
) -> LLMIntegration:
    """
    Create an LLM integration with configured providers.
    
    Args:
        openai_client: OpenAI client instance
        anthropic_client: Anthropic client instance
        default_provider: Default provider to use
        use_mock: Whether to include mock provider
        
    Returns:
        Configured LLMIntegration instance
    """
    integration = LLMIntegration(default_provider=default_provider)
    
    # Register OpenAI if client provided
    if openai_client:
        integration.register_provider(
            "openai",
            OpenAIProvider(openai_client)
        )
    
    # Register Anthropic if client provided
    if anthropic_client:
        integration.register_provider(
            "anthropic",
            AnthropicProvider(anthropic_client)
        )
    
    # Register mock provider if requested
    if use_mock:
        integration.register_provider(
            "mock",
            MockProvider()
        )
    
    return integration