"""
LLM Provider Interface - Abstract interface for multi-LLM support.

This interface enables the framework to work with different LLM providers
(Ollama, OpenAI, Anthropic, Gemini, etc.) through a unified API.

Key features:
- Sync and async invoke methods
- Provider-agnostic interface
- Extensible for new providers
- Supports complex experiments with multiple LLMs
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncIterator
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 60.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized response from LLM provider."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)  # tokens used
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    
    Implementations:
    - OllamaProvider: Local models via Ollama
    - OpenAIProvider: OpenAI API (GPT-4, etc.)
    - AnthropicProvider: Anthropic API (Claude)
    - GeminiProvider: Google Gemini API
    
    Usage:
        provider = OllamaProvider(config)
        response = provider.invoke("Hello, world!")
        # or async
        response = await provider.ainvoke("Hello, world!")
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'ollama', 'openai')."""
        pass
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.config.model
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Synchronously invoke the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for this call
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Asynchronously invoke the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for this call
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Stream response asynchronously.
        
        Default implementation returns the full response.
        Override for true streaming support.
        """
        response = await self.ainvoke(prompt, **kwargs)
        yield response.content
    
    def validate_connection(self) -> bool:
        """
        Validate connection to the LLM provider.
        
        Returns:
            True if connection is valid
        """
        try:
            response = self.invoke("ping", max_tokens=10)
            return len(response.content) > 0
        except Exception:
            return False


class LLMProviderRegistry:
    """
    Registry for LLM providers.
    
    Enables multi-LLM experiments where different agents or decisions
    use different LLM providers.
    
    Usage:
        registry = LLMProviderRegistry()
        registry.register("local", OllamaProvider(...))
        registry.register("cloud", OpenAIProvider(...))
        
        # Use specific provider
        response = registry.get("local").invoke(prompt)
        
        # Use default
        response = registry.default.invoke(prompt)
    """
    
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._default: Optional[str] = None
    
    def register(self, name: str, provider: LLMProvider, set_default: bool = False) -> None:
        """Register a provider with a name."""
        self._providers[name] = provider
        if set_default or self._default is None:
            self._default = name
    
    def get(self, name: str) -> LLMProvider:
        """Get a provider by name."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]
    
    @property
    def default(self) -> LLMProvider:
        """Get the default provider."""
        if self._default is None:
            raise ValueError("No default provider set")
        return self._providers[self._default]
    
    def set_default(self, name: str) -> None:
        """Set the default provider."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        self._default = name
    
    def list_providers(self) -> list:
        """List all registered provider names."""
        return list(self._providers.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._providers


class RoutingLLMProvider(LLMProvider):
    """
    A meta-provider that routes requests to different providers based on rules.
    
    Use cases:
    - Route high-stakes decisions to cloud LLM
    - Use local LLM for simple decisions
    - Fallback to cloud if local fails
    
    Usage:
        router = RoutingLLMProvider(
            config=LLMConfig(model="router"),
            registry=registry,
            routing_rules={
                "high_stakes": "cloud",
                "default": "local"
            }
        )
    """
    
    def __init__(
        self,
        config: LLMConfig,
        registry: LLMProviderRegistry,
        routing_rules: Dict[str, str],
        fallback_provider: Optional[str] = None
    ):
        super().__init__(config)
        self.registry = registry
        self.routing_rules = routing_rules
        self.fallback_provider = fallback_provider
    
    @property
    def provider_name(self) -> str:
        return "routing"
    
    def _get_provider(self, context: str = "default") -> LLMProvider:
        """Get provider based on context."""
        provider_name = self.routing_rules.get(context, self.routing_rules.get("default"))
        return self.registry.get(provider_name)
    
    def invoke(self, prompt: str, context: str = "default", **kwargs) -> LLMResponse:
        """Route and invoke."""
        provider = self._get_provider(context)
        try:
            return provider.invoke(prompt, **kwargs)
        except Exception as e:
            if self.fallback_provider:
                fallback = self.registry.get(self.fallback_provider)
                return fallback.invoke(prompt, **kwargs)
            raise e
    
    async def ainvoke(self, prompt: str, context: str = "default", **kwargs) -> LLMResponse:
        """Route and async invoke."""
        provider = self._get_provider(context)
        try:
            return await provider.ainvoke(prompt, **kwargs)
        except Exception as e:
            if self.fallback_provider:
                fallback = self.registry.get(self.fallback_provider)
                return await fallback.ainvoke(prompt, **kwargs)
            raise e
