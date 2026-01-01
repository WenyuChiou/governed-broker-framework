"""
Rate Limiter - Control request rates for cloud LLM APIs.

Prevents hitting API rate limits during batch processing.
Supports both token-based and request-based limiting.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Supports both synchronous and asynchronous usage.
    
    Usage:
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=60))
        
        # Sync
        limiter.wait()
        response = api_call()
        
        # Async
        await limiter.await_wait()
        response = await async_api_call()
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._request_times: deque = deque()
        self._token_usage: deque = deque()
        self._window_seconds = 60.0
        self._lock = asyncio.Lock()
    
    def _cleanup_old_entries(self, entries: deque, current_time: float) -> None:
        """Remove entries older than the rate window."""
        cutoff = current_time - self._window_seconds
        while entries and entries[0] < cutoff:
            entries.popleft()
    
    def _get_wait_time(self) -> float:
        """Calculate wait time needed before next request."""
        current_time = time.time()
        self._cleanup_old_entries(self._request_times, current_time)
        
        if len(self._request_times) < self.config.requests_per_minute:
            return 0.0
        
        # Calculate when the oldest request will expire
        oldest_time = self._request_times[0]
        wait_time = (oldest_time + self._window_seconds) - current_time
        return max(0.0, wait_time)
    
    def wait(self) -> None:
        """Synchronously wait until rate limit allows."""
        wait_time = self._get_wait_time()
        if wait_time > 0:
            time.sleep(wait_time)
        self._request_times.append(time.time())
    
    async def await_wait(self) -> None:
        """Asynchronously wait until rate limit allows."""
        async with self._lock:
            wait_time = self._get_wait_time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._request_times.append(time.time())
    
    def record_tokens(self, token_count: int) -> None:
        """Record token usage for token-based limiting."""
        self._token_usage.append((time.time(), token_count))
    
    def get_available_tokens(self) -> int:
        """Get remaining tokens in current window."""
        current_time = time.time()
        cutoff = current_time - self._window_seconds
        
        # Remove old entries and sum recent usage
        recent_usage = 0
        new_usage = deque()
        for timestamp, tokens in self._token_usage:
            if timestamp > cutoff:
                new_usage.append((timestamp, tokens))
                recent_usage += tokens
        self._token_usage = new_usage
        
        return self.config.tokens_per_minute - recent_usage
    
    @property
    def requests_remaining(self) -> int:
        """Get remaining requests in current window."""
        current_time = time.time()
        self._cleanup_old_entries(self._request_times, current_time)
        return self.config.requests_per_minute - len(self._request_times)


class RateLimitedProvider:
    """
    Wrapper that adds rate limiting to any LLM provider.
    
    Usage:
        base_provider = OpenAIProvider(config)
        limited = RateLimitedProvider(
            base_provider,
            RateLimitConfig(requests_per_minute=60)
        )
        response = limited.invoke(prompt)
    """
    
    def __init__(
        self,
        provider: "LLMProvider",
        rate_config: RateLimitConfig
    ):
        self.provider = provider
        self.limiter = RateLimiter(rate_config)
    
    @property
    def provider_name(self) -> str:
        return f"rate_limited_{self.provider.provider_name}"
    
    @property
    def model_name(self) -> str:
        return self.provider.model_name
    
    def invoke(self, prompt: str, **kwargs) -> "LLMResponse":
        """Rate-limited synchronous invoke."""
        self.limiter.wait()
        response = self.provider.invoke(prompt, **kwargs)
        
        # Track token usage if available
        if hasattr(response, 'usage') and response.usage:
            total_tokens = response.usage.get('total_tokens', 0)
            self.limiter.record_tokens(total_tokens)
        
        return response
    
    async def ainvoke(self, prompt: str, **kwargs) -> "LLMResponse":
        """Rate-limited asynchronous invoke."""
        await self.limiter.await_wait()
        response = await self.provider.ainvoke(prompt, **kwargs)
        
        if hasattr(response, 'usage') and response.usage:
            total_tokens = response.usage.get('total_tokens', 0)
            self.limiter.record_tokens(total_tokens)
        
        return response


class RetryHandler:
    """
    Retry handler with exponential backoff.
    
    Usage:
        handler = RetryHandler(max_retries=3)
        response = await handler.execute(async_func, *args)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        retry_exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_multiplier = backoff_multiplier
        self.retry_exceptions = retry_exceptions
    
    async def execute(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        delay = self.base_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except self.retry_exceptions as e:
                last_exception = e
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.backoff_multiplier
        
        raise last_exception
