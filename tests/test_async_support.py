"""
Test Async Support

Tests for rate limiter, async adapter, and batch processing.
"""
import pytest
import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.rate_limiter import RateLimiter, RateLimitConfig, RetryHandler


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_allows_requests_under_limit(self):
        """Test requests allowed when under limit."""
        config = RateLimitConfig(requests_per_minute=100)
        limiter = RateLimiter(config)
        
        # Should not wait for first requests
        for _ in range(5):
            wait_time = limiter._get_wait_time()
            assert wait_time == 0
            limiter._request_times.append(time.time())
    
    def test_tracks_remaining_requests(self):
        """Test remaining requests tracking."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)
        
        assert limiter.requests_remaining == 10
        
        limiter._request_times.append(time.time())
        assert limiter.requests_remaining == 9


class TestRetryHandler:
    """Test retry handler with backoff."""
    
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test successful execution without retries."""
        handler = RetryHandler(max_retries=3)
        
        async def success_func():
            return "success"
        
        result = await handler.execute(success_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        handler = RetryHandler(max_retries=3, base_delay=0.01)
        attempts = [0]
        
        async def fail_then_succeed():
            attempts[0] += 1
            if attempts[0] < 3:
                raise Exception("Transient error")
            return "success"
        
        result = await handler.execute(fail_then_succeed)
        assert result == "success"
        assert attempts[0] == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test error raised after max retries."""
        handler = RetryHandler(max_retries=2, base_delay=0.01)
        
        async def always_fail():
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError):
            await handler.execute(always_fail)


class TestAsyncModelAdapter:
    """Test async model adapter."""
    
    @pytest.mark.asyncio
    async def test_batch_invoke(self):
        """Test batch processing with mock provider."""
        from broker.async_adapter import AsyncModelAdapter
        from interfaces.llm_provider import LLMResponse
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.ainvoke = AsyncMock(return_value=LLMResponse(
            content="Final Decision: do_nothing",
            model="test",
            usage={}
        ))
        
        adapter = AsyncModelAdapter(mock_provider)
        
        prompts = ["prompt1", "prompt2", "prompt3"]
        contexts = [{"agent_id": f"agent_{i}"} for i in range(3)]
        
        proposals = await adapter.batch_invoke(prompts, contexts, max_concurrent=2)
        
        assert len(proposals) == 3
        assert all(p.skill_name == "do_nothing" for p in proposals)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
