"""Tests for TokenBucket rate limiter."""

import asyncio
import time
import pytest

from async_toolformer.rate_limiter import TokenBucket, RateLimiterRegistry


@pytest.mark.asyncio
async def test_token_bucket_immediate_when_full():
    """Starts full — first acquire should be instant."""
    bucket = TokenBucket(calls_per_min=60)
    t0 = time.monotonic()
    await bucket.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"Expected instant acquire, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_token_bucket_rate_limited():
    """
    With calls_per_min=60, tokens refill at 1/s.
    Drain the bucket, then time the next acquire — should be ~1s.
    """
    bucket = TokenBucket(calls_per_min=2)  # 2 calls/min = 1 token per 30s (fast for test)
    # Actually use a very fast rate for testing: 120 calls/min = 2/s = 0.5s per token
    bucket = TokenBucket(calls_per_min=120)  # 2 tokens/sec

    # Drain all tokens (starts at capacity=120, so drain many)
    # Force tokens to 0 by manipulating internal state
    bucket._tokens = 0.0

    t0 = time.monotonic()
    await bucket.acquire()
    elapsed = time.monotonic() - t0

    # Should have waited ~0.5s (1 token at 2 tokens/sec)
    assert 0.3 < elapsed < 1.5, f"Expected ~0.5s wait, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_registry_no_op_for_unregistered():
    """acquire() on an unregistered tool should return immediately."""
    reg = RateLimiterRegistry()
    t0 = time.monotonic()
    await reg.acquire("unknown_tool")  # should not raise or block
    assert time.monotonic() - t0 < 0.05


@pytest.mark.asyncio
async def test_registry_limits_registered_tool():
    """Registered tool gets a real bucket."""
    reg = RateLimiterRegistry()
    reg.register("fast", calls_per_min=120)  # 2/sec

    bucket = reg.get("fast")
    assert bucket is not None
    assert bucket.calls_per_min == 120


@pytest.mark.asyncio
async def test_concurrent_acquires_serialised():
    """Multiple coroutines acquiring the same bucket are serialised."""
    bucket = TokenBucket(calls_per_min=600)  # 10 tokens/sec
    bucket._tokens = 3.0  # Give exactly 3 tokens

    order = []

    async def worker(n: int):
        await bucket.acquire()
        order.append(n)

    await asyncio.gather(*[worker(i) for i in range(3)])
    assert sorted(order) == [0, 1, 2]
