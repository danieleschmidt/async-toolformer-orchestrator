"""Shared fixtures for the test suite."""

import asyncio
import pytest
from async_toolformer import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
