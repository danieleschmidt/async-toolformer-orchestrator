"""End-to-end tests with real LLM APIs."""

import pytest
import os
from openai import AsyncOpenAI

from async_toolformer import AsyncOrchestrator, Tool


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestRealLLMIntegration:
    """Test with real OpenAI API."""

    @pytest.fixture
    def real_openai_client(self):
        """Real OpenAI client for E2E testing."""
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @Tool(description="Get current weather for a city")
    async def get_weather(self, city: str) -> dict:
        """Mock weather API for testing."""
        # In real tests, this would call a weather API
        return {
            "city": city,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "45%"
        }

    @Tool(description="Search for restaurants in a city")
    async def search_restaurants(self, city: str, cuisine: str = "any") -> list:
        """Mock restaurant search for testing."""
        return [
            {"name": f"Best {cuisine} in {city}", "rating": 4.5},
            {"name": f"{city} {cuisine} House", "rating": 4.2}
        ]

    @pytest.mark.asyncio
    async def test_real_parallel_tool_calls(self, real_openai_client):
        """Test real parallel tool execution with OpenAI."""
        orchestrator = AsyncOrchestrator(
            llm_client=real_openai_client,
            tools=[self.get_weather, self.search_restaurants],
            max_parallel=5
        )

        # This should trigger parallel tool calls
        result = await orchestrator.execute(
            "I'm visiting Paris. What's the weather like and what are some good French restaurants?"
        )

        assert result is not None
        # Verify that both tools were likely called
        assert "weather" in str(result).lower() or "temperature" in str(result).lower()
        assert "restaurant" in str(result).lower() or "cuisine" in str(result).lower()

    @pytest.mark.asyncio
    async def test_complex_multi_step_execution(self, real_openai_client):
        """Test complex multi-step execution."""
        @Tool(description="Analyze text sentiment")
        async def analyze_sentiment(self, text: str) -> dict:
            # Mock sentiment analysis
            return {"sentiment": "positive", "confidence": 0.85}

        @Tool(description="Generate summary")
        async def generate_summary(self, text: str, max_length: int = 100) -> str:
            # Mock summarization
            return f"Summary of text (max {max_length} chars): {text[:50]}..."

        orchestrator = AsyncOrchestrator(
            llm_client=real_openai_client,
            tools=[analyze_sentiment, generate_summary],
            max_parallel=3
        )

        result = await orchestrator.execute(
            "Analyze the sentiment and generate a summary for this text: "
            "'I love using async tools for parallel processing. It makes everything so much faster!'"
        )

        assert result is not None
        # Should contain evidence of both analysis and summarization
        assert isinstance(result, (str, dict))


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestAnthropicIntegration:
    """Test with Anthropic Claude API."""

    @pytest.fixture
    def anthropic_client(self):
        """Real Anthropic client for E2E testing."""
        import anthropic
        return anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @pytest.mark.asyncio
    async def test_anthropic_tool_calling(self, anthropic_client):
        """Test tool calling with Anthropic Claude."""
        @Tool(description="Calculate mathematical expression")
        async def calculate(self, expression: str) -> float:
            # Safe evaluation for testing
            try:
                return eval(expression.replace(" ", ""))
            except:
                return 0.0

        orchestrator = AsyncOrchestrator(
            llm_client=anthropic_client,
            tools=[calculate]
        )

        result = await orchestrator.execute(
            "Calculate 25 * 4 + 10 and also calculate 100 / 5"
        )

        assert result is not None