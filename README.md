# async-toolformer-orchestrator

> Parallel tool execution for LLM agents — rate limiting, timeouts, and branch cancellation via pure `asyncio`.

When an LLM emits multiple tool calls in a single turn, there's no reason to run them sequentially. This library executes them concurrently, respects per-tool rate limits (token bucket), enforces hard timeouts, and lets you cancel entire speculative branches the moment a result is no longer useful.

## Concurrency model

```
LLM response → [tool_call_1, tool_call_2, tool_call_3, ...]
                      │              │              │
              asyncio.Task   asyncio.Task   asyncio.Task
                      │              │              │
               rate limiter   rate limiter   rate limiter
               (token bucket) (token bucket) (token bucket)
                      │              │              │
               timeout guard  timeout guard  timeout guard
                      └──────────────┴──────────────┘
                                    │
                              gather / stream
                                    │
                           [ToolResult, ...]
```

Sequential baseline for N independent tools: `sum(latencies)`.
Parallel: `max(latencies)`. Typical 3–5× speedup for realistic LLM workloads.

## Demo

```
$ python examples/demo.py

SEQUENTIAL   wall-clock: 3207 ms
  ✓ weather               301 ms
  ✓ stock_price           501 ms
  ✓ web_search            801 ms
  ...

PARALLEL     wall-clock: 802 ms   (4.0x speedup)
  ✓ weather               301 ms
  ✓ stock_price           501 ms
  ✓ web_search            801 ms
  ...

Branch cancellation demo
  ⚡ cancelled branch 'branch-A' at t+100ms
  web_search         ✗ (Branch cancelled)
  summarise          ✓
```

## Installation

```bash
pip install -e .                  # zero runtime dependencies
pip install -e ".[dev]"           # + pytest / ruff for development
```

Requires Python ≥ 3.10. No external packages at runtime — pure stdlib asyncio.

## Quick start

```python
import asyncio
from async_toolformer import AsyncOrchestrator, ToolCall, ToolRegistry

registry = ToolRegistry()

@registry.tool("Fetch weather data", timeout_s=5, calls_per_min=60)
async def weather(city: str) -> dict:
    ...  # your async implementation

@registry.tool("Run web search", timeout_s=8, calls_per_min=20)
async def web_search(query: str) -> list[str]:
    ...

async def main():
    orch = AsyncOrchestrator(registry)

    results = await orch.run([
        ToolCall("weather", {"city": "New York"}),
        ToolCall("weather", {"city": "London"}),
        ToolCall("web_search", {"query": "LLM tool calling"}),
    ])

    for r in results:
        print(r.tool_name, "→", r.output if r.ok else f"ERROR: {r.error}")

asyncio.run(main())
```

## Streaming results

```python
async for result in orch.stream(calls):
    # Arrives as each tool finishes — fastest tool first
    print(result)
```

## Branch cancellation

Useful for speculative execution: fire off multiple strategies in parallel, cancel the rest when the first good answer arrives.

```python
calls = [
    ToolCall("search", {"q": "fast path"}, branch_id="strategy-A"),
    ToolCall("search", {"q": "slow path"}, branch_id="strategy-A"),
    ToolCall("lookup",  {"id": "db-99"},   branch_id="strategy-B"),
]

# Cancel strategy-A mid-flight (e.g. on a timer or external signal)
orch.cancel_branch("strategy-A")

results = await orch.run(calls)
# strategy-A calls → ToolResult.error = "Branch cancelled"
# strategy-B       → runs to completion
```

## API reference

### `ToolRegistry`

| Method | Description |
|--------|-------------|
| `@registry.tool(description, *, timeout_s, calls_per_min, tags)` | Decorator to register an async function |
| `registry.register(ToolSpec(...))` | Register imperatively |
| `registry.get(name)` | Look up a tool; raises `ToolNotFoundError` |
| `registry.list_tools()` | All registered specs |
| `registry.schema()` | LLM-friendly list of dicts |

### `AsyncOrchestrator`

| Method | Description |
|--------|-------------|
| `await orch.run(calls, *, default_timeout_s)` | Execute in parallel, return ordered results |
| `async for r in orch.stream(calls)` | Yield results as they complete |
| `orch.cancel_branch(branch_id)` | Signal a branch to abort |
| `orch.reset_branch(branch_id)` | Clear the cancellation signal |

### `ToolCall`

```python
@dataclass
class ToolCall:
    tool_name: str
    kwargs: dict[str, Any]
    branch_id: str | None = None
```

### `ToolResult`

```python
@dataclass
class ToolResult:
    tool_name: str
    output: Any          # None on error
    latency_ms: float
    error: str | None    # None on success

    @property
    def ok(self) -> bool: ...
```

## Rate limiting

Each tool gets its own token bucket. Tokens refill continuously at `calls_per_min / 60` per second. The bucket starts full so the first burst of calls is never penalised.

```python
@registry.tool("Slow external API", calls_per_min=10)
async def my_tool(x: str) -> str: ...
```

If a tool has no `calls_per_min`, it runs without rate limiting.

## Running tests

```bash
pytest                         # all 22 tests
pytest -k "rate_limiter"       # targeted
```

## Project structure

```
src/async_toolformer/
├── __init__.py        # public API
├── orchestrator.py    # AsyncOrchestrator, ToolCall
├── tools.py           # ToolRegistry, ToolSpec, ToolResult
├── rate_limiter.py    # TokenBucket, RateLimiterRegistry
└── exceptions.py      # typed exceptions

tests/
├── test_orchestrator.py
├── test_rate_limiter.py
└── test_tools.py

examples/
└── demo.py            # parallel vs sequential + branch cancellation
```

## License

MIT
