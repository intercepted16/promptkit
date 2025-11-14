## PromptKit

PromptKit is a lightweight, configuration-driven prompt orchestration library that separates prompt definitions from application code. By storing templates, model specifications, and tool configurations in TOML files, PromptKit enables teams to share and version prompt definitions independently while swapping LLM providers through a unified interface.

### Key Features

- **Configuration-first design**: Define prompts, models, providers, and temperatures in TOML files separate from code
- **Opt-in caching**: No default cache behavior—provide a `PromptCacheProtocol` implementation to `PromptRunner` for caching
- **Streaming support**: Stream LLM responses token-by-token using `PromptRunner.run_stream(...)`
- **Built-in LiteLLM integration**: Ships with a `LiteLLMClient` that provides access to 100+ LLM providers (OpenAI, Anthropic, Google, Azure, and more) via the [LiteLLM SDK](https://docs.litellm.ai/)
- **Tool-calling**: Define HTTP, stdio, or SSE tools in TOML; automatically passed to LLM for execution
- **Extensible hooks**: Observe and react to prompt lifecycle events with custom `PromptHook` implementations
- **Type-safe**: Built with Pydantic models for runtime validation and type safety

### Quick Start: TOML Configuration

PromptKit uses TOML files to define prompts and their execution parameters. Here's a minimal example:

```toml
# prompts.toml
[models]
welcome = "demo/unit-test"

[providers]
welcome = "demo"

[temperatures]
welcome = 0.1

[welcome]
template = "Hello {name}, welcome to {product}!"
```

**Configuration sections:**
- `[models]`: Maps prompt names to model identifiers (format varies by provider)
- `[providers]`: Maps prompt names to provider keys (registered with `PromptRunner`)
- `[temperatures]`: Maps prompt names to sampling temperatures (0.0–2.0)
- `[prompt_name]`: Individual prompt sections containing the template and optional tool configurations

### Basic Usage: Echo Client Example

Here's a minimal example using a custom echo client:

```python
from src.py_promptkit import PromptLoader, PromptRunner
from src.py_promptkit.models.clients import LLMResponse, ToolSpecification


class EchoClient:
    """Simple echo client for testing.

    This implements the `LLMClient` protocol (no subclassing required).
    Clients are registered with `PromptRunner` as instances; the runner
    will pass the prompt's `model` and `temperature` to `generate`/
    `stream_generate` at call time.
    """

    supports_tools = False

    def generate(
            self,
            prompt: str,
            tools: list[ToolSpecification] | None = None,
            model: str | None = None,
            temperature: float | None = None,
    ) -> LLMResponse:
        return {"reasoning": "echo", "output": prompt}

    def stream_generate(
            self,
            prompt: str,
            tools: list[ToolSpecification] | None = None,
            model: str | None = None,
            temperature: float | None = None,
    ):
        yield prompt


# Load configuration
loader = PromptLoader("prompts.toml")
loader.load()

# Create runner (no caching by default)
runner = PromptRunner(loader)

# Register client instance for the "demo" provider
# Note: PromptRunner registers client *instances* directly (the older
# ClientFactory pattern was removed). The runner will call the instance
# methods and provide model/temperature at call time.
runner.register_client("demo", EchoClient())

# Execute prompt
result = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})
print(result["output"])  # Output: Hello Ada, welcome to PromptKit!
```

### Streaming Responses

Stream tokens as they arrive from the LLM:

```python
chunks = runner.run_stream("welcome", {"name": "Ada", "product": "PromptKit"})
for chunk in chunks:
    print(chunk, end="", flush=True)
```

### Using the Built-in LiteLLM Client

PromptKit includes a `LiteLLMClient` adapter that wraps the [LiteLLM SDK](https://docs.litellm.ai/), providing unified access to 100+ LLM providers including:

- **OpenAI**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, o1, o3-mini, and more
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro
- **Azure OpenAI**: Enterprise-grade OpenAI models
- **Open-source models**: Via providers like Together AI, Replicate, Hugging Face, Ollama
- **And many more**: See the [full provider list](https://docs.litellm.ai/docs/providers)

#### Installation

The LiteLLM client requires the `litellm` package:

```bash
pip install litellm
```

#### Basic LiteLLM Usage

```python
from src.py_promptkit import PromptLoader, PromptRunner
from src.py_promptkit.litellm.core import LiteLLMClient

# Load prompts
loader = PromptLoader("prompts.toml")
loader.load()

# Configure secrets for LiteLLM
secrets = {
    "OPENAI_API_KEY": "sk-...",
    # "ANTHROPIC_API_KEY": "...",
    # "GOOGLE_API_KEY": "...",
}

# Use context manager for automatic cleanup
with PromptRunner(loader) as runner:
    runner.register_client("openai", LiteLLMClient(secrets=secrets))

    result = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})
    print(result["output"])
```
```

#### Model Naming Conventions

LiteLLM uses provider-specific model identifiers. Here are some examples:

**OpenAI models:**
```python
"gpt-4o"              # Latest GPT-4o
"gpt-4o-mini"         # Smaller, faster GPT-4o variant
"gpt-4-turbo"         # GPT-4 Turbo
"gpt-3.5-turbo"       # GPT-3.5 Turbo
"o1"                  # OpenAI o1 reasoning model
"o3-mini"             # OpenAI o3-mini
```

**Anthropic models:**
```python
"claude-3-5-sonnet-20241022"   # Claude 3.5 Sonnet (latest)
"claude-3-opus-20240229"       # Claude 3 Opus
"claude-3-haiku-20240307"      # Claude 3 Haiku
```

**Google models:**
```python
"gemini-2.0-flash-exp"         # Gemini 2.0 Flash (experimental)
"gemini-1.5-pro"               # Gemini 1.5 Pro
```

**Azure OpenAI:**
```python
"azure/gpt-4o"                 # Azure-hosted GPT-4o
```

For a complete list of supported models and naming conventions, see the [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers).

#### TOML Configuration for LiteLLM

```toml
# prompts.toml
[models]
chat = "gpt-4o-mini"                    # OpenAI model
code_gen = "claude-3-5-sonnet-20241022" # Anthropic model

[providers]
chat = "openai"
code_gen = "anthropic"

[temperatures]
chat = 0.7
code_gen = 0.3

[chat]
template = "You are a helpful assistant. {user_message}"

[code_gen]
template = "Write clean, documented code for: {task}"
```

#### Streaming with LiteLLM

```python
with PromptRunner(loader) as runner:
    runner.register_client("openai", LiteLLMClient(secrets=secrets))
    
    for chunk in runner.run_stream("chat", {"user_message": "Explain quantum computing"}):
        print(chunk, end="", flush=True)
```

### Tool-Calling

PromptKit supports automatic tool execution when tools are defined in your TOML configuration. The `PromptLoader` reads tool specifications and passes them to the LLM client during execution.

#### Defining Tools in TOML

Tools are configured per-prompt in the TOML file:

```toml
[models]
assistant = "gpt-4o"

[providers]
assistant = "openai"

[temperatures]
assistant = 0.7

[assistant]
template = "Help the user with: {request}"

# Tool configuration for this prompt
[assistant.tool]
type = "http"
url = "https://api.example.com/tools/calculator"
name = "calculator"
description = "Performs mathematical calculations"
parameters = '{"type": "object", "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate"}}, "required": ["expression"]}'
```

When you run this prompt, the tool specification is automatically passed to the client:

```python
from src.py_promptkit import PromptLoader, PromptRunner
from src.py_promptkit.litellm.core import LiteLLMClient

loader = PromptLoader("prompts.toml")
loader.load()

with PromptRunner(loader) as runner:
    runner.register_client("openai", LiteLLMClient(secrets={"OPENAI_API_KEY": "sk-..."}))

    # Tools from TOML are automatically used
    result = runner.run("assistant", {"request": "What is 25 * 4?"})
    print(result["output"])
```

#### Supported Tool Types

**HTTP Tools** (`type = "http"`):
- Makes POST requests to the specified URL
- Sends tool arguments as JSON in request body
- Returns response text to the LLM

**MCP Tools** (`type = "stdio"` or `type = "sse"`):
- Connects to MCP (Model Context Protocol) servers
- `stdio`: Uses standard I/O for local MCP server executables
- `sse`: Uses Server-Sent Events for remote MCP servers

#### MCP Tool Configuration

For MCP tools, you need to initialize MCP clients when creating the `LiteLLMClient`:

```python
from src.py_promptkit.litellm.core import LiteLLMClient

# Initialize MCP client connections
mcp_tools = [
    {
        "name": "file_reader",
        "type": "stdio",
        "url": "./tools/file_reader"  # Path to MCP server executable
    }
]

with PromptRunner(loader) as runner:
    runner.register_client("openai", LiteLLMClient(mcp_tools=mcp_tools, secrets=secrets))

    # Use prompt that references the MCP tool
    result = runner.run("file_assistant", {"filename": "data.json"})
```

Then in your TOML:

```toml
[file_assistant]
template = "Read and summarize the file: {filename}"

[file_assistant.tool]
type = "stdio"
name = "file_reader"
description = "Reads file contents"
parameters = '{"type": "object", "properties": {"path": {"type": "string"}}}'
```

#### Tool Parameters Format

Tool parameters should be a JSON schema (as a string or dict):

```toml
# As a JSON string
parameters = '{"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}'

# Or as a TOML table (will be converted)
[my_prompt.tool.parameters]
type = "object"
[my_prompt.tool.parameters.properties.query]
type = "string"
description = "Search query"
```

**Note:** Tool-calling support varies by model. The `LiteLLMClient` uses pattern matching to detect tool-capable models (e.g., GPT-4, GPT-4o, Claude 3+, Gemini Pro). See the [LiteLLM documentation](https://docs.litellm.ai/docs/completion/function_call) for model-specific support.

### Extension Points

#### Custom LLM Clients

Implement the `LLMClient` protocol to integrate any LLM provider:

```python
from src.py_promptkit.models.clients import LLMClient, LLMResponse, ToolSpecification
from typing import Iterator


class MyCustomClient:
    """Example custom client implementing the `LLMClient` protocol.

    Note: implement the same method signatures as the `LLMClient` protocol.
    Register client instances directly with `PromptRunner`.
    """

    supports_tools = False  # Set to True if your client supports tool-calling

    def generate(
            self,
            prompt: str,
            tools: list[ToolSpecification] | None = None,
            model: str | None = None,
            temperature: float | None = None,
    ) -> LLMResponse:
        # Your custom LLM API logic here
        response_text = call_my_llm_api(prompt, model, temperature)
        return {"reasoning": "", "output": response_text}

    def stream_generate(
            self,
            prompt: str,
            tools: list[ToolSpecification] | None = None,
            model: str | None = None,
            temperature: float | None = None,
    ) -> Iterator[str]:
        # Stream tokens from your LLM
        for token in stream_my_llm_api(prompt, model, temperature):
            yield token


# Register your custom client instance
runner.register_client("my_provider", MyCustomClient())
```

#### Hooks for Observability

Implement `PromptHook` to observe or modify prompt execution:

```python
from src.py_promptkit.models.hooks import PromptHook, HookContext
from src.py_promptkit.models.clients import LLMResponse


class LoggingHook(PromptHook):
    def before_run(self, context: HookContext) -> None:
        print(f"Running prompt: {context.prompt_name}")
        print(f"Model: {context.model.name}")
        print(f"Variables: {context.variables}")

    def after_run(self, context: HookContext, response: LLMResponse) -> None:
        print(f"Completed: {context.prompt_name}")
        print(f"Output length: {len(response['output'])} chars")

    def on_error(self, context: HookContext, error: Exception) -> None:
        print(f"Error in {context.prompt_name}: {error}")


# Register hooks when creating the runner
runner = PromptRunner(loader, hooks=[LoggingHook()])
```

**Hook use cases:**
- Logging and observability
- Cost tracking and usage metrics
- Custom validation or transformation
- Integration with monitoring systems
- A/B testing and experimentation

### Caching (Opt-In)

PromptKit does **not** cache responses by default. To enable caching, implement the `PromptCacheProtocol` and pass it to `PromptRunner`.

#### Cache Protocol

```python
from typing import Protocol, Mapping

class PromptCacheProtocol(Protocol):
    """Protocol for custom cache implementations."""
    
    def build_key(
        self,
        prompt: str,
        model_name: str,
        provider: str,
        temperature: float,
        variables: Mapping[str, str],
    ) -> str:
        """Generate a deterministic cache key."""
        ...
    
    def get(self, key: str) -> str | None:
        """Retrieve cached value if present."""
        ...
    
    def set(self, key: str, value: str) -> None:
        """Store a value in the cache."""
        ...
```

#### Example: In-Memory Cache

```python
import json
import hashlib
from typing import Mapping

class InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
    
    def build_key(
        self,
        prompt: str,
        model_name: str,
        provider: str,
        temperature: float,
        variables: Mapping[str, str],
    ) -> str:
        # Create deterministic hash from all inputs
        payload = {
            "prompt": prompt,
            "model": model_name,
            "provider": provider,
            "temperature": round(temperature, 3),
            "variables": dict(sorted(variables.items())),
        }
        content = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()
    
    def get(self, key: str) -> str | None:
        return self._store.get(key)
    
    def set(self, key: str, value: str) -> None:
        self._store[key] = value

# Use the cache
cache = InMemoryCache()
runner = PromptRunner(loader, cache=cache)

# First call executes the LLM
result1 = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})

# Second call with same inputs returns cached result
result2 = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})
```

#### Disabling Cache Per-Request

```python
# Bypass cache for a specific request
result = runner.run("welcome", {"name": "Ada"}, use_cache=False)
```

**Note:** Streaming via `run_stream()` never uses caching, even when a cache is configured.

### Advanced Configuration

#### Structured Output

Configure prompts to expect structured JSON responses:

```toml
[models]
data_extractor = "gpt-4o"

[providers]
data_extractor = "openai"

[temperatures]
data_extractor = 0.0

[data_extractor]
template = "Extract structured data from: {text}"
structured = true
schema_path = "schemas/extraction.json"
```

The `schema_path` should point to a JSON schema file that defines the expected output structure.

#### Multiple Providers

Configure different prompts to use different providers:

```toml
[models]
chat = "gpt-4o-mini"
analysis = "claude-3-5-sonnet-20241022"
embedding = "text-embedding-3-small"

[providers]
chat = "openai"
analysis = "anthropic"
embedding = "openai"

[temperatures]
chat = 0.7
analysis = 0.3
embedding = 0.0

[chat]
template = "Chat with the user about: {topic}"

[analysis]
template = "Analyze this code:\n\n{code}\n\nProvide suggestions for improvement."

[embedding]
template = "{text}"
```

Register multiple clients:

```python
from src.py_promptkit.litellm.core import LiteLLMClient

openai_secrets = {"OPENAI_API_KEY": "sk-..."}
anthropic_secrets = {"ANTHROPIC_API_KEY": "sk-ant-..."}

with PromptRunner(loader) as runner:
    runner.register_client("openai", LiteLLMClient(secrets=openai_secrets))
    runner.register_client("anthropic", LiteLLMClient(secrets=anthropic_secrets))

    # Use different models for different tasks
    chat_result = runner.run("chat", {"topic": "AI safety"})
    analysis_result = runner.run("analysis", {"code": "def factorial(n): ..."})
```

### API Reference

#### Core Classes

**PromptLoader**
- `__init__(config_path: str | Path)`: Initialize with path to TOML configuration
- `load() -> Dict[str, PromptDefinition]`: Load and validate all prompt definitions
- `get(name: str) -> PromptDefinition`: Retrieve a specific prompt definition
- `available_prompts: Iterable[str]`: List of all available prompt names

**PromptRunner**
- `__init__(loader: PromptLoader, *, hooks: Sequence[PromptHook] | None = None, cache: PromptCacheProtocol | None = None)`: Initialize runner
- `register_client(provider: str, client: LLMClient) -> None`: Register an LLM client instance for a provider (pass a ready-to-use client object)
- `run(prompt_name: str, variables: Mapping[str, object] | None = None, *, tools: Sequence[ToolSpecification] | None = None, use_cache: bool = True) -> LLMResponse`: Execute a prompt
- `run_stream(prompt_name: str, variables: Mapping[str, object] | None = None, *, tools: Sequence[ToolSpecification] | None = None) -> Iterator[str]`: Stream prompt execution
- `close() -> None`: Close all registered LLM clients
- Context manager support: Use `with PromptRunner(loader) as runner:` for automatic cleanup

**LiteLLMClient**
- `__init__(mcp_tools: list[dict[str, Any]] | None = None, secrets: dict[str, str | None] | None = None, verbose: bool = False)`: Initialize LiteLLM client
- `close() -> None`: Close all MCP clients and clean up resources
- Context manager support: Use `with LiteLLMClient(...) as client:` for automatic cleanup

#### Type Definitions

**LLMResponse** (TypedDict)
```python
{
    "reasoning": str,  # Model's reasoning or chain-of-thought (if available)
    "output": str      # Final response text
}
```

**ToolSpecification** (TypedDict)
```python
{
    "name": str,                    # Tool identifier
    "description": str,             # Tool description for the LLM
    "parameters": Dict[str, Any],   # JSON schema for tool parameters
    "type": str,                    # Transport type: "stdio", "sse", or "http"
    "url": str                      # Tool endpoint URL or path
}
```

**HookContext** (dataclass)
```python
@dataclass(frozen=True)
class HookContext:
    prompt_name: str                           # Name of the prompt being executed
    model: ModelConfig                         # Model configuration
    variables: Mapping[str, str]               # Rendered variables
    rendered_prompt: str                       # Final prompt after template rendering
    tools: Sequence[ToolSpecification] | None  # Tools available for this execution
```

### Error Handling

PromptKit defines specific exception types for different failure modes:

```python
from src.py_promptkit.errors import (
    PromptKitError,  # Base exception
    PromptConfigError,  # Configuration/TOML parsing errors
    PromptValidationError,  # Variable validation errors
    PromptProviderError,  # Provider/client registration errors
    ModelRequestError,  # LLM API request failures (LiteLLM)
    MCPError  # MCP tool execution errors
)

try:
    result = runner.run("my_prompt", {"var": "value"})
except PromptValidationError as e:
    print(f"Missing or invalid variable: {e}")
except PromptProviderError as e:
    print(f"Provider not registered: {e}")
except ModelRequestError as e:
    print(f"LLM request failed: {e}")
```

### Installation

```bash
# Basic installation
pip install py_promptkit

# With LiteLLM support
pip install litellm

# Development installation
git clone https://github.com/yourusername/promptkit.git
cd py_promptkit
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- Pydantic >= 1.10
- tomli >= 2.0.1 (Python < 3.11)
- typing-extensions >= 4.8.0

Optional dependencies:
- litellm (for LiteLLMClient and 100+ provider support)

### Reference Links

- **LiteLLM Documentation**: https://docs.litellm.ai/
- **LiteLLM Providers**: https://docs.litellm.ai/docs/providers
- **Model Context Protocol (MCP)**: https://modelcontextprotocol.io/
- **OpenAI Models**: https://platform.openai.com/docs/models
- **Anthropic Models**: https://docs.anthropic.com/
- **Google AI Models**: https://ai.google.dev/models

### License

MIT License - see LICENSE file for details.

---

**PromptKit** — Configuration-driven prompt orchestration for LLM applications.