## PromptKit

PromptKit is a lightweight prompt orchestration library that turns TOML configuration files into ready-to-run LLM prompts. It keeps template, provider, and tool metadata outside of application code so multiple services can share the same prompt definitions while swapping in custom LLM clients.

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

```python
from promptkit import PromptLoader, PromptRunner
from promptkit.clients import LLMClient, LLMResponse, ToolSpecification

class EchoClient(LLMClient):
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.supports_tools = False

    def generate(self, prompt: str, tools: list[ToolSpecification] | None = None) -> LLMResponse:
        return {"reasoning": "echo", "output": prompt}

    def stream_generate(self, prompt: str, tools: list[ToolSpecification] | None = None):
        yield prompt

loader = PromptLoader("prompts.toml")
loader.load()

runner = PromptRunner(loader)

# Register a factory that creates client instances with model/temperature from the prompt config
def echo_factory(model: str, temperature: float) -> LLMClient:
    return EchoClient(model=model, temperature=temperature)

runner.register_client("demo", echo_factory)

result = runner.run("welcome", {"name": "Ada", "product": "PromptKit"})
print(result["output"])  # Hello Ada, welcome to PromptKit!
print(result["reasoning"])  # Model reasoning if provided
```

### Extension points

- **Hooks** – Implement `PromptHook` to observe or modify prompt runs (`before_run`, `after_run`, `on_error`).
- **Clients** – Provide any `LLMClient` adapter; register by provider name and override on demand.
- **Tools** – Attach tool metadata in TOML; PromptRunner validates that the chosen client advertises `supports_tools`.
