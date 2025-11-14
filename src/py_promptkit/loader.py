# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Prompt configuration loading utilities."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml  # type: ignore

from src.py_promptkit.errors import PromptConfigError
from src.py_promptkit.models.config import ModelConfig, PromptDefinition, ToolConfig


class PromptLoader:
    """Loads prompt definitions from a TOML configuration file."""

    def __init__(self, config_path: str | Path) -> None:
        """Initialize the PromptLoader.

        Args:
            config_path: str
        """
        self.config_path = Path(config_path).expanduser().resolve()
        self._definitions: Dict[str, PromptDefinition] = {}
        self._models: Dict[str, str] = {}
        self._providers: Dict[str, str] = {}
        self._temperatures: Dict[str, float] = {}

    @property
    def available_prompts(self) -> Iterable[str]:
        """Return a sorted tuple of available prompt names."""
        return tuple(sorted(self._definitions))

    def load(self) -> Dict[str, PromptDefinition]:
        """Load the configuration."""
        document = self._read_document()

        # All sections must be explicitly defined
        if "models" not in document:
            raise PromptConfigError("Missing [models] section.")
        if "providers" not in document:
            raise PromptConfigError("Missing [providers] section.")
        if "temperatures" not in document:
            self._temperatures = {}
        else:
            self._temperatures = document["temperatures"]

        self._models = document["models"]
        self._providers = document["providers"]

        reserved = {"models", "providers", "temperatures"}
        definitions: Dict[str, PromptDefinition] = {}

        for name, section in document.items():
            if name in reserved:
                continue
            definitions[name] = self._build_definition(name, section)

        if not definitions:
            raise PromptConfigError("No prompt sections found in configuration.")

        self._definitions = definitions
        return dict(self._definitions)

    def get(self, name: str) -> PromptDefinition:
        """Get a prompt defenition."""
        if name not in self._definitions:
            available = ", ".join(self.available_prompts)
            raise PromptConfigError(f"Prompt '{name}' not found. Available prompts: {available}")
        return self._definitions[name]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_document(self) -> Dict[str, Dict[str, Any]]:
        if not self.config_path.exists():
            raise PromptConfigError(f"Prompt file not found at {self.config_path}")
        try:
            with self.config_path.open("rb") as f:
                data = _toml.load(f)
        except Exception as exc:
            raise PromptConfigError(f"Failed to read TOML: {exc}") from exc
        if not isinstance(data, dict):
            raise PromptConfigError("TOML must decode to a mapping.")
        return data

    def _build_definition(self, name: str, section: dict[str, Any]) -> PromptDefinition:
        # Require explicit model & provider
        if name not in self._models:
            raise PromptConfigError(f"Prompt '{name}' missing entry in [models].")
        if name not in self._providers:
            raise PromptConfigError(f"Prompt '{name}' missing entry in [providers].")

        model_name = self._models[name]
        provider = self._providers[name]
        temperature = self._temperatures.get(name, 0.0)

        if "template" not in section:
            raise PromptConfigError(f"Prompt '{name}' missing 'template' key.")
        template = section["template"]

        if "structured" not in section:
            structured = False
        else:
            structured: bool = bool(section["structured"])
        schema_path = section.get("schema_path")

        # Require structured prompts to have schema_path
        if structured and not schema_path:
            raise PromptConfigError(f"Prompt '{name}' is structured but missing 'schema_path'.")

        tool_cfg: Optional[ToolConfig] = None
        raw_tool: MappingABC[str, Any] | None = section.get("tool", None)
        if raw_tool is not None:
            tool_cfg = ToolConfig(**raw_tool)

        model_cfg = ModelConfig(
            name=model_name,
            provider=provider,
            temperature=temperature,
            template=template,
            structured=structured,
            schema_path=schema_path,
            tool=tool_cfg,
        )

        return PromptDefinition(
            name=name,
            model=model_cfg,
            required_variables=model_cfg.expected_variables(),
        )
