"""Configuration models and helpers for PromptKit."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, ValidationInfo, field_validator  # type: ignore lib constraint

from src.py_promptkit.errors import PromptConfigError, PromptValidationError
from src.py_promptkit.models.clients import ToolSpecification


def _ensure_str_key(raw_key: Any, context: str) -> str:
    if not isinstance(raw_key, str):
        raise PromptConfigError(f"{context} keys must be strings.")
    return raw_key


def _clean_string(raw: Any, field_name: str) -> str:
    if raw is None:
        return ""
    value = str(raw).strip()
    if "\x00" in value:
        raise PromptConfigError(f"{field_name} contains a null byte, which is not allowed.")
    return value


class ToolConfig(BaseModel):
    """Normalized configuration for an optional external tool."""

    type: str = Field(default="http")
    url: str = Field(default="")
    name: str = Field(default="external_tool")
    description: str = Field(default="Generic external connector.")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        allowed = {"http", "stdio", "sse"}
        cleaned = _clean_string(value, "tool.type").lower()
        if cleaned not in allowed:
            raise PromptConfigError(f"Unsupported tool type '{value}'. Allowed values: {', '.join(sorted(allowed))}.")
        return cleaned

    @field_validator("url", "name", "description", mode="before")
    @classmethod
    def _validate_text_fields(cls, value: Any, info: ValidationInfo) -> str:
        field_name = info.field_name or "tool.field"
        cleaned = _clean_string(value, f"tool.{field_name}")
        tool_type = info.data.get("type")
        if (
            tool_type in ("sse", "http")
            and field_name == "url"
            and not cleaned.startswith(
                (
                    "http://",
                    "https://",
                )
            )
        ):
            raise PromptConfigError("tool.url must be an HTTP(S) address if using 'sse' or 'http' tool types.")
        return cleaned

    @field_validator("parameters", mode="before")
    @classmethod
    def _coerce_parameters(cls, value: Optional[Dict[str, Any] | str]) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, str):
            return cls._parse_parameters_string(value)
        result: Dict[str, Any] = {}
        for key_val, item_val in value.items():
            result[key_val] = item_val
        return result

    def as_tool_spec(self) -> ToolSpecification:
        """Return the configuration as a ToolSpecification understood by PromptRunner."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "type": self.type,
            "url": self.url,
        }

    @staticmethod
    def _parse_parameters_string(raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        if not cleaned:
            return {}
        try:
            decoded: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PromptConfigError(f"tool.parameters must contain valid JSON: {exc}") from exc

        result: Dict[str, Any] = {}
        for key in decoded.keys():
            str_key = _ensure_str_key(key, "tool.parameters")
            result[str_key] = decoded[key]
        return result


class ModelConfig(BaseModel):
    """Container describing how a prompt should be executed."""

    name: str
    provider: str = Field(default="openai")
    temperature: float = Field(default=0.0)
    template: str = Field(default="")
    structured: bool = Field(default=False)
    schema_path: Optional[str] = Field(default=None)
    tool: Optional[ToolConfig] = Field(default=None)

    @field_validator("name", "provider", mode="before")
    @classmethod
    def _sanitize_simple_strings(cls, value: Any, info: ValidationInfo) -> str:
        field_name = info.field_name or "value"
        cleaned = _clean_string(value, field_name)
        if field_name == "provider":
            if not re.fullmatch(r"[A-Za-z0-9_\-./]+", cleaned):
                raise PromptConfigError(
                    "provider must contain only letters, numbers, dashes, underscores, dots, or slashes."
                )
        return cleaned

    @field_validator("temperature", mode="before")
    @classmethod
    def _coerce_temperature(cls, value: Any) -> float:
        if value is None:
            return 0.0
        try:
            temp = float(value)
        except (TypeError, ValueError) as exc:
            raise PromptConfigError("temperature must be numeric.") from exc
        if temp < 0.0 or temp > 2.0:
            raise PromptConfigError("temperature must be between 0.0 and 2.0 inclusive.")
        return temp

    @field_validator("template", mode="before")
    @classmethod
    def _ensure_template(cls, value: Any) -> str:
        cleaned = _clean_string(value, "template")
        if cleaned == "":
            raise PromptConfigError("template must not be empty.")
        return cleaned

    @field_validator("schema_path", mode="before")
    @classmethod
    def _sanitize_schema_path(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = _clean_string(value, "schema_path")
        if cleaned:
            normalized = Path(cleaned).as_posix()
            if ".." in Path(normalized).parts:
                raise PromptConfigError("schema_path must not traverse parent directories.")
            return normalized
        return None

    def expected_variables(self) -> Tuple[str, ...]:
        """Return the set of required template variables."""
        extractor = _TemplateVariablesExtractor(self.template)
        return extractor.required_variables

    def tool_specifications(self) -> Optional[Sequence[ToolSpecification]]:
        """Return the tool specification, if one was configured."""
        if self.tool is None:
            return None
        return [self.tool.as_tool_spec()]


_ALLOWED_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _TemplateVariablesExtractor:
    """Helper that inspects templates to find referenced variables."""

    def __init__(self, template: str) -> None:
        self.template = template
        self.required_variables = self._collect_variables()

    def _collect_variables(self) -> Tuple[str, ...]:
        formatter = Formatter()
        names: Set[str] = set()
        for _, field_name, _, _ in formatter.parse(self.template):
            if field_name is None or field_name == "":
                continue
            simple_name = field_name.split(".")[0]
            if not _ALLOWED_VARIABLE.fullmatch(simple_name):
                raise PromptConfigError(
                    f"Template variable '{field_name}' is not allowed. Use simple snake_case names only."
                )
            names.add(simple_name)
        return tuple(sorted(names))


@dataclass(frozen=True)
class PromptDefinition:
    """Materialized prompt definition produced by the loader."""

    name: str
    model: ModelConfig
    required_variables: Tuple[str, ...]

    def render(self, variables: Mapping[str, Any]) -> str:
        """Render the prompt after validating input variables."""
        rendered, _ = self.render_with(variables)
        return rendered

    def render_with(self, variables: Mapping[str, Any]) -> Tuple[str, Dict[str, str]]:
        """Render the prompt and return both text and sanitized variables."""
        safe_variables = self.normalize_variables(variables)
        try:
            rendered = self.model.template.format(**safe_variables)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise PromptValidationError(
                f"Prompt '{self.name}' is missing variable '{missing_key}' at render time."
            ) from exc
        return rendered, safe_variables

    def normalize_variables(self, variables: Mapping[str, Any]) -> Dict[str, str]:
        """Return sanitized template variables ensuring required keys exist."""
        missing = [key for key in self.required_variables if key not in variables]
        if missing:
            joined = ", ".join(missing)
            raise PromptValidationError(f"Prompt '{self.name}' requires variables: {joined}.")

        safe_variables: Dict[str, str] = {}
        for key in self.required_variables:
            value = variables.get(key)
            safe_variables[key] = self._sanitize_variable_value(key, value)

        return safe_variables

    @staticmethod
    def _sanitize_variable_value(key: str, value: Any) -> str:
        if value is None:
            raise PromptValidationError(f"Variable '{key}' must not be None.")
        if isinstance(value, (str, int, float, bool)):
            sanitized_value = str(value)
        else:
            raise PromptValidationError(f"Variable '{key}' must be a string, int, float, or bool.")
        if "\x00" in sanitized_value:
            raise PromptValidationError(f"Variable '{key}' contains a null byte.")
        return sanitized_value

    def build_tools(self) -> Optional[Sequence[ToolSpecification]]:
        """Return tool specifications declared for this prompt."""
        return self.model.tool_specifications()
