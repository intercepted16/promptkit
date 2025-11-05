# library constraints; thus ignore
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Prompt configuration loading utilities."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, cast

try:  # pragma: no cover - python >=3.11
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - python <3.11
    import tomli as _toml  # type: ignore

from src.promptkit.errors import PromptConfigError
from src.promptkit.models.config import ModelConfig, PromptDefinition, ToolConfig


def _ensure_mapping_key(section_name: str, raw_key: Any) -> str:
    if not isinstance(raw_key, str):
        raise PromptConfigError(f"Keys in section '{section_name}' must be strings.")
    return raw_key


class PromptLoader:
    """Loads and validates prompt definitions from TOML configuration files."""

    def __init__(self, config_path: str | Path) -> None:
        """Create a loader for the provided TOML configuration path."""
        self.config_path = Path(config_path).expanduser().resolve()
        self._definitions: MutableMapping[str, PromptDefinition] = {}
        self._models: Dict[str, str] = {}
        self._providers: Dict[str, str] = {}
        self._temperatures: Dict[str, float] = {}

    @property
    def available_prompts(self) -> Iterable[str]:
        """Return prompt names that were loaded."""
        return tuple(sorted(self._definitions))

    def load(self) -> Dict[str, PromptDefinition]:
        """Parse the TOML file and return prompt definitions."""
        document = self._read_document()
        self._models = self._coerce_string_map(document.get("models", {}), "models")
        self._providers = self._coerce_string_map(
            document.get("providers", {}), "providers"
        )
        self._temperatures = self._coerce_float_map(
            document.get("temperatures", {}), "temperatures"
        )

        reserved = {"models", "providers", "temperatures"}

        definitions: Dict[str, PromptDefinition] = {}
        for key, raw_section in document.items():
            if key in reserved:
                continue
            section_mapping = self._section_as_mapping(raw_section, key)
            definitions[key] = self._build_definition(key, section_mapping)

        if not definitions:
            raise PromptConfigError(
                "No prompt sections were found in the configuration file."
            )

        self._definitions = definitions
        return dict(self._definitions)

    def get(self, name: str) -> PromptDefinition:
        """Return the definition matching the given name."""
        if name not in self._definitions:
            available = ", ".join(self.available_prompts)
            raise PromptConfigError(
                f"Prompt '{name}' is not defined. Available prompts: {available}"
            )
        return self._definitions[name]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_document(self) -> Dict[str, object]:
        if not self.config_path.exists():
            raise PromptConfigError(
                f"Prompt configuration not found at {self.config_path}"
            )
        try:
            with self.config_path.open("rb") as handle:
                data = _toml.load(handle)
        except Exception as exc:
            raise PromptConfigError(
                f"Failed to read TOML configuration: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise PromptConfigError("TOML configuration must decode to a mapping.")
        return data

    @staticmethod
    def _coerce_string_map(raw_section: object, section_name: str) -> Dict[str, str]:
        pairs = PromptLoader._iter_section_items(raw_section, section_name)
        result: Dict[str, str] = {}
        for key, value in pairs:
            if value is None:
                continue
            result[key.strip()] = str(value).strip()
        return result

    @staticmethod
    def _coerce_float_map(raw_section: object, section_name: str) -> Dict[str, float]:
        pairs = PromptLoader._iter_section_items(raw_section, section_name)
        result: Dict[str, float] = {}
        for key, value in pairs:
            if value is None:
                continue
            result[key.strip()] = PromptLoader._to_float(value, key)
        return result

    @staticmethod
    def _iter_section_items(
        raw_section: object, section_name: str
    ) -> List[tuple[str, object]]:
        if raw_section is None:
            return []
        if isinstance(raw_section, dict):
            return [
                (_ensure_mapping_key(section_name, key), value)
                for key, value in raw_section.items()
            ]
        if isinstance(raw_section, MappingABC):
            return [
                (_ensure_mapping_key(section_name, key), value)
                for key, value in raw_section.items()
            ]
        raise PromptConfigError(f"Section '{section_name}' must be a mapping.")

    @staticmethod
    def _section_as_mapping(
        raw_section: object, section_name: str
    ) -> Dict[str, object]:
        pairs = PromptLoader._iter_section_items(raw_section, section_name)
        return dict(pairs)

    @staticmethod
    def _to_float(value: object, key: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return 0.0
            try:
                return float(stripped)
            except ValueError as exc:
                raise PromptConfigError(
                    f"Temperature for '{key}' must be numeric."
                ) from exc
        raise PromptConfigError(f"Temperature for '{key}' must be numeric.")

    def _build_definition(
        self, name: str, section: Mapping[str, object]
    ) -> PromptDefinition:
        model_name = self._models.get(name)
        if not model_name:
            raise PromptConfigError(
                f"Prompt '{name}' is missing a model entry in [models]."
            )
        provider = self._providers.get(name, "openai")
        temperature = self._temperatures.get(name, 0.0)

        raw_tool = section.get("tool")
        tool_cfg: Optional[ToolConfig] = None
        if isinstance(raw_tool, MappingABC):
            # cast to object so the static type of the argument matches _iter_section_items(raw_section: object, ...)
            tool_pairs = PromptLoader._iter_section_items(
                cast(object, raw_tool), f"{name}.tool"
            )
            tool_data: Dict[str, Any] = dict(tool_pairs)
            tool_cfg = ToolConfig(**tool_data)

        section_data = dict(section)
        template_value: Any = section_data.get("template", "")
        structured_value = bool(section_data.get("structured", False))
        schema_value: Any = section_data.get("schema_path")

        model_cfg = ModelConfig(
            name=model_name,
            provider=provider,
            temperature=temperature,
            template=template_value,
            structured=structured_value,
            schema_path=schema_value,
            tool=tool_cfg,
        )

        if model_cfg.structured and model_cfg.schema_path is None:
            raise PromptConfigError(
                f"Prompt '{name}' is structured but missing schema_path."
            )

        return PromptDefinition(
            name=name,
            model=model_cfg,
            required_variables=model_cfg.expected_variables(),
        )
