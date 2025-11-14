"""Tests for prompt config loading and validation."""

import textwrap
from pathlib import Path

import pytest

from src.py_promptkit.errors import PromptConfigError
from src.py_promptkit.loader import PromptLoader


@pytest.fixture
def toml_path(tmp_path: Path) -> Path:
    """Write a valid TOML file to a temporary path."""
    path = tmp_path / "prompts.toml"
    path.write_text(
        textwrap.dedent(
            """
            [models]
            test = "test-model"

            [providers]
            test = "test-provider"

            [temperatures]
            test = 0.5

            [test]
            template = "This is a test."
            """
        )
    )
    return path


def test_load_valid_toml(toml_path: Path):
    """Test that a valid TOML file loads correctly."""
    loader = PromptLoader(toml_path)
    defs = loader.load()
    assert "test" in defs
    assert defs["test"].model.name == "test-model"
    assert defs["test"].model.provider == "test-provider"
    assert defs["test"].model.temperature == 0.5
    assert defs["test"].model.template == "This is a test."


def test_load_invalid_toml(tmp_path: Path):
    """Test that an invalid TOML file raises an error."""
    path = tmp_path / "invalid.toml"
    path.write_text("this is not valid toml")
    loader = PromptLoader(path)
    with pytest.raises(PromptConfigError, match=r"Failed to read TOML"):
        loader.load()


def test_load_missing_models_section(tmp_path: Path):
    """Test that a TOML file missing the [models] section raises an error."""
    path = tmp_path / "missing_models.toml"
    path.write_text(
        textwrap.dedent(
            """
            [providers]
            test = "test-provider"

            [test]
            template = "This is a test."
            """
        )
    )
    loader = PromptLoader(path)
    with pytest.raises(PromptConfigError, match=r"Missing \[models\] section"):
        loader.load()


def test_load_missing_providers_section(tmp_path: Path):
    """Test that a TOML file missing the [providers] section raises an error."""
    path = tmp_path / "missing_providers.toml"
    path.write_text(
        textwrap.dedent(
            """
            [models]
            test = "test-model"

            [test]
            template = "This is a test."
            """
        )
    )
    loader = PromptLoader(path)
    with pytest.raises(PromptConfigError, match=r"Missing \[providers\] section"):
        loader.load()


def test_load_no_prompt_definitions(tmp_path: Path):
    """Test that a TOML file with no prompt definitions raises an error."""
    path = tmp_path / "no_prompts.toml"
    path.write_text(
        textwrap.dedent(
            """
            [models]
            test = "test-model"

            [providers]
            test = "test-provider"
            """
        )
    )
    loader = PromptLoader(path)
    with pytest.raises(PromptConfigError):
        loader.load()


def test_load_missing_template(tmp_path: Path):
    """Test that a prompt missing a template raises an error."""
    path = tmp_path / "missing_template.toml"
    path.write_text(
        textwrap.dedent(
            """
            [models]
            test = "test-model"

            [providers]
            test = "test-provider"

            [test]
            foo = "bar"
            """
        )
    )
    loader = PromptLoader(path)
    with pytest.raises(PromptConfigError, match="missing 'template' key"):
        loader.load()


def test_load_structured_missing_schema_path(tmp_path: Path):
    """Test that a structured prompt missing a schema_path raises an error."""
    path = tmp_path / "missing_schema_path.toml"
    path.write_text(
        textwrap.dedent(
            """
            [models]
            test = "test-model"

            [providers]
            test = "test-provider"

            [test]
            template = "This is a test."
            structured = true
            """
        )
    )
    loader = PromptLoader(path)
    with pytest.raises(PromptConfigError, match="missing 'schema_path'"):
        loader.load()


def test_get_prompt_not_found(toml_path: Path):
    """Test that get() raises an error if the prompt is not found."""
    loader = PromptLoader(toml_path)
    loader.load()
    with pytest.raises(PromptConfigError):
        loader.get("not-found")
