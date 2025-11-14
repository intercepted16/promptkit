"""Test a mock OpenAI LLM client with hooks and caching."""

from __future__ import annotations

from pathlib import Path
from tests.utils import TEST_RESOURCES_DIR

from dotenv import dotenv_values, find_dotenv

from src.py_promptkit import PromptLoader, PromptRunner
from src.py_promptkit.litellm import LiteLLMClient
from src.py_promptkit.utils._logger import logger
import subprocess

dotenv_filename = ".env.test"
dotenv_path = find_dotenv(dotenv_filename)
if not dotenv_path:
    raise RuntimeError(f"Could not find {dotenv_filename} file")

secrets = dotenv_values(dotenv_path)


def _start_mockai():
    """Start the mock AI server using pnpm."""
    logger.info("Starting MockAI server...")
    command = "pnpm start"
    with open("/dev/null", "wb") as logfile:
        subprocess.Popen(
            args=command,
            shell=True,
            cwd=Path(__file__).parent.parent / "mockai",
            stdout=logfile,
            stderr=logfile,
        )


def test_openai():
    """Test OpenAI client integration."""
    toml_path = TEST_RESOURCES_DIR / "openai.toml"
    # set secrets, load prompts, create runner
    # ^^ basically a happy path of the app; expected usage, and with common client like openai and basic settings.
    # set_secrets(secrets)
    loader = PromptLoader(toml_path)
    loader.load()
    runner = PromptRunner(
        loader,
    )
    _start_mockai()
    runner.register_client("litellm", LiteLLMClient(secrets=secrets))
    response = runner.run(
        "demo",
        {"name": "Ada", "product": "PromptKit"},
    )
    runner.close()
    logger.info(f"OpenAI response: {response}")
