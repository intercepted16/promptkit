"""App-wide logger setup with colors and a custom NOTICE level.

Provides a custom NOTICE level between INFO and WARNING.
All third-party libs stay quiet; your logs shine.
"""

import logging
import sys
from typing import Any, Optional

# -------------------------------------------------------------------
# 1. Define custom level (between INFO=20 and WARNING=30)
# -------------------------------------------------------------------
NOTICE_LEVEL = 25
logging.addLevelName(NOTICE_LEVEL, "NOTICE")


# -------------------------------------------------------------------
# 2. ANSI color setup
# -------------------------------------------------------------------
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "NOTICE": "\033[38;5;33m",  # Blue-ish
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[1;41m",  # White on red background
}


class ColorFormatter(logging.Formatter):
    """Formatter that adds color to level names."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with ANSI color codes."""
        color = COLORS.get(record.levelname.replace("\033", "").split(":")[0], "")
        levelname = record.levelname
        colored_level = f"{color}{levelname:<7}{RESET}"
        record.levelname = colored_level
        return super().format(record)


# -------------------------------------------------------------------
# 3. Base formatting
# -------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))

root_logger = logging.getLogger()
root_logger.setLevel(NOTICE_LEVEL)
root_logger.handlers.clear()
root_logger.addHandler(handler)


# -------------------------------------------------------------------
# 4. Silence noisy libraries
# -------------------------------------------------------------------
for noisy in ["httpx", "urllib3", "opensearch"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# -------------------------------------------------------------------
# 5. Custom logger class and helpers
# -------------------------------------------------------------------
class CustomLogger(logging.Logger):
    """Proxy that routes .info() to NOTICE internally."""

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log message at NOTICE level instead of INFO."""
        super().log(NOTICE_LEVEL, msg, *args, **kwargs)


logging.setLoggerClass(CustomLogger)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-specific colored logger."""
    return logging.getLogger(name or "app")


# -------------------------------------------------------------------
# 6. Global shared logger
# -------------------------------------------------------------------
logger = get_logger("app")
