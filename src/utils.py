"""
Utility functions for the AI Agent.
Logging, formatting, and helper methods.
"""

import logging
import time
import functools
from typing import Any, Callable


def setup_logger(name: str = "ai_agent", level: str = "INFO") -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger = setup_logger()
        logger.info(f"{func.__name__} executed in {elapsed:.3f}s")
        return result
    return wrapper


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_tool_result(tool_name: str, result: str) -> str:
    """Format tool execution result for display."""
    separator = "─" * 40
    return f"\n{separator}\n🔧 Tool: {tool_name}\n📤 Result: {truncate_text(result)}\n{separator}"


def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
    return len(text) // 4
