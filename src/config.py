"""
Configuration loader for the AI Agent.
Loads settings from config.yaml and environment variables.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM model configuration."""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 1.0


@dataclass
class MemoryConfig:
    """Memory and vector store configuration."""
    persist_dir: str = "./memory_store"
    embedding_model: str = "text-embedding-3-small"
    max_history: int = 10
    similarity_top_k: int = 3


@dataclass
class AgentConfig:
    """Main agent configuration."""
    max_iterations: int = 10
    early_stopping: str = "generate"
    verbose: bool = True
    handle_parsing_errors: bool = True


@dataclass
class AppConfig:
    """Application-level configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    api_key: Optional[str] = None


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file with environment variable overrides."""
    config = AppConfig()

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "agent" in data:
            config.agent = AgentConfig(**data["agent"])

    # Environment variable overrides
    config.api_key = os.getenv("OPENAI_API_KEY", config.api_key)

    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")
    if os.getenv("LLM_TEMPERATURE"):
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))

    return config
