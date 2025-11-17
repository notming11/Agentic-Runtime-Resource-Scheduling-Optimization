"""
LLM Client Abstraction Layer

This module provides a unified interface for interacting with different LLM backends:
- Local LLM servers (OpenAI-compatible APIs like vLLM, llama.cpp, Ollama)
- Google Gemini API
- Other cloud providers

This allows the UDF optimizer to work with any LLM backend without code changes.
"""

import os
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    backend: str = "local"  # "local", "gemini", "openai"

    # Local LLM settings
    local_api_base: str = "http://localhost:8000/v1"
    local_model: str = "local-model"
    local_api_key: Optional[str] = None  # Some local servers require API keys

    # Gemini settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash-exp"

    # OpenAI settings (for cloud OpenAI or compatible)
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_model: str = "gpt-4"

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9

    # Timeout settings
    timeout_seconds: float = 120.0


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to request JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Synchronous version of generate.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to request JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass


class LocalLLMClient(BaseLLMClient):
    """
    Client for local LLM servers using OpenAI-compatible API.

    Compatible with:
    - vLLM (with --api-key or without)
    - llama.cpp server
    - Ollama (via OpenAI compatibility)
    - text-generation-webui (in OpenAI mode)
    - LocalAI
    - Any OpenAI-compatible endpoint
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize local LLM client.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Import OpenAI client
        try:
            from openai import AsyncOpenAI, OpenAI
            self._has_openai = True
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            self._has_openai = False
            return

        # Create clients
        client_kwargs = {
            "base_url": config.local_api_base,
            "api_key": config.local_api_key or "dummy-key",  # Some servers require any key
            "timeout": config.timeout_seconds
        }

        self.async_client = AsyncOpenAI(**client_kwargs)
        self.sync_client = OpenAI(**client_kwargs)

        logger.info(f"Initialized LocalLLMClient for {config.local_api_base}")
        logger.info(f"Using model: {config.local_model}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from local LLM asynchronously.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to request JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._has_openai:
            raise RuntimeError("openai package not installed")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request
        request_kwargs = {
            "model": self.config.local_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Add JSON mode if requested (some servers support this)
        if json_mode:
            try:
                request_kwargs["response_format"] = {"type": "json_object"}
            except:
                # If not supported, add JSON instruction to prompt
                if system_prompt:
                    messages[0]["content"] += "\n\nYou must respond with valid JSON only."
                else:
                    messages.insert(0, {
                        "role": "system",
                        "content": "You must respond with valid JSON only."
                    })

        try:
            logger.debug(f"Sending request to local LLM: {self.config.local_api_base}")

            response = await self.async_client.chat.completions.create(**request_kwargs)

            result = response.choices[0].message.content
            logger.debug(f"Received response: {len(result)} characters")

            return result

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            raise

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from local LLM synchronously.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to request JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._has_openai:
            raise RuntimeError("openai package not installed")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request
        request_kwargs = {
            "model": self.config.local_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Add JSON mode if requested
        if json_mode:
            try:
                request_kwargs["response_format"] = {"type": "json_object"}
            except:
                if system_prompt:
                    messages[0]["content"] += "\n\nYou must respond with valid JSON only."
                else:
                    messages.insert(0, {
                        "role": "system",
                        "content": "You must respond with valid JSON only."
                    })

        try:
            response = self.sync_client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            raise


class GeminiLLMClient(BaseLLMClient):
    """Client for Google Gemini API."""

    def __init__(self, config: LLMConfig):
        """
        Initialize Gemini client.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Import Gemini
        try:
            import google.generativeai as genai
            self._genai = genai
            self._has_gemini = True
        except ImportError:
            logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")
            self._has_gemini = False
            return

        # Configure API
        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in config or environment")
            self._has_gemini = False
            return

        genai.configure(api_key=api_key)

        # Create base model
        self.model_name = config.gemini_model

        logger.info(f"Initialized GeminiLLMClient with model: {self.model_name}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from Gemini asynchronously.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to request JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._has_gemini:
            raise RuntimeError("google-generativeai not installed or API key not configured")

        # Create model with config
        gen_config = self._genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )

        # Add JSON mode if requested
        if json_mode:
            gen_config.response_mime_type = "application/json"

        # Create model
        model_kwargs = {
            "generation_config": gen_config
        }
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        model = self._genai.GenerativeModel(self.model_name, **model_kwargs)

        try:
            # Run in thread pool to not block async event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from Gemini synchronously.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to request JSON output
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._has_gemini:
            raise RuntimeError("google-generativeai not installed or API key not configured")

        # Create model with config
        gen_config = self._genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )

        # Add JSON mode if requested
        if json_mode:
            gen_config.response_mime_type = "application/json"

        # Create model
        model_kwargs = {
            "generation_config": gen_config
        }
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        model = self._genai.GenerativeModel(self.model_name, **model_kwargs)

        try:
            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


def create_llm_client(config: Optional[LLMConfig] = None) -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client based on configuration.

    Args:
        config: LLM configuration (if None, loads from environment/defaults)

    Returns:
        Appropriate LLM client instance

    Raises:
        ValueError: If backend is not supported
    """
    if config is None:
        # Load from environment
        backend = os.getenv("LLM_BACKEND", "local")
        config = LLMConfig(
            backend=backend,
            local_api_base=os.getenv("LOCAL_LLM_API_BASE", "http://localhost:8000/v1"),
            local_model=os.getenv("LOCAL_LLM_MODEL", "local-model"),
            local_api_key=os.getenv("LOCAL_LLM_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
        )

    backend = config.backend.lower()

    if backend == "local":
        return LocalLLMClient(config)
    elif backend == "gemini":
        return GeminiLLMClient(config)
    elif backend == "openai":
        # Use LocalLLMClient with OpenAI settings
        config.local_api_base = config.openai_api_base or "https://api.openai.com/v1"
        config.local_model = config.openai_model
        config.local_api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        return LocalLLMClient(config)
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")


def load_llm_config_from_yaml(yaml_path: str) -> LLMConfig:
    """
    Load LLM configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        LLMConfig object
    """
    import yaml

    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)

    llm_config = config_data.get("llm", {})

    return LLMConfig(
        backend=llm_config.get("backend", "local"),
        local_api_base=llm_config.get("local_api_base", "http://localhost:8000/v1"),
        local_model=llm_config.get("local_model", "local-model"),
        local_api_key=llm_config.get("local_api_key"),
        gemini_api_key=llm_config.get("gemini_api_key", os.getenv("GEMINI_API_KEY")),
        gemini_model=llm_config.get("gemini_model", "gemini-2.0-flash-exp"),
        openai_api_key=llm_config.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
        openai_api_base=llm_config.get("openai_api_base"),
        openai_model=llm_config.get("openai_model", "gpt-4"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
        top_p=llm_config.get("top_p", 0.9),
        timeout_seconds=llm_config.get("timeout_seconds", 120.0),
    )
