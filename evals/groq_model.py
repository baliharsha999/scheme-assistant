import os
import time
import json
import asyncio
from typing import Optional, Union, Any, Type
from pydantic import BaseModel

from groq import Groq, AsyncGroq, RateLimitError, APIError
from deepeval.models.base_model import DeepEvalBaseLLM
from evals.config import (
    GROQ_API_KEY,
    EVAL_JUDGE_MODEL_NAME,
    EVAL_TEMPERATURE,
    EVAL_REQUEST_TIMEOUT,
    EVAL_MAX_RETRIES,
    EVAL_RATE_LIMIT_DELAY,
)


def _clean_markdown_json(text: str) -> str:
    """Strip markdown code block fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop leading and trailing fence lines
        cleaned_lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        return "\n".join(cleaned_lines).strip()
    return text


class GroqJudgeModel(DeepEvalBaseLLM):
    """
    Custom DeepEvalBaseLLM implementation that routes evaluation judging
    through Groq models without invoking OpenAI.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = EVAL_TEMPERATURE,
        timeout: int = EVAL_REQUEST_TIMEOUT,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name or EVAL_JUDGE_MODEL_NAME
        self.api_key = api_key or GROQ_API_KEY
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your environment or .env file."
            )

        self._sync_client: Optional[Groq] = None
        self._async_client: Optional[AsyncGroq] = None

        super().__init__(model=self.model_name)

    def load_model(self) -> Groq:
        """Initialize the underlying Groq sync client."""
        self._sync_client = Groq(
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self._sync_client

    def _get_async_client(self) -> AsyncGroq:
        """Lazily initialize the underlying AsyncGroq client."""
        if self._async_client is None:
            self._async_client = AsyncGroq(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._async_client

    def get_model_name(self) -> str:
        return self.model_name

    def supports_json_mode(self) -> bool:
        return True

    def supports_structured_outputs(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> Union[str, BaseModel]:
        """
        Synchronous completion using Groq with exponential backoff retries.
        """
        client = self.model or self.load_model()
        messages = [{"role": "user", "content": prompt}]

        response_format = None
        if schema is not None:
            response_format = {"type": "json_object"}

        for attempt in range(1, EVAL_MAX_RETRIES + 1):
            try:
                kwargs_call = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if response_format:
                    kwargs_call["response_format"] = response_format

                response = client.chat.completions.create(**kwargs_call)
                choice = response.choices[0]
                content = choice.message.content or ""
                content = _clean_markdown_json(content)

                if schema is not None:
                    try:
                        return schema.model_validate_json(content)
                    except Exception:
                        return content

                return content

            except RateLimitError as rle:
                wait_time = (2 ** attempt) * EVAL_RATE_LIMIT_DELAY
                print(f"[GroqJudgeModel] Rate limit hit on attempt {attempt}/{EVAL_MAX_RETRIES}. Sleeping {wait_time:.1f}s...")
                time.sleep(wait_time)
                if attempt == EVAL_MAX_RETRIES:
                    raise rle

            except Exception as exc:
                wait_time = (2 ** attempt) * EVAL_RATE_LIMIT_DELAY
                if attempt < EVAL_MAX_RETRIES:
                    print(f"[GroqJudgeModel] Error ({exc}) on attempt {attempt}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise exc

        return ""

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> Union[str, BaseModel]:
        """
        Asynchronous completion using AsyncGroq with exponential backoff retries.
        """
        client = self._get_async_client()
        messages = [{"role": "user", "content": prompt}]

        response_format = None
        if schema is not None:
            response_format = {"type": "json_object"}

        for attempt in range(1, EVAL_MAX_RETRIES + 1):
            try:
                kwargs_call = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if response_format:
                    kwargs_call["response_format"] = response_format

                response = await client.chat.completions.create(**kwargs_call)
                choice = response.choices[0]
                content = choice.message.content or ""
                content = _clean_markdown_json(content)

                if schema is not None:
                    try:
                        return schema.model_validate_json(content)
                    except Exception:
                        return content

                return content

            except RateLimitError as rle:
                wait_time = (2 ** attempt) * EVAL_RATE_LIMIT_DELAY
                print(f"[GroqJudgeModel Async] Rate limit on attempt {attempt}/{EVAL_MAX_RETRIES}. Sleeping {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                if attempt == EVAL_MAX_RETRIES:
                    raise rle

            except Exception as exc:
                wait_time = (2 ** attempt) * EVAL_RATE_LIMIT_DELAY
                if attempt < EVAL_MAX_RETRIES:
                    print(f"[GroqJudgeModel Async] Error ({exc}) on attempt {attempt}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise exc

        return ""

    def generate_with_schema(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> Union[str, BaseModel]:
        return self.generate(prompt, schema=schema, **kwargs)

    async def a_generate_with_schema(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> Union[str, BaseModel]:
        return await self.a_generate(prompt, schema=schema, **kwargs)
