"""
LLM Interface — swappable between Claude API and local model.

Dev mode: Uses Anthropic Claude API (no GPU required)
Lambda mode: Uses local model via OpenAI-compatible API (vLLM/TGI on A6000)

Handles:
1. Edge weight inference — given two stats, estimate contagion weight + mechanism
2. Scenario probability — assess likelihood of economic scenarios
3. Stat inference — estimate values for stats without direct API sources
4. Report generation — briefings for different personas (finance minister, etc.)
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    raw_text: str
    parsed_json: Optional[dict] = None
    model: str = ""
    tokens_used: int = 0


class BaseLLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict:
        """Complete and parse response as JSON."""
        ...


class ClaudeAPIProvider(BaseLLMProvider):
    """
    Claude API provider for development and prototyping.

    Uses the Anthropic Python SDK. Requires ANTHROPIC_API_KEY env var.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("Install anthropic SDK: pip install anthropic")
        return self._client

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        return LLMResponse(
            raw_text=text,
            model=self._model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict:
        resp = await self.complete(
            system_prompt=system_prompt + "\n\nRespond ONLY with valid JSON. No markdown, no explanation.",
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            # Strip potential markdown fencing
            text = resp.raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON: %s\nRaw: %s", exc, resp.raw_text[:500])
            return {"error": str(exc), "raw": resp.raw_text[:500]}


class LocalModelProvider(BaseLLMProvider):
    """
    Local model provider for Lambda/GPU deployment.

    Connects to a vLLM or text-generation-inference server running locally.
    Compatible with OpenAI-style API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        api_key: str = "not-needed",
    ):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                raw_text=text,
                model=self._model,
                tokens_used=usage.get("total_tokens", 0),
            )

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict:
        resp = await self.complete(
            system_prompt=system_prompt + "\n\nRespond ONLY with valid JSON. No markdown, no explanation.",
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            text = resp.raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse local model JSON: %s", exc)
            return {"error": str(exc), "raw": resp.raw_text[:500]}


# ─── Prompt Templates ────────────────────────────────────────────────────────

EDGE_WEIGHT_SYSTEM = """You are a quantitative macroeconomist. Given two economic statistics
from different countries, estimate the strength and mechanism of their causal relationship.

Respond with JSON:
{
    "weight": <float 0.0-1.0>,
    "mechanism": "<one-sentence explanation of transmission channel>",
    "lag_months": <int, typical lag in months>,
    "confidence": <float 0.0-1.0>
}"""

SCENARIO_PROBABILITY_SYSTEM = """You are a geopolitical risk analyst. Given a scenario description,
current economic conditions, and affected countries, estimate the probability and impact.

Respond with JSON:
{
    "probability_12m": <float 0.0-1.0>,
    "severity": "<low|medium|high|critical>",
    "primary_channel": "<trade|financial|commodity|political>",
    "affected_stats": ["<stat_name_1>", "<stat_name_2>"],
    "impact_magnitude": <float, estimated % change in affected stats>,
    "confidence": <float 0.0-1.0>
}"""

STAT_INFERENCE_SYSTEM = """You are a sovereign economic data analyst. Given a country and a statistic
that cannot be directly sourced from public APIs, provide your best estimate based on
publicly available information, reports, and academic research.

Respond with JSON:
{
    "value": <float>,
    "unit": "<string>",
    "confidence": <float 0.0-1.0>,
    "sources": ["<source_1>", "<source_2>"],
    "reasoning": "<brief explanation>"
}"""


class LLMProcessor:
    """
    High-level LLM processing for the knowledge graph.

    Uses the configured provider (Claude API or local model) to:
    - Infer edge weights between stat nodes
    - Estimate scenario probabilities
    - Fill in LLM-dependent statistics
    """

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self._total_tokens = 0

    async def infer_edge_weight(
        self,
        source_country: str,
        source_stat: str,
        source_value: Optional[float],
        target_country: str,
        target_stat: str,
        target_value: Optional[float],
    ) -> dict:
        """Estimate the causal weight between two cross-border statistics."""
        prompt = (
            f"Source: {source_country} — {source_stat}"
            f"{f' (current value: {source_value})' if source_value is not None else ''}\n"
            f"Target: {target_country} — {target_stat}"
            f"{f' (current value: {target_value})' if target_value is not None else ''}\n\n"
            f"Estimate the contagion weight and transmission mechanism."
        )
        result = await self.provider.complete_json(EDGE_WEIGHT_SYSTEM, prompt)
        return result

    async def assess_scenario(
        self,
        scenario_title: str,
        scenario_description: str,
        current_conditions: dict,
    ) -> dict:
        """Estimate probability and impact of an economic scenario."""
        conditions_text = json.dumps(current_conditions, indent=2, default=str)
        prompt = (
            f"Scenario: {scenario_title}\n"
            f"Description: {scenario_description}\n\n"
            f"Current global conditions:\n{conditions_text}\n\n"
            f"Assess the probability and impact of this scenario over the next 12 months."
        )
        result = await self.provider.complete_json(SCENARIO_PROBABILITY_SYSTEM, prompt)
        return result

    async def infer_stat_value(
        self,
        country: str,
        stat_name: str,
        stat_description: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Estimate a statistic that cannot be directly sourced from APIs."""
        context_text = ""
        if context:
            context_text = f"\nAdditional context:\n{json.dumps(context, indent=2, default=str)}"

        prompt = (
            f"Country: {country}\n"
            f"Statistic: {stat_name}\n"
            f"Description: {stat_description}\n"
            f"{context_text}\n\n"
            f"Provide your best estimate for this statistic as of 2025-2026."
        )
        result = await self.provider.complete_json(STAT_INFERENCE_SYSTEM, prompt)
        return result


def get_llm_provider(mode: str = "claude") -> BaseLLMProvider:
    """
    Factory function to get the appropriate LLM provider.

    Args:
        mode: "claude" for API-based dev, "local" for GPU-based Lambda deployment
    """
    if mode == "claude":
        return ClaudeAPIProvider()
    elif mode == "local":
        return LocalModelProvider()
    else:
        raise ValueError(f"Unknown LLM mode: {mode}. Use 'claude' or 'local'.")
