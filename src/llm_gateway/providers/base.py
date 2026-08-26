"""Shared provider-call contract.

Every provider module exposes a single `async def call(config, system, prompt,
max_tokens) -> ProviderResult`, so `router.py` can treat all providers
uniformly. Providers own their own client caching internally.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
