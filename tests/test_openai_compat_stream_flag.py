import inspect

import pytest

from llm_gateway.providers import STREAM_CALLS


@pytest.mark.parametrize("name", sorted(STREAM_CALLS))
def test_every_stream_provider_accepts_cache_system(name):
    # Providers other than Anthropic/Vertex ignore the flag but must accept it.
    assert "cache_system" in inspect.signature(STREAM_CALLS[name]).parameters
