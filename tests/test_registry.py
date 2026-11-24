from __future__ import annotations

import sys

import pytest

from pyflue.harnesses.registry import create_backend
from pyflue.types import PyFlueConfig


def test_default_registry_resolves_deepagents():
    assert create_backend("deepagents").name == "deepagents"


def test_planned_backends_are_registered():
    assert create_backend("openai-agents").name == "openai_agents"
    assert create_backend("google-adk").name == "google_adk"
    assert create_backend("pydanticai").name == "pydanticai"


def test_unknown_backend_error_lists_available():
    with pytest.raises(KeyError, match="Available"):
        create_backend("missing")


@pytest.mark.asyncio
async def test_pydanticai_backend_raises_install_hint_when_optional_dep_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    backend = create_backend("pydanticai")
    with pytest.raises(ImportError, match=r"pyflue\[pydanticai\]"):
        await backend.run(
            prompt="hello",
            system_prompt="",
            config=PyFlueConfig(root=tmp_path),
            skills={},
            sandbox=None,
            session_id="s",
        )
