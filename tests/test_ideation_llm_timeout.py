"""Regression test: strategy_ideation._call_llm's HTTP timeout must be
configurable via HYPO_LLM_TIMEOUT (mirroring the existing pattern in
llm_hypothesis.py), not hardcoded to 60s.

Found 2026-07-29: after redirecting HYPO_LLM_BASE_URL to alibaba-token-plan
(the deepseek.com account had run out of balance, HTTP 402), the brainstorm
stage started hitting "The read operation timed out" on every attempt --
alibaba-token-plan's qwen3.8-max-preview needs longer than 60s for the
brainstorm prompt's context size, and there was no way to configure that
without editing code.
"""
import os
from unittest import mock

import strategy_ideation as si


def test_default_timeout_is_60_when_env_unset(monkeypatch):
    monkeypatch.delenv("HYPO_LLM_TIMEOUT", raising=False)
    cfg = si._get_llm_config()
    assert cfg["timeout"] == 60


def test_timeout_configurable_via_env(monkeypatch):
    monkeypatch.setenv("HYPO_LLM_TIMEOUT", "180")
    cfg = si._get_llm_config()
    assert cfg["timeout"] == 180


def test_call_llm_passes_configured_timeout_to_post_json(monkeypatch):
    monkeypatch.setenv("HYPO_LLM_TIMEOUT", "180")
    monkeypatch.setenv("HYPO_LLM_MODEL", "qwen3.8-max-preview")

    captured = {}

    def fake_post_json(url, headers, payload, timeout, retries=2):
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    with mock.patch.object(si, "_post_json", side_effect=fake_post_json):
        si._call_llm([{"role": "user", "content": "hi"}])

    assert captured["timeout"] == 180, (
        f"_call_llm must use the configured HYPO_LLM_TIMEOUT, got {captured.get('timeout')!r}"
    )
