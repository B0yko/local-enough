from __future__ import annotations

import sys

import pytest

from local_enough.providers.caffeinate import spawn_caffeinate


def test_noop_outside_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert spawn_caffeinate(1234) is None


def test_spawns_caffeinate_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    captured: dict[str, object] = {}

    class _FakeProc:
        pass

    def fake_popen(args: list[str], **kwargs: object) -> _FakeProc:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr("local_enough.providers.caffeinate.subprocess.Popen", fake_popen)
    result = spawn_caffeinate(4242)
    assert isinstance(result, _FakeProc)
    assert captured["args"] == ["caffeinate", "-i", "-w", "4242"]
