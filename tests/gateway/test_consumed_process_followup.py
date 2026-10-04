"""A queued process result must be rechecked after the launching turn reads it."""

import asyncio
import time

import pytest

from gateway.platforms.event import MessageEvent
from tests.gateway.test_background_process_notifications import _build_runner
from tests.gateway.test_queued_followup_processing_hooks import (
    HookRecordingAdapter, SESSION_KEY, _install_fake_agent, _make_runner, _source,
)
from tools.process_registry import ProcessRegistry, ProcessSession


async def _run_queued_case(monkeypatch, tmp_path, *, observation="wait", variant="single"):
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    process = ProcessSession(
        id="proc_failed_start", command="acm company Hexatronic", started_at=time.time(),
        exited=True, exit_code=127, output_buffer="launcher missing\n",
    )
    registry._finished[process.id] = process
    adapter = HookRecordingAdapter()
    injector = _build_runner(monkeypatch, tmp_path, "concise")
    injector._build_process_event_source = lambda event: _source()
    injector._resolve_injection_adapter = lambda platform, source: adapter
    injector._completion_notification_batch_window = 0

    async def queue(event):
        adapter._pending_messages[SESSION_KEY] = event
        event._gateway_accepted = True

    monkeypatch.setattr(adapter, "handle_message", queue)
    completion = {
        "type": "completion", "session_id": process.id, "started_at": process.started_at,
        "session_key": SESSION_KEY, "exit_code": 127, "command": process.command,
        "output": process.output_buffer,
    }
    if variant in {"all_read", "part_read"}:
        sibling = ProcessSession(
            id="proc_sibling", command="other job", started_at=time.time(),
            exited=True, exit_code=0, output_buffer="done\n",
        )
        registry._finished[sibling.id] = sibling
        other = {**completion, "session_id": sibling.id, "started_at": sibling.started_at,
                 "exit_code": 0, "command": sibling.command}
        assert all(await asyncio.gather(
            injector._enqueue_process_completion_notification("failed start", completion),
            injector._enqueue_process_completion_notification("other job done", other),
        ))
    else:
        if variant == "watch":
            completion["type"] = "watch_match"
        assert await injector._inject_watch_notification("failed start", completion)

    queued = adapter._pending_messages[SESSION_KEY]
    if variant == "human":
        adapter._pending_messages[SESSION_KEY] = MessageEvent(
            text=queued.text, source=_source(), metadata=queued.metadata.copy(),
        )
    elif variant == "merged":
        queued.text += "\n\nPlease analyse another company."

    calls = []

    class Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, **kwargs):
            calls.append(message)
            if len(calls) == 1:
                if observation == "wait":
                    assert registry.wait(process.id, timeout=1)["exit_code"] == 127
                elif observation == "log":
                    assert registry.read_log(process.id)["output"] == "launcher missing"
                elif observation == "poll":
                    registry.poll(process.id)
                if variant == "all_read":
                    registry.wait(sibling.id, timeout=1)
                elif variant == "reused":
                    process.started_at += 1
                elif variant == "missing":
                    registry._finished.pop(process.id)
            return {"final_response": "Hexatronic complete", "messages": [], "api_calls": 1}

    _install_fake_agent(monkeypatch, tmp_path, Agent)
    runner = _make_runner(adapter)
    result = await runner._run_agent(
        message="Analyse Hexatronic", context_prompt="", history=[], source=_source(),
        session_id="hexatronic-request", session_key=SESSION_KEY,
    )
    return calls, result


@pytest.mark.asyncio
@pytest.mark.parametrize("observation", ["wait", "log", "poll", "none"])
async def test_queued_completion_does_not_repeat_an_already_read_result(monkeypatch, tmp_path, observation):
    calls, result = await _run_queued_case(monkeypatch, tmp_path, observation=observation)
    assert len(calls) == (1 if observation in {"wait", "log"} else 2)
    assert result["final_response"] == "Hexatronic complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["all_read", "part_read", "reused", "missing", "human", "merged", "watch"])
async def test_only_wholly_consumed_process_notices_are_suppressed(monkeypatch, tmp_path, variant):
    calls, result = await _run_queued_case(monkeypatch, tmp_path, variant=variant)
    assert len(calls) == (1 if variant == "all_read" else 2)
    assert result["final_response"] == "Hexatronic complete"
