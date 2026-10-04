"""Revalidate queued process completions at the boundary of an agent turn."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent


def process_completion_is_consumed(event: MessageEvent) -> bool:
    """Drop only an unchanged, wholly read completion notice from this process incarnation.

    The watcher checks consumption before queueing, but wait/log can read the result
    while the launching turn is still running. Its queued wake is then obsolete.
    A mixed batch, merged human message, or unknown producer must still be delivered.
    The snapshot is process-local provenance, never deserialized ingress metadata.
    """
    snapshot = getattr(event, "_process_completion_snapshot", None)
    if not event.internal or snapshot is None or event.media_urls:
        return False
    original_text, identities = snapshot
    if event.text != original_text or not identities:
        return False

    from tools.process_registry import process_registry

    for process_id, started_at in identities:
        if started_at is None or not process_registry.is_completion_consumed(process_id):
            return False
        session = process_registry.get(process_id)
        if session is None or session.started_at != started_at:
            return False
    return True
