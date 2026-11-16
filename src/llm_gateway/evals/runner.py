"""Optional model runner. The toolkit never calls a model unless you hand it a callable."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .schema import Case, Sample

ModelFn = Callable[[list[dict[str, str]]], str]


def run_cases(
    cases: Sequence[Case], fn: ModelFn, variant: str, *, samples: int = 1
) -> list[Sample]:
    """Play each case through ``fn(messages) -> reply`` and return the samples.

    ``messages`` is the OpenAI-style history so far, ending with the user turn to answer. For
    multi-turn cases the produced reply is appended to the history before the next user turn
    (a scripted assistant turn in the case replaces it). ``fn`` can wrap any client, including
    the gateway router; no network call happens otherwise.
    """
    out: list[Sample] = []
    for case in cases:
        for n in range(samples):
            history: list[dict[str, str]] = []
            replies: list[str] = []
            for i, turn in enumerate(case.turns):
                history.append({"role": turn.role, "content": turn.content})
                nxt = case.turns[i + 1] if i + 1 < len(case.turns) else None
                if turn.role == "user" and (nxt is None or nxt.role == "user"):
                    reply = fn(list(history))
                    replies.append(reply)
                    history.append({"role": "assistant", "content": reply})
            out.append(Sample(case.id, variant, tuple(replies), n))
    return out
