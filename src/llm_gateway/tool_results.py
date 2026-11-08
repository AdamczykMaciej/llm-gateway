"""Size guard for tool results sent back to the model.

A tool that returns a huge payload (a whole file, an unpaginated API reply)
is resent on every later turn of an agent loop, so it can overflow the
context window or multiply cost. With a cap set, `cap_tool_results` cuts any
`role: "tool"` message's text down to the cap and appends a marker saying how
much was dropped, so the model knows the result is partial.

Off by default (`tool_result_max_chars=0`): truncation changes what the model
sees, so it is an explicit choice. Only text is cut, and only on tool
messages; the caller's own message list is never modified. Messages name
lengths only, never content.
"""

from .errors import InvalidRequestError


def resolve_cap(per_call: int | None, configured: int) -> int:
    """The effective cap: the per-call value when given, else the configured
    one. 0 means no cap."""
    if per_call is None:
        return max(configured, 0)
    if isinstance(per_call, bool) or not isinstance(per_call, int) or per_call < 0:
        raise InvalidRequestError("max_tool_result_chars must be a non-negative integer.")
    return per_call


def _marker(dropped: int, total: int) -> str:
    return (
        f"\n...[truncated by llm-gateway: {dropped} of {total} characters omitted "
        "(tool_result_max_chars)]"
    )


def _cut(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + _marker(len(text) - cap, len(text))


def cap_tool_results(messages: list[dict], cap: int) -> list[dict]:
    """`messages` with oversized tool-result text truncated to `cap`
    characters (plus a marker). Returns the same list when nothing changes."""
    if cap <= 0:
        return messages
    out: list[dict] = []
    changed = False
    for message in messages:
        content = message.get("content") if message.get("role") == "tool" else None
        new_content = content
        if isinstance(content, str):
            new_content = _cut(content, cap)
        elif isinstance(content, list):
            new_content = [
                {**part, "text": _cut(part["text"], cap)}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
        if new_content != content:
            changed = True
            out.append({**message, "content": new_content})
        else:
            out.append(message)
    return out if changed else messages
