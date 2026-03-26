"""
In-memory context storage per user.
Each user has: messages list, selected provider, model, temperature, max_tokens.
"""
from typing import Any
from config import MAX_CONTEXT_MESSAGES, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

# { user_id: { "messages": [...], "provider": str, "model": dict, "temperature": float, "max_tokens": int } }
_store: dict[int, dict[str, Any]] = {}


def get_context(user_id: int) -> dict[str, Any]:
    if user_id not in _store:
        _store[user_id] = {
            "messages": [],
            "provider": None,
            "model": None,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
    return _store[user_id]


def add_message(user_id: int, role: str, content: str) -> None:
    ctx = get_context(user_id)
    ctx["messages"].append({"role": role, "content": content})
    # trim to last N messages
    if len(ctx["messages"]) > MAX_CONTEXT_MESSAGES:
        ctx["messages"] = ctx["messages"][-MAX_CONTEXT_MESSAGES:]


def clear_context(user_id: int) -> None:
    ctx = get_context(user_id)
    ctx["messages"] = []


def set_session(user_id: int, provider: str, model: dict,
                temperature: float, max_tokens: int) -> None:
    ctx = get_context(user_id)
    ctx["provider"] = provider
    ctx["model"] = model
    ctx["temperature"] = temperature
    ctx["max_tokens"] = max_tokens
    ctx["messages"] = []  # reset context on new session setup


def is_configured(user_id: int) -> bool:
    ctx = get_context(user_id)
    return ctx["provider"] is not None and ctx["model"] is not None
