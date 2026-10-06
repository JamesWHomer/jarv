"""Best-effort disposal of owned provider clients."""

from contextlib import suppress


def close_client(client) -> None:
    """Release a client without hiding the outcome of a completed/cancelled run."""
    close = getattr(client, "close", None)
    if callable(close):
        with suppress(Exception):
            close()
