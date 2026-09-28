"""Exceptions that the CLI maps to exit codes."""


class StemcueError(Exception):
    """Base of every error stemcue reports to the user."""


class UsageError(StemcueError):
    """Invalid arguments (exit 2)."""


class WeightsIntegrityError(StemcueError):
    """A weight file failed a hash, structure or allowlist check (exit 3)."""


class InputError(StemcueError):
    """Unreadable or missing input, or an unwritable weights directory (exit 4)."""


class NetworkError(StemcueError):
    """Downloading weights failed (exit 5)."""
