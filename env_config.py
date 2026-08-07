"""
Minimal .env loader for the qwen_testing scripts.

No third-party dependency: python-dotenv is used when available, otherwise a
small built-in parser handles the same KEY=VALUE format. Scripts in the
subdirectories reach this via:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from env_config import load_env, require_env

The .env file lives at qwen_testing/.env and is gitignored. Copy
.env.example to .env and fill in your own values.
"""

import os
from pathlib import Path

# qwen_testing/ — the directory this module sits in
ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT / ".env"

_loaded = False


def _parse_env_file(path: Path) -> dict:
    """Parse a KEY=VALUE .env file. Ignores blanks, comments and `export `."""
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # strip a single layer of matching quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path=None, override=False):
    """
    Load qwen_testing/.env into os.environ.

    Real environment variables win by default, so `EAS_API_KEY=... python x.py`
    overrides the file. Safe to call repeatedly; only the first call does work
    unless a custom path or override=True is given.

    Args:
        path: alternative .env location (default: qwen_testing/.env)
        override: if True, .env values replace existing environment variables

    Returns:
        True if a file was read, False if none was found.
    """
    global _loaded
    if _loaded and path is None and not override:
        return True

    env_path = Path(path) if path else DEFAULT_ENV_PATH

    try:
        from dotenv import load_dotenv  # optional dependency
        found = load_dotenv(env_path, override=override)
    except ImportError:
        if not env_path.exists():
            found = False
        else:
            for key, value in _parse_env_file(env_path).items():
                if override or key not in os.environ:
                    os.environ[key] = value
            found = True

    if path is None:
        _loaded = True
    return found


def get_env(name, default=None):
    """Read a variable, loading .env first if it has not been read yet."""
    load_env()
    return os.environ.get(name, default)


def require_env(name, hint=""):
    """
    Read a required variable, or exit with an actionable message.

    Args:
        name: environment variable name
        hint: extra guidance appended to the error
    """
    load_env()
    value = os.environ.get(name)
    if not value:
        message = (
            f"\nMissing required environment variable: {name}\n"
            f"  1. cp {ROOT / '.env.example'} {DEFAULT_ENV_PATH}\n"
            f"  2. set {name} in that file\n"
            f"  (or export {name}=... in your shell)"
        )
        if hint:
            message += f"\n  {hint}"
        raise SystemExit(message)
    return value
