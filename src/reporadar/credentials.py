"""Where the API key lives, and the one function that finds it.

The key has to be readable by a process the *editor* spawns, and four otherwise-obvious
routes are closed to it. Shell environment is not reliably inherited by an MCP server
(Copilot inherits PATH and nothing else). The plugin's ``.mcp.json`` is committed to a
public repository. VS Code's ``inputs`` prompt does not exist for plugin-bundled configs --
that format is a closed schema. And a tool argument would put the key in the model's
context and the transcript. So it is written once, deliberately, by ``rr auth``, and read
back from a file: the ``gh auth login`` / ``docker login`` shape.

A file rather than the OS keyring, and not for convenience. The server launches as
``uvx --from "reporadar-papers[mcp]==1.0.1" rr mcp``, so the executable path changes on
every version bump -- and macOS Keychain ACLs are keyed to the accessing binary's identity,
which would make a routine plugin update look like a new application and silently break the
server. ``gh`` documents that exact failure for Homebrew's unsigned builds. The server is
also spawned by an editor rather than a shell, which is where keyring backends are least
likely to exist at all (Remote-SSH, devcontainers, Codespaces, WSL). A mode-0600 file in
the user's own home has neither failure mode. Of ten comparable CLIs surveyed, only two
encrypt by default and both keep a plaintext path; Hugging Face, whose situation is closest
to this one, has no keyring at all.

**What 0600 protects against, precisely:** other users on the machine. It does not protect
against code running as you, and neither does a keyring in practice -- GitHub's own notes
concede a stored token can be retrieved by calling ``security`` directly. That is the whole
claim, and there is no more to it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# provider -> (config attribute, environment variable, key in the credentials file)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openai": ("openai_api_key", "OPENAI_API_KEY", "openai"),
    "claude": ("claude_api_key", "ANTHROPIC_API_KEY", "anthropic"),
}

_ENV_CONFIG_DIR = "REPORADAR_CONFIG_DIR"


def config_dir() -> Path:
    """The per-user configuration directory, overridable like GH_CONFIG_DIR.

    ``~/.config/reporadar`` on macOS as well as Linux, deliberately: the GitHub CLI does the
    same, and "~/Library/Application Support" is materially worse to say out loud in a
    support reply.
    """
    override = os.environ.get(_ENV_CONFIG_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", "")
        if base:
            return Path(base) / "reporadar"
        return Path.home() / "AppData" / "Roaming" / "reporadar"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return root / "reporadar"


def auth_path() -> Path:
    """The credentials file.

    Separate from every other config file on purpose: "the one with the secret in it" should
    be a single path you can name, back up deliberately, or delete in anger. The GitHub CLI
    splits hosts.yml from config.yml for the same reason.
    """
    return config_dir() / "auth.json"


def _read() -> dict[str, Any]:
    path = auth_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt credentials file must not take down a run that has a perfectly good
        # environment variable sitting right there. Treated as "nothing stored".
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict[str, Any]) -> Path:
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = json.dumps(data, indent=2) + "\n"
    # The mode goes to os.open rather than a chmod afterwards: creating the file and then
    # narrowing it leaves a window where it is world-readable, and that window is exactly
    # when a secret is being written into it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def store(provider: str, api_key: str) -> Path:
    """Save *api_key* for *provider*, leaving any other provider's key untouched."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDERS))}")
    _, _, slot = PROVIDERS[provider]
    data = _read()
    entry = data.get(slot)
    data[slot] = {**entry, "api_key": api_key} if isinstance(entry, dict) else {"api_key": api_key}
    return _write(data)


def remove(provider: str) -> bool:
    """Forget *provider*'s key. True if there was one to forget."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")
    _, _, slot = PROVIDERS[provider]
    data = _read()
    if slot not in data:
        return False
    del data[slot]
    _write(data)
    return True


def stored(provider: str) -> str:
    """The stored key for *provider*, or "". The file only -- not env, not config.

    Never raises. This is consulted on the way to every LLM call, so it has to degrade to
    "nothing stored" for anything that can go wrong while looking: an unreadable path, and
    in particular a missing home directory. ``Path.home()`` raises outright when neither
    HOME nor USERPROFILE is set -- true inside containers, under ``env -i``, and on some CI
    runners -- and a key lookup that dies there would take the run with it while an
    environment variable may have been sitting right there all along.
    """
    if provider not in PROVIDERS:
        return ""
    _, _, slot = PROVIDERS[provider]
    try:
        entry = _read().get(slot)
    except (OSError, RuntimeError):
        return ""
    if not isinstance(entry, dict):
        return ""
    value = entry.get("api_key", "")
    return value if isinstance(value, str) else ""


def fingerprint(api_key: str) -> str:
    """A key you can recognise but cannot use. ``rr auth --status`` prints this, never the
    key itself -- the point of storing a secret is defeated by a command that echoes it."""
    if not api_key:
        return "(none)"
    if len(api_key) < 12:
        return "****"
    return f"{api_key[:3]}****{api_key[-4:]}"


def resolve_api_key(provider: str, cfg: Any = None) -> str:
    """Find *provider*'s key: configuration, then environment, then the credentials file.

    THE point of this function is that there is exactly one of it. This resolution used to
    be spelled out at six call sites -- three in ``llm_client``, three in ``rr doctor`` --
    which is the shape that let ``rr doctor`` certify a gate as healthy while the pipeline
    skipped it entirely. Two implementations of one rule disagree eventually, and the
    failure is silent, because in isolation both of them look right.

    The order changes nothing that already works. A configured value is the most deliberate
    thing a user can write, and it may itself be ``${OPENAI_API_KEY}`` -- which is how the
    GitHub Action injects a secret without putting it in the repository. The environment
    comes next, so an exported key keeps working and costs nothing. The file is last
    because it exists for the case neither of the others reaches: a server an editor
    started.
    """
    if provider not in PROVIDERS:
        return ""
    attr, env_var, _ = PROVIDERS[provider]
    if cfg is not None:
        configured = getattr(cfg, attr, "") or ""
        if configured:
            return str(configured)
    from_env = os.environ.get(env_var, "").strip()
    if from_env:
        return from_env
    return stored(provider)


def source_of(provider: str, cfg: Any = None) -> str:
    """Where :func:`resolve_api_key` would take the key from: config, environment, file, none.

    ``rr doctor`` says this out loud, because "a key is present" is not the useful fact when
    something is misbehaving. "The key is coming from the file, not the variable you just
    exported" is.
    """
    if provider not in PROVIDERS:
        return "none"
    attr, env_var, _ = PROVIDERS[provider]
    if cfg is not None:
        configured = getattr(cfg, attr, "") or ""
        if configured:
            return "config"
    if os.environ.get(env_var, "").strip():
        return "environment"
    if stored(provider):
        return "file"
    return "none"
