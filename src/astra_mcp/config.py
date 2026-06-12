"""Config management — reads/writes ~/.astra_mcp/config.yaml with hot reload.

Schema:
  servers:
    <server_name>:
      base_url: str
      api_key: str
      username: str
      agents:
        <alias>:
          config_id: str
          config_name: str

  groups:
    <group_name>:
      agents:
        - <server_name>/<alias>   # reference format
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

CONFIG_DIR = Path.home() / ".astra_mcp"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Hot reload state
_reload_callbacks: list[Callable[[], None]] = []
_reload_thread: threading.Thread | None = None
_reload_stop = threading.Event()
_last_mtime: float = 0.0


def _default() -> dict[str, Any]:
    return {"servers": {}, "groups": {}}


def load() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return _default()
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return _default()

    merged = {**_default(), **data}
    if not isinstance(merged.get("servers"), dict):
        merged["servers"] = {}
    if not isinstance(merged.get("groups"), dict):
        merged["groups"] = {}
    return merged


def save(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    global _last_mtime
    _last_mtime = _file_mtime()


def _file_mtime() -> float:
    try:
        return CONFIG_FILE.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def get_servers() -> dict[str, Any]:
    return load().get("servers", {})


def get_server(name: str) -> dict[str, Any] | None:
    return get_servers().get(name)


def add_server(name: str, server: dict[str, Any]) -> None:
    config = load()
    config.setdefault("servers", {})[name] = server
    save(config)


def remove_server(name: str) -> None:
    config = load()
    config.get("servers", {}).pop(name, None)
    save(config)


def list_server_names() -> list[str]:
    return list(get_servers().keys())


# ---------------------------------------------------------------------------
# Group management
# ---------------------------------------------------------------------------

def get_groups() -> dict[str, Any]:
    return load().get("groups", {})


def get_group(name: str) -> dict[str, Any] | None:
    return get_groups().get(name)


def add_group(name: str, group: dict[str, Any]) -> None:
    config = load()
    config.setdefault("groups", {})[name] = group
    save(config)


def remove_group(name: str) -> None:
    config = load()
    config.get("groups", {}).pop(name, None)
    save(config)


def list_group_names() -> list[str]:
    return list(get_groups().keys())


def _resolve_refs(refs: list[str]) -> list[tuple[str, str, dict, dict]]:
    """
    Resolve agent references from a list of "server/alias" strings.

    Returns list of (server_name, alias, agent_cfg, server_cfg).
    Skips references that can't be resolved.
    """
    servers = get_servers()
    result = []
    for ref in refs:
        parts = str(ref).split("/", 1)
        if len(parts) != 2:
            continue
        server_name, alias = parts
        server_cfg = servers.get(server_name)
        if server_cfg is None:
            continue
        agent_cfg = server_cfg.get("agents", {}).get(alias)
        if agent_cfg is None:
            continue
        result.append((server_name, alias, agent_cfg, server_cfg))
    return result


def get_agents_for_group(group_name: str) -> list[tuple[str, str, dict, dict]]:
    """
    Resolve agents for a group (from the 'agents' list).

    Returns list of (server_name, alias, agent_cfg, server_cfg).
    Skips references that can't be resolved.
    """
    group = get_group(group_name)
    if group is None:
        return []
    return _resolve_refs(group.get("agents", []))


def get_direct_agents_for_group(group_name: str) -> list[tuple[str, str, dict, dict]]:
    """
    Resolve direct-tool agents for a group (from the 'direct_tools' list).

    Returns list of (server_name, alias, agent_cfg, server_cfg).
    Skips references that can't be resolved.
    """
    group = get_group(group_name)
    if group is None:
        return []
    return _resolve_refs(group.get("direct_tools", []))


# ---------------------------------------------------------------------------
# Legacy compat: profiles → groups alias
# (allows old code/tests that call list_profile_names / get_profile to keep working)
# ---------------------------------------------------------------------------

def list_profile_names() -> list[str]:
    """Alias for list_group_names (backward compat)."""
    return list_group_names()


def get_profile(name: str) -> dict[str, Any] | None:
    """Backward compat: return group as a profile-like dict with 'agents' key."""
    agents_resolved = get_agents_for_group(name)
    if not agents_resolved and get_group(name) is None:
        return None
    # Build a flat agents dict keyed by alias (last-write-wins on duplicate alias)
    agents: dict[str, dict] = {}
    for server_name, alias, agent_cfg, server_cfg in agents_resolved:
        agents[alias] = {
            **agent_cfg,
            "_server": server_name,
            "_base_url": server_cfg.get("base_url", ""),
            "_api_key": server_cfg.get("api_key", ""),
            "_username": server_cfg.get("username", "astra_mcp"),
        }
    return {"agents": agents}


def get_profiles() -> dict[str, Any]:
    """Backward compat: return all groups as profile-like dicts."""
    return {name: get_profile(name) for name in list_group_names()}


def all_agents() -> list[tuple[str, str, dict]]:
    """Return [(group_name, alias, agent_dict), ...] for all configured groups."""
    result = []
    for group_name in list_group_names():
        for _, alias, agent_cfg, _ in get_agents_for_group(group_name):
            result.append((group_name, alias, agent_cfg))
    return result


def get_agents_for_profile(profile_name: str) -> dict[str, dict]:
    """Backward compat: return {alias: agent_dict} for a group."""
    prof = get_profile(profile_name)
    if prof is None:
        return {}
    return prof.get("agents", {})


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------

def on_reload(callback: Callable[[], None]) -> None:
    """Register a callback to be called when config file changes."""
    if callback not in _reload_callbacks:
        _reload_callbacks.append(callback)


def _notify_reload() -> None:
    for cb in _reload_callbacks:
        try:
            cb()
        except Exception:
            pass


def _watch_config() -> None:
    global _last_mtime
    _last_mtime = _file_mtime()
    while not _reload_stop.is_set():
        _reload_stop.wait(timeout=1.0)
        current = _file_mtime()
        if current != _last_mtime and current > 0:
            _last_mtime = current
            _notify_reload()


def start_watcher() -> None:
    """Start background thread watching config file for changes."""
    global _reload_thread
    if _reload_thread is not None and _reload_thread.is_alive():
        return
    _reload_stop.clear()
    _reload_thread = threading.Thread(target=_watch_config, daemon=True)
    _reload_thread.start()


def stop_watcher() -> None:
    """Stop the config file watcher."""
    _reload_stop.set()
    if _reload_thread is not None:
        _reload_thread.join(timeout=2.0)


def reload() -> dict[str, Any]:
    """Force reload and notify callbacks. Returns new config."""
    _notify_reload()
    return load()


def setup_signal_handler() -> None:
    """Setup SIGHUP handler to reload config (Unix only)."""
    if os.name != "nt":
        def _handle_sighup(signum, frame):
            reload()
        try:
            signal.signal(signal.SIGHUP, _handle_sighup)
        except (OSError, ValueError):
            pass
