"""FastMCP server — daemon mode mounts each group at /mcp/<group>."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from astramcp import config as cfg
from astramcp.client import AstrBotClient, TaskStatus, get_task, task_store
from fastmcp import FastMCP

# Group agent cache: { group_name: [(server_name, alias, agent_cfg, server_cfg), ...] }
_group_agents: dict[str, list] = {}

# Flag: are we running inside the daemon process?
_running_in_daemon: bool = False


def _refresh_agents() -> None:
    global _group_agents
    _group_agents = {}
    for group_name in cfg.list_group_names():
        _group_agents[group_name] = cfg.get_agents_for_group(group_name)


def _make_client(server_cfg: dict) -> AstrBotClient:
    return AstrBotClient(
        base_url=server_cfg.get("base_url", "http://localhost:6185"),
        api_key=server_cfg.get("api_key", ""),
        username=server_cfg.get("username", "astramcp"),
    )


def _resolve_agent(group: str, alias: str) -> tuple[AstrBotClient, dict] | None:
    """Find (client, agent_cfg) for a given group + alias."""
    for server_name, ag_alias, agent_cfg, server_cfg in _group_agents.get(group, []):
        if ag_alias == alias:
            return _make_client(server_cfg), agent_cfg
    return None


# ---------------------------------------------------------------------------
# Build per-group MCP server (3 tools)
# ---------------------------------------------------------------------------

def build_mcp_server(group: str) -> FastMCP:
    """Build a FastMCP instance for a single group."""
    if cfg.get_group(group) is None:
        raise ValueError(
            f"Group '{group}' not found. Available: {', '.join(cfg.list_group_names())}"
        )

    _refresh_agents()
    mcp = FastMCP(f"astramcp-{group}")

    @mcp.tool()
    async def list_agents() -> str:
        """
        List all available AstrBot agents for the current group.

        Returns a list of agents with their config_id and name.
        Use this first to discover available agents before calling them.
        """
        entries = _group_agents.get(group, [])
        if not entries:
            return f"No agents configured for group '{group}'."

        lines = [f"Group: {group}", f"Available agents ({len(entries)}):", ""]
        for server_name, alias, agent_cfg, _ in entries:
            cid = agent_cfg.get("config_id", "-")
            cname = agent_cfg.get("config_name", alias)
            lines.append(f"  - {alias} (server: {server_name}): config_id={cid}, name={cname}")
        return "\n".join(lines)

    @mcp.tool()
    async def call_agent(agent: str, message: str, background: bool = False) -> str:
        """
        Send a message to a specific AstrBot agent and get its response.

        Args:
            agent:      The agent alias (use list_agents to see available agents).
            message:    The message to send to the agent.
            background: If True, run asynchronously and return a task_id immediately.
                        Use poll_result(task_id) to retrieve the result later.
                        NOTE: background=True requires the daemon to be running
                        (start with `astra daemon`).
        """
        if background and not _running_in_daemon:
            return (
                "Background tasks require the daemon to be running. "
                "Please start it with `astra daemon` and connect via the MCP endpoint."
            )

        resolved = _resolve_agent(group, agent)
        if resolved is None:
            return f"Agent '{agent}' not found in group '{group}'. Use list_agents to see available agents."

        client, agent_cfg = resolved
        config_id = agent_cfg.get("config_id")
        config_name = agent_cfg.get("config_name")

        if background:
            task = client.chat_background(
                message=message,
                config_id=config_id,
                config_name=config_name,
                profile=group,
                agent=agent,
            )
            return (
                f"Task started in background.\n"
                f"task_id: {task.task_id}\n"
                f"Use poll_result(task_id='{task.task_id}') to check status."
            )

        return await client.chat_async(
            message=message,
            config_id=config_id,
            config_name=config_name,
        )

    @mcp.tool()
    async def poll_result(task_id: str) -> str:
        """
        Poll the status of a background agent task.

        Args:
            task_id: The task_id returned by call_agent with background=True.

        Returns the result when done, current status if still running,
        or error message if the task failed.
        """
        task = get_task(task_id)
        if task is None:
            return f"Unknown task_id: {task_id}. Task may have expired or never existed."

        if task.status == TaskStatus.DONE:
            return f"Task completed.\n\nResult:\n{task.result}"
        if task.status == TaskStatus.ERROR:
            return f"Task failed.\n\nError: {task.error}"
        if task.status == TaskStatus.RUNNING:
            elapsed = time.time() - task.created_at if task.created_at else 0
            return f"Task is still running (elapsed: {elapsed:.1f}s). Try again later."
        return f"Task status: {task.status.value}"

    return mcp


# ---------------------------------------------------------------------------
# Daemon: FastAPI app with each group mounted at /mcp/<group>
# ---------------------------------------------------------------------------

def build_daemon_app(group: str | None = None) -> Any:
    """
    Build the daemon FastAPI app.

    Each group is mounted as a FastMCP HTTP server at /mcp/<group>.
    Management endpoints are available under /api/.
    """
    global _running_in_daemon
    from fastapi import FastAPI

    _running_in_daemon = True
    _refresh_agents()
    cfg.on_reload(_refresh_agents)

    groups_to_serve = [group] if group else cfg.list_group_names()
    mcp_apps: list[tuple[str, Any]] = []
    for group_name in groups_to_serve:
        try:
            mcp = build_mcp_server(group_name)
            mcp_app = mcp.http_app(path="/")
            mcp_apps.append((group_name, mcp_app))
        except ValueError:
            pass

    @contextlib.asynccontextmanager
    async def combined_lifespan(app: Any):
        async with contextlib.AsyncExitStack() as stack:
            for _, mcp_app in mcp_apps:
                await stack.enter_async_context(mcp_app.lifespan(app))
            yield

    app = FastAPI(title="astramcp daemon", lifespan=combined_lifespan)

    for group_name, mcp_app in mcp_apps:
        app.mount(f"/mcp/{group_name}", mcp_app)

    # ----------------------------------------------------------------
    # Management API
    # ----------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "groups": cfg.list_group_names()}

    @app.get("/api/agents")
    async def api_list_agents(group_name: str | None = None) -> dict:
        """List agents, optionally filtered by group."""
        if group_name:
            entries = _group_agents.get(group_name, [])
            return {
                "group": group_name,
                "agents": [
                    {"server": sn, "alias": al, **ag}
                    for sn, al, ag, _ in entries
                ],
            }
        all_agents = []
        for gname, entries in _group_agents.items():
            for sn, al, ag, _ in entries:
                all_agents.append({"group": gname, "server": sn, "alias": al, **ag})
        return {"agents": all_agents}

    @app.post("/api/reload")
    async def api_reload() -> dict:
        """Force reload config."""
        cfg.reload()
        _refresh_agents()
        return {"status": "ok", "groups": cfg.list_group_names()}

    @app.get("/api/poll/{task_id}")
    async def api_poll_result(task_id: str) -> dict:
        from fastapi import HTTPException
        task = get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "result": task.result if task.status == TaskStatus.DONE else None,
            "error": task.error if task.status == TaskStatus.ERROR else None,
        }

    return app


def run_mcp_stdio(group: str) -> None:
    """Run MCP server in stdio mode for a group."""
    mcp = build_mcp_server(group)
    mcp.run(transport="stdio")


def run_daemon(group: str | None, host: str = "127.0.0.1", port: int = 18765) -> None:
    """Run daemon HTTP server with all groups mounted at /mcp/<group>."""
    import uvicorn

    cfg.setup_signal_handler()
    cfg.start_watcher()

    app = build_daemon_app(group)

    groups = cfg.list_group_names()
    print(f"[astramcp] Daemon starting on http://{host}:{port}")
    for g in ([group] if group else groups):
        print(f"[astramcp]   MCP endpoint: http://{host}:{port}/mcp/{g}")
    print(f"[astramcp] Reload: SIGHUP or POST /api/reload")

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        cfg.stop_watcher()
