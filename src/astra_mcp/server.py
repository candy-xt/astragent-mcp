"""FastMCP server — daemon with dynamic routing and hot-reload."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

from astra_mcp import config as cfg
from astra_mcp.client import (
    AstrBotClient,
    TaskStatus,
    get_task,
    task_store,
)
from fastmcp import FastMCP

# Running daemon flag
_running_in_daemon: bool = False


def _make_client(server_cfg: dict) -> AstrBotClient:
    return AstrBotClient(
        base_url=server_cfg.get("base_url", "http://localhost:6185"),
        api_key=server_cfg.get("api_key", ""),
        username=server_cfg.get("username", "astra_mcp"),
    )


# ---------------------------------------------------------------------------
# Build per-group MCP server
# ---------------------------------------------------------------------------

def build_mcp_server(group: str) -> FastMCP:
    """Build a FastMCP instance for a single group.

    Tool registration is dynamic based on config:
      - If group has *agents* → creates ``list_agents`` + ``call_agent``
      - If group has *direct_tools* → creates one ``astra_<alias>`` per entry
      - ``poll_result`` is **always** available.
    """
    if cfg.get_group(group) is None:
        raise ValueError(
            f"Group '{group}' not found. Available: {', '.join(cfg.list_group_names())}"
        )

    mcp = FastMCP(f"astra_mcp-{group}")

    # Resolve agent lists fresh from config each time
    agents = cfg.get_agents_for_group(group)
    direct_agents = cfg.get_direct_agents_for_group(group)

    # ------------------------------------------------------------------
    # 1. list_agents + call_agent  (only if agents list non-empty)
    # ------------------------------------------------------------------
    if agents:

        @mcp.tool(output_schema=None)
        async def list_agents() -> str:
            """
            List all available AstrBot agents for the current group.

            Returns a list of agents with their config_id, name, and optional description.
            Use this first to discover available agents before calling them.
            """
            entries = cfg.get_agents_for_group(group)
            if not entries:
                return f"No agents configured for group '{group}'."

            lines = [f"Group: {group}", f"Available agents ({len(entries)}):", ""]
            for server_name, alias, agent_cfg, _ in entries:
                cid = agent_cfg.get("config_id", "-")
                cname = agent_cfg.get("config_name", alias)
                desc = agent_cfg.get("description", "")
                line = f"  - {alias} (server: {server_name}): config_id={cid}, name={cname}"
                if desc:
                    line += f" — {desc}"
                lines.append(line)
            return "\n".join(lines)

        @mcp.tool(output_schema=None)
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

            entries = cfg.get_agents_for_group(group)
            resolved = None
            for server_name, alias, agent_cfg, server_cfg in entries:
                if alias == agent:
                    resolved = (_make_client(server_cfg), agent_cfg)
                    break
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

    else:
        # No agents configured — register a dummy tool as a warning
        @mcp.tool(output_schema=None)
        async def list_agents() -> str:
            """No agents are configured for this group."""
            return (
                f"No agents are configured for group '{group}'. "
                f"Add entries to the `agents` list in your config file."
            )

    # ------------------------------------------------------------------
    # 2. Direct tools  (only if direct_tools non-empty)
    # ------------------------------------------------------------------
    if direct_agents:
        def _register_direct(
            name: str,
            desc: str,
            client: AstrBotClient,
            config_id: str | None,
            config_name: str | None,
            alias: str,
        ) -> None:
            @mcp.tool(name=name, description=desc, output_schema=None)
            async def direct_tool(message: str, background: bool = False) -> str:
                """
                Send a message to this agent and get its response.

                Args:
                    message:    The message to send to the agent.
                    background: If True, run asynchronously and return a task_id immediately.
                                Use poll_result(task_id) to retrieve the result later.
                """
                if background and not _running_in_daemon:
                    return (
                        "Background tasks require the daemon to be running. "
                        "Please start it with `astra daemon` and connect via the MCP endpoint."
                    )

                if background:
                    task = client.chat_background(
                        message=message,
                        config_id=config_id,
                        config_name=config_name,
                        profile=group,
                        agent=alias,
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
            return direct_tool

        for server_name, alias, agent_cfg, server_cfg in direct_agents:
            desc = agent_cfg.get("description", f"Call the {alias} agent in group '{group}'")
            _register_direct(
                name=f"astra_{alias}",
                desc=desc,
                client=_make_client(server_cfg),
                config_id=agent_cfg.get("config_id"),
                config_name=agent_cfg.get("config_name"),
                alias=alias,
            )

    # ------------------------------------------------------------------
    # 3. poll_result  — always available
    # ------------------------------------------------------------------
    @mcp.tool(output_schema=None)
    async def poll_result(task_id: str) -> str:
        """
        Poll the status of a background agent task.

        Args:
            task_id: The task_id returned by call_agent or astra_<agent> with background=True.

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
            partial = task.result or ""
            msg = f"Task is still running (elapsed: {elapsed:.1f}s)."
            if partial:
                msg += f"\n\nPartial result:\n{partial}"
            return msg
        return f"Task status: {task.status.value}"

    return mcp


# ---------------------------------------------------------------------------
# Daemon: FastAPI app with mounted per-group MCP sub-apps
# ---------------------------------------------------------------------------

# Cache of Starlette sub-apps (from FastMCP.http_app()) keyed by group name
_mcp_starlette_apps: dict[str, Any] = {}

# Lifespan management: each running Starlette sub-app has a stop Event
# so we can signal shutdown.  Multiple instances per group may coexist
# during hot-reload (old lifespan keeps existing SSE connections alive).
_lifespan_stop_events: dict[int, asyncio.Event] = {}
_main_loop: asyncio.AbstractEventLoop | None = None


async def _run_group_lifespan(app: Any, stop: asyncio.Event) -> None:
    """Run a single FastMCP Starlette sub-app's lifespan."""
    try:
        async with app.lifespan(app):
            await stop.wait()
    except Exception:
        pass


def _start_lifespan(app: Any) -> None:
    """Fire-and-forget a sub-app's lifespan task on the main event loop."""
    stop = asyncio.Event()
    _lifespan_stop_events[id(app)] = stop
    asyncio.create_task(_run_group_lifespan(app, stop))


def _stop_lifespan(app: Any) -> None:
    """Signal a sub-app's lifespan to stop."""
    evt = _lifespan_stop_events.pop(id(app), None)
    if evt is not None:
        evt.set()


def _rebuild_all_groups() -> None:
    """Rebuild MCP server caches for all configured groups.

    Old Starlette apps are ***kept alive*** (existing connections remain)
    while new apps are created in their place.
    """
    new_groups = set(cfg.list_group_names())

    # Remove groups that no longer exist in config
    for g in list(_mcp_starlette_apps.keys()):
        if g not in new_groups:
            old_app = _mcp_starlette_apps.pop(g, None)
            if old_app is not None:
                _stop_lifespan(old_app)

    # Rebuild every configured group (creates new FastMCP + Starlette)
    for g in new_groups:
        if cfg.get_group(g) is None:
            _mcp_starlette_apps.pop(g, None)
            continue
        try:
            mcp = build_mcp_server(g)
            _mcp_starlette_apps[g] = mcp.http_app(path="/")
        except ValueError:
            _mcp_starlette_apps.pop(g, None)


def _mount_group(app: Any, group: str) -> None:
    """Mount a group's MCP sub-app to the parent FastAPI app."""
    from starlette.routing import Mount

    starlette_app = _mcp_starlette_apps.get(group)
    if starlette_app is None:
        return
    # Remove any existing mount for this group
    app.router.routes = [
        r for r in app.router.routes
        if not (isinstance(r, Mount) and getattr(r, "name", None) == f"mcp-{group}")
    ]
    app.mount(f"/mcp/{group}", starlette_app, name=f"mcp-{group}")


async def _sync_mounts_and_lifespans(app: Any) -> None:
    """Async reconcilation: update mounts + start lifespans for new apps."""
    for g, starlette_app in list(_mcp_starlette_apps.items()):
        _mount_group(app, g)
        # Start lifespan for this instance if not already running
        if id(starlette_app) not in _lifespan_stop_events:
            _start_lifespan(starlette_app)

    # Unmount removed groups
    from starlette.routing import Mount
    kept = set(_mcp_starlette_apps.keys())
    app.router.routes = [
        r for r in app.router.routes
        if not (
            isinstance(r, Mount)
            and getattr(r, "name", "").startswith("mcp-")
            and r.name.replace("mcp-", "", 1) not in kept
        )
    ]


def build_daemon_app(group: str | None = None) -> Any:
    """
    Build the daemon FastAPI app.

    Each group's FastMCP server is mounted as a Starlette sub-application
    at ``/mcp/<group>`` with proper lifespan chaining.

    On config reload:
      * Agent changes within existing groups take effect immediately
        (tools read config directly — no rebuild needed).
      * ``direct_tools`` changes rebuild the FastMCP instance and start
        a *new* lifespan for the new sub-app; the *old* lifespan stays
        alive so existing SSE connections are not disrupted.
      * Group additions / removals update the route table and lifespans.
    """
    global _running_in_daemon, _main_loop
    from fastapi import FastAPI, HTTPException
    from starlette.responses import StreamingResponse

    _running_in_daemon = True
    _rebuild_all_groups()

    @contextlib.asynccontextmanager
    async def combined_lifespan(app: Any):
        global _main_loop
        _main_loop = asyncio.get_event_loop()

        # Mount and start lifespans for all current groups
        for g, sa in list(_mcp_starlette_apps.items()):
            _mount_group(app, g)
            _start_lifespan(sa)

        try:
            yield
        finally:
            # Daemon shutdown: stop all sub-app lifespans
            for evt in list(_lifespan_stop_events.values()):
                evt.set()
            _lifespan_stop_events.clear()

    app = FastAPI(title="astra_mcp daemon", lifespan=combined_lifespan)

    # ------------------------------------------------------------------
    # Config reload handler (called from watcher thread → bridge to async)
    # ------------------------------------------------------------------
    def _on_config_reload() -> None:
        print("[astra_mcp] Config file changed — rebuilding MCP apps ...")
        _rebuild_all_groups()
        if _main_loop is not None and not _main_loop.is_closed():
            fut = asyncio.run_coroutine_threadsafe(
                _sync_mounts_and_lifespans(app), _main_loop
            )
            fut.result(timeout=10)
        print(f"[astra_mcp] Reload complete. Groups: {', '.join(cfg.list_group_names()) or '(none)'}")

    cfg.on_reload(_on_config_reload)

    # ------------------------------------------------------------------
    # Management API
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "groups": cfg.list_group_names()}

    @app.get("/api/agents")
    async def api_list_agents(group_name: str | None = None) -> dict:
        """List agents, optionally filtered by group."""
        if group_name:
            entries = cfg.get_agents_for_group(group_name)
            return {
                "group": group_name,
                "agents": [
                    {"server": sn, "alias": al, **ag}
                    for sn, al, ag, _ in entries
                ],
            }
        all_agents = []
        for gname in cfg.list_group_names():
            for sn, al, ag, _ in cfg.get_agents_for_group(gname):
                all_agents.append({"group": gname, "server": sn, "alias": al, **ag})
        return {"agents": all_agents}

    @app.post("/api/reload")
    async def api_reload() -> dict:
        """Force reload config and rebuild MCP apps."""
        print("[astra_mcp] Reload triggered via API — rebuilding ...")
        cfg.reload()
        _rebuild_all_groups()
        await _sync_mounts_and_lifespans(app)
        groups = cfg.list_group_names()
        print(f"[astra_mcp] Reload complete. Groups: {', '.join(groups) or '(none)'}")
        return {"status": "ok", "groups": groups}

    @app.get("/api/poll/{task_id}")
    async def api_poll_result(task_id: str) -> dict:
        task = get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "result": task.result if task.status == TaskStatus.DONE else task.result,
            "error": task.error if task.status == TaskStatus.ERROR else None,
            "partial": task.result if task.status == TaskStatus.RUNNING else None,
        }

    @app.get("/api/tasks")
    async def api_list_tasks(group: str | None = None, limit: int = 100) -> dict:
        """List background tasks, optionally filtered by group."""
        if group:
            tasks = task_store.list_by_group(group, limit=limit)
        else:
            tasks = task_store.list_tasks(limit=limit)
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "session_id": t.session_id,
                    "profile": t.profile,
                    "agent": t.agent,
                    "status": t.status.value,
                    "result": t.result,
                    "error": t.error,
                    "created_at": t.created_at,
                    "finished_at": t.finished_at,
                }
                for t in tasks
            ]
        }

    @app.get("/api/tasks/{task_id}")
    async def api_get_task(task_id: str) -> dict:
        task = get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "profile": task.profile,
            "agent": task.agent,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "finished_at": task.finished_at,
        }

    @app.get("/api/tasks/{task_id}/stream")
    async def api_task_stream(task_id: str) -> StreamingResponse:
        """SSE endpoint that streams task progress in real-time."""
        task = get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        async def event_generator():
            last_result = ""
            while True:
                current = get_task(task_id)
                if current is None:
                    yield f"data: {json.dumps({'type': 'error', 'data': 'Task not found'})}\n\n"
                    break

                payload = {
                    "task_id": current.task_id,
                    "status": current.status.value,
                    "result": current.result,
                    "error": current.error,
                    "created_at": current.created_at,
                    "finished_at": current.finished_at,
                }

                if current.result != last_result or current.status in (
                    TaskStatus.DONE, TaskStatus.ERROR
                ):
                    yield f"data: {json.dumps(payload)}\n\n"
                    last_result = current.result or ""

                if current.status in (TaskStatus.DONE, TaskStatus.ERROR):
                    break

                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def run_mcp_stdio(group: str) -> None:
    """Run MCP server in stdio mode for a group."""
    mcp = build_mcp_server(group)
    mcp.run(transport="stdio")


def run_daemon(group: str | None, host: str = "127.0.0.1", port: int = 18765) -> None:
    """Run daemon HTTP server with dynamic MCP routing."""
    import uvicorn

    cfg.setup_signal_handler()
    cfg.start_watcher()

    app = build_daemon_app(group)

    groups = cfg.list_group_names()
    print(f"[astra_mcp] Daemon starting on http://{host}:{port}")
    targets = [group] if group else groups
    for g in targets:
        print(f"[astra_mcp]   MCP endpoint: http://{host}:{port}/mcp/{g}")
    print(f"[astra_mcp]   Tasks API:     http://{host}:{port}/api/tasks")
    print(f"[astra_mcp]   Task stream:   http://{host}:{port}/api/tasks/<task_id>/stream")
    print(f"[astra_mcp] Reload: SIGHUP or POST /api/reload")

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        cfg.stop_watcher()
