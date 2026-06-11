"""CLI entry point for astramcp."""

from __future__ import annotations

from typing import Optional

import typer
from pyclack.prompts import (
    confirm,
    intro,
    is_cancel,
    multiselect,
    outro,
    password,
    select,
    text,
)

from astramcp import config as cfg
from astramcp.server import run_daemon, run_mcp_stdio

app = typer.Typer(
    name="astra",
    help="Expose AstrBot agents as MCP tools via CLI.",
    no_args_is_help=True,
)

config_app = typer.Typer(help="Manage astramcp configuration.")
app.add_typer(config_app, name="config")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _abort(msg: str = "Cancelled.") -> None:
    typer.echo(f"\n{msg}")
    raise typer.Exit(1)


def _check_cancel(val, msg: str = "Cancelled.") -> None:
    if is_cancel(val):
        _abort(msg)


# ---------------------------------------------------------------------------
# astra mcp <group>  — stdio mode  (mcp-std kept as hidden alias)
# ---------------------------------------------------------------------------

def _run_mcp_stdio(group: str):
    groups = cfg.list_group_names()
    if group not in groups:
        typer.echo(f"Group '{group}' not found.")
        typer.echo(f"Available groups: {', '.join(groups) or 'none'}")
        typer.echo("Run `astra config init` to add a group.")
        raise typer.Exit(1)

    typer.echo(f"Starting MCP stdio for group: {group}", err=True)
    run_mcp_stdio(group)


@app.command("mcp")
def mcp(
    group: str = typer.Argument(
        help="Group name to expose as MCP tools.",
    ),
):
    """Start MCP server in stdio mode for a specific group.

    Exposes 3 tools: list_agents, call_agent, poll_result.
    For background/async calls, the daemon must be running.
    """
    _run_mcp_stdio(group)


@app.command("mcp-std", hidden=True)
def mcp_std(
    group: str = typer.Argument(
        help="Group name to expose as MCP tools.",
    ),
):
    """Alias for `mcp` (deprecated)."""
    _run_mcp_stdio(group)


# ---------------------------------------------------------------------------
# astra daemon [group]  — HTTP server mode
# ---------------------------------------------------------------------------

@app.command()
def daemon(
    group: Optional[str] = typer.Argument(
        default=None,
        help="Group name to serve. Omit to serve all groups.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="HTTP bind address."),
    port: int = typer.Option(18765, "--port", "-p", help="HTTP bind port."),
):
    """Start daemon HTTP server for AstrBot agents.

    Serves all configured groups via dynamic MCP routing at /mcp/<group>.
    Handles background tasks and provides task streaming via SSE.
    Required for async/background agent calls from mcp mode.

    Reload config: SIGHUP or POST /api/reload
    Tasks API:    GET /api/tasks
    Stream:       GET /api/tasks/<task_id>/stream
    """
    groups = cfg.list_group_names()
    if group and group not in groups:
        typer.echo(f"Group '{group}' not found.")
        typer.echo(f"Available groups: {', '.join(groups) or 'none'}")
        typer.echo("Run `astra config init` to add a group.")
        raise typer.Exit(1)

    run_daemon(group, host=host, port=port)


# ---------------------------------------------------------------------------
# astra config init  — interactive wizard
# ---------------------------------------------------------------------------

@config_app.command("init")
def config_init():
    """Interactively add a new AstrBot server and import its agents into a group."""
    import asyncio
    asyncio.run(_config_init_async())


async def _config_init_async():
    intro("astramcp config wizard")

    # Server name
    server_name = await text("Server name (e.g. local, home, work):", placeholder="local")
    _check_cancel(server_name)
    server_name = str(server_name).strip()
    if not server_name:
        _abort("Server name cannot be empty.")

    existing_servers = cfg.get_servers()
    if server_name in existing_servers:
        overwrite = await confirm(f"Server '{server_name}' already exists. Overwrite?")
        _check_cancel(overwrite)
        if not overwrite:
            _abort("Aborted.")

    # base_url
    base_url = await text(
        "AstrBot base URL:",
        placeholder="http://localhost:6185",
    )
    _check_cancel(base_url)
    base_url = str(base_url).strip() or "http://localhost:6185"
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        base_url = f"http://{base_url}"
        typer.echo(f"  (no scheme detected, using: {base_url})")

    # api_key
    api_key = await password("API Key (leave blank if none):")
    _check_cancel(api_key)
    api_key = str(api_key).strip()

    # username
    username = await text("Username for chat sessions:", placeholder="astramcp")
    _check_cancel(username)
    username = str(username).strip() or "astramcp"

    # Fetch configs from AstrBot
    from astramcp.client import AstrBotClient
    client = AstrBotClient(base_url=base_url, api_key=api_key, username=username)

    typer.echo("\nFetching agent configs from AstrBot...")
    try:
        configs = client.list_configs()
    except Exception as e:
        typer.echo(f"Could not connect to AstrBot: {e}", err=True)
        configs = []

    agents: dict = {}

    if configs:
        options = [
            {"label": f"{c['name']} (id: {c['id']})", "value": c}
            for c in configs
        ]
        selected = await multiselect(
            "Select agents to import (space to toggle, enter to confirm):",
            options=options,
        )
        _check_cancel(selected)

        for c in selected:
            default_alias = c["name"].replace(" ", "_").lower()
            alias = await text(
                f"Alias for '{c['name']}':",
                placeholder=default_alias,
            )
            _check_cancel(alias)
            alias = str(alias).strip() or default_alias
            agents[alias] = {
                "config_id": c["id"],
                "config_name": c["name"],
            }
    else:
        typer.echo("No configs fetched. You can add agents manually later.")

    cfg.add_server(server_name, {
        "base_url": base_url,
        "api_key": api_key,
        "username": username,
        "agents": agents,
    })

    # Offer to create a group referencing these agents
    if agents:
        make_group = await confirm(f"Create a group for these agents?")
        _check_cancel(make_group)
        if make_group:
            group_name = await text("Group name:", placeholder=server_name)
            _check_cancel(group_name)
            group_name = str(group_name).strip() or server_name
            refs = [f"{server_name}/{alias}" for alias in agents]
            cfg.add_group(group_name, {"agents": refs})
            outro(f"Server '{server_name}' and group '{group_name}' saved with {len(agents)} agent(s).")
            return

    outro(f"Server '{server_name}' saved with {len(agents)} agent(s).")


# ---------------------------------------------------------------------------
# astra config list
# ---------------------------------------------------------------------------

@config_app.command("list")
def config_list():
    """List all configured servers and groups."""
    servers = cfg.get_servers()
    groups = cfg.get_groups()

    if not servers and not groups:
        typer.echo("No servers or groups configured. Run `astra config init` to add one.")
        return

    if servers:
        typer.echo("\n[Servers]")
        for srv_name, srv in servers.items():
            typer.echo(f"\n  {srv_name}  {srv.get('base_url')}")
            agents = srv.get("agents", {})
            if agents:
                for alias, agent in agents.items():
                    cid = agent.get("config_id", "-")
                    cname = agent.get("config_name", "-")
                    typer.echo(f"    {alias}  →  id={cid}  name={cname}")
            else:
                typer.echo("    (no agents)")
    else:
        typer.echo("No servers configured.")

    if groups:
        typer.echo("\n[Groups]")
        for grp_name, grp in groups.items():
            refs = grp.get("agents", [])
            typer.echo(f"\n  {grp_name}  →  {', '.join(refs) or '(empty)'}")
    else:
        typer.echo("\nNo groups configured.")


# ---------------------------------------------------------------------------
# astra config remove
# ---------------------------------------------------------------------------

@config_app.command("remove")
def config_remove():
    """Interactively remove a configured server or group."""
    import asyncio
    asyncio.run(_config_remove_async())


async def _config_remove_async():
    intro("Remove a server or group")

    target_type = await select(
        "What do you want to remove?",
        options=[
            {"label": "Server", "value": "server"},
            {"label": "Group", "value": "group"},
        ],
    )
    _check_cancel(target_type)

    if target_type == "server":
        servers = cfg.get_servers()
        if not servers:
            typer.echo("No servers to remove.")
            return
        options = [{"label": f"{n}  ({s.get('base_url')})", "value": n}
                   for n, s in servers.items()]
        choice = await select("Select server to remove:", options=options)
        _check_cancel(choice)
        ok = await confirm(f"Remove server '{choice}'?")
        _check_cancel(ok)
        if not ok:
            _abort("Aborted.")
        cfg.remove_server(str(choice))
        outro(f"Server '{choice}' removed.")
    else:
        groups = cfg.get_groups()
        if not groups:
            typer.echo("No groups to remove.")
            return
        options = [{"label": n, "value": n} for n in groups]
        choice = await select("Select group to remove:", options=options)
        _check_cancel(choice)
        ok = await confirm(f"Remove group '{choice}'?")
        _check_cancel(ok)
        if not ok:
            _abort("Aborted.")
        cfg.remove_group(str(choice))
        outro(f"Group '{choice}' removed.")


# ---------------------------------------------------------------------------
# astra config reload
# ---------------------------------------------------------------------------

@config_app.command("reload")
def config_reload():
    """Reload configuration from disk (hot reload)."""
    cfg.reload()
    groups = cfg.list_group_names()
    typer.echo(f"Config reloaded. Groups: {', '.join(groups) or 'none'}")
