"""Tests for astramcp — no live AstrBot connection required."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _patch_config(module, tmp_path: Path):
    module.CONFIG_DIR = tmp_path
    module.CONFIG_FILE = tmp_path / "config.yaml"


def _sample_config() -> dict:
    return {
        "servers": {
            "local": {
                "base_url": "http://localhost:6185",
                "api_key": "",
                "username": "u",
                "agents": {
                    "coder": {"config_id": "abc", "config_name": "coding", "description": "A coding bot"},
                    "writer": {"config_id": "def", "config_name": "writing"},
                },
            }
        },
        "groups": {
            "main": {
                "agents": ["local/coder", "local/writer"],
            },
            "lite": {
                "agents": ["local/coder"],
            },
            "both": {
                "agents": ["local/writer"],
                "direct_tools": ["local/coder"],
            },
            "direct-only": {
                "direct_tools": ["local/coder"],
            },
            "empty": {},
        },
    }


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_load_defaults(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    data = config.load()
    assert data == {"servers": {}, "groups": {}}


def test_config_add_and_get_server(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.add_server("local", {
        "base_url": "http://localhost:6185",
        "api_key": "tok",
        "username": "u",
        "agents": {"coder": {"config_id": "abc", "config_name": "coding"}},
    })
    srv = config.get_server("local")
    assert srv is not None
    assert srv["base_url"] == "http://localhost:6185"
    assert "coder" in srv["agents"]


def test_config_remove_server(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.add_server("todel", {"base_url": "http://x", "api_key": "", "username": "u", "agents": {}})
    config.remove_server("todel")
    assert config.get_server("todel") is None


def test_config_add_and_get_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    grp = config.get_group("main")
    assert grp is not None
    assert "local/coder" in grp["agents"]


def test_config_remove_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    config.remove_group("lite")
    assert config.get_group("lite") is None


def test_config_list_group_names(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    names = config.list_group_names()
    assert set(names) == {"main", "lite", "both", "direct-only", "empty"}


def test_config_get_agents_for_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    entries = config.get_agents_for_group("main")
    assert len(entries) == 2
    aliases = {alias for _, alias, _, _ in entries}
    assert aliases == {"coder", "writer"}


def test_config_get_agents_for_group_lite(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    entries = config.get_agents_for_group("lite")
    assert len(entries) == 1
    assert entries[0][1] == "coder"


def test_config_get_agents_for_missing_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    entries = config.get_agents_for_group("nonexistent")
    assert entries == []


def test_config_get_agents_allows_description(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    entries = config.get_agents_for_group("main")
    coder = next((ag for _, alias, ag, _ in entries if alias == "coder"), None)
    assert coder is not None
    assert coder.get("description") == "A coding bot"


def test_config_get_direct_agents_for_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    entries = config.get_direct_agents_for_group("both")
    assert len(entries) == 1
    assert entries[0][1] == "coder"


def test_config_get_direct_agents_empty(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    assert config.get_direct_agents_for_group("main") == []


def test_config_all_agents(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())
    agents = config.all_agents()
    # main:2 + lite:1 + both:1 + direct-only:0 + empty:0 = 4
    assert len(agents) == 4


def test_config_reload(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save({"servers": {}, "groups": {}})
    callbacks_called = []
    config.on_reload(lambda: callbacks_called.append(1))
    config.reload()
    assert len(callbacks_called) >= 1


def test_config_reload_no_duplicates(tmp_path):
    """on_reload should not register the same callback twice."""
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save({"servers": {}, "groups": {}})
    calls = []
    cb = lambda: calls.append(1)
    config.on_reload(cb)
    config.on_reload(cb)
    config.reload()
    assert calls.count(1) == 1


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

def test_client_headers_with_key():
    from astramcp.client import AstrBotClient
    c = AstrBotClient("http://localhost:6185", "mykey", "u")
    assert c._headers()["Authorization"] == "Bearer mykey"


def test_client_headers_no_key():
    from astramcp.client import AstrBotClient
    c = AstrBotClient("http://localhost:6185", "", "u")
    assert "Authorization" not in c._headers()


def test_task_store_lifecycle(tmp_path):
    from astramcp.client import TaskStore, TaskStatus

    store = TaskStore(db_path=tmp_path / "tasks.db")
    task = store.create(session_id="test-sid", profile="main", agent="coder")
    assert task.task_id
    assert task.status == TaskStatus.PENDING

    fetched = store.get(task.task_id)
    assert fetched is not None
    assert fetched.task_id == task.task_id

    store.update(task.task_id, status=TaskStatus.RUNNING)
    assert store.get(task.task_id).status == TaskStatus.RUNNING

    store.update(task.task_id, result="done!", status=TaskStatus.DONE, finished_at=time.time())
    assert store.get(task.task_id).status == TaskStatus.DONE
    assert store.get(task.task_id).result == "done!"


def test_task_store_list_by_profile(tmp_path):
    from astramcp.client import TaskStore, TaskStatus

    store = TaskStore(db_path=tmp_path / "tasks.db")
    t1 = store.create(session_id="s1", profile="main", agent="coder")
    t2 = store.create(session_id="s2", profile="lite", agent="writer")
    store.update(t1.task_id, status=TaskStatus.DONE, finished_at=time.time())

    main_tasks = store.list_by_profile("main")
    assert len(main_tasks) == 1
    assert main_tasks[0].task_id == t1.task_id

    lite_tasks = store.list_by_profile("lite")
    assert len(lite_tasks) == 1
    assert lite_tasks[0].task_id == t2.task_id


def test_task_store_list_tasks(tmp_path):
    from astramcp.client import TaskStore

    store = TaskStore(db_path=tmp_path / "tasks.db")
    t1 = store.create(session_id="s1", profile="main", agent="coder")
    t2 = store.create(session_id="s2", profile="lite", agent="writer")
    all_tasks = store.list_tasks()
    assert len(all_tasks) == 2
    ids = {t.task_id for t in all_tasks}
    assert ids == {t1.task_id, t2.task_id}


def test_task_store_list_by_group(tmp_path):
    from astramcp.client import TaskStore

    store = TaskStore(db_path=tmp_path / "tasks.db")
    t1 = store.create(session_id="s1", profile="main", agent="coder")
    store.create(session_id="s2", profile="lite", agent="writer")
    group_tasks = store.list_by_group("main")
    assert len(group_tasks) == 1
    assert group_tasks[0].task_id == t1.task_id


def test_task_store_evict_old(tmp_path):
    from astramcp.client import TaskStore, TaskStatus

    store = TaskStore(db_path=tmp_path / "tasks.db")
    t1 = store.create(session_id="s1")
    store.update(t1.task_id, status=TaskStatus.DONE, finished_at=time.time() - 3600 * 24 * 8)

    store.evict_old()
    assert store.get(t1.task_id) is None


def test_task_store_persistence(tmp_path):
    from astramcp.client import TaskStore, TaskStatus

    db = tmp_path / "tasks.db"
    store1 = TaskStore(db_path=db)
    task = store1.create(session_id="persist-test", profile="main", agent="coder")
    store1.update(task.task_id, status=TaskStatus.DONE, result="hello", finished_at=time.time())

    store2 = TaskStore(db_path=db)
    fetched = store2.get(task.task_id)
    assert fetched is not None
    assert fetched.status == TaskStatus.DONE
    assert fetched.result == "hello"


def test_background_task_lifecycle(tmp_path):
    from astramcp.client import AstrBotClient, TaskStatus, TaskStore

    store = TaskStore(db_path=tmp_path / "tasks.db")
    import astramcp.client as client_mod
    orig_store = client_mod.task_store
    client_mod.task_store = store
    try:
        c = AstrBotClient("http://localhost:9999", "", "u")
        task = c.chat_background(message="hello", config_id="x", profile="main", agent="coder")
        assert task.task_id

        deadline = time.time() + 5
        while time.time() < deadline:
            t = store.get(task.task_id)
            if t and t.status in (TaskStatus.DONE, TaskStatus.ERROR):
                break
            time.sleep(0.1)

        final = store.get(task.task_id)
        assert final is not None
        assert final.status == TaskStatus.ERROR
    finally:
        client_mod.task_store = orig_store


# ---------------------------------------------------------------------------
# server — MCP tool registration
# ---------------------------------------------------------------------------

def _agent_names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_build_mcp_server_both_lists(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_mcp_server
    mcp = build_mcp_server("both")
    names = _agent_names(mcp)
    # has agents (writer) → list_agents + call_agent
    assert "list_agents" in names
    assert "call_agent" in names
    # has direct_tools (coder) → astra_coder
    assert "astra_coder" in names
    # poll_result always present
    assert "poll_result" in names


def test_build_mcp_server_agents_only(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_mcp_server
    mcp = build_mcp_server("main")
    names = _agent_names(mcp)
    assert "list_agents" in names
    assert "call_agent" in names
    assert "poll_result" in names
    # no direct_tools → no astra_ tools
    assert not any(n.startswith("astra_") for n in names)


def test_build_mcp_server_direct_only(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_mcp_server
    mcp = build_mcp_server("direct-only")
    names = _agent_names(mcp)
    # no agents → list_agents is dummy warning, no call_agent
    assert "list_agents" in names
    assert "call_agent" not in names
    assert "astra_coder" in names
    assert "poll_result" in names


def test_build_mcp_server_empty(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_mcp_server
    mcp = build_mcp_server("empty")
    names = _agent_names(mcp)
    # no agents → list_agents dummy, no call_agent
    assert "list_agents" in names
    assert "call_agent" not in names
    # no direct_tools → no astra_ tools
    assert not any(n.startswith("astra_") for n in names)
    # poll_result always
    assert "poll_result" in names


def test_build_mcp_server_unknown_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_mcp_server
    try:
        build_mcp_server("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_poll_unknown_task():
    from astramcp.client import get_task
    assert get_task("nonexistent-id") is None


def test_list_agents_shows_description(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_mcp_server
    mcp = build_mcp_server("main")

    async def _run():
        tools = await mcp.list_tools()
        list_tool = next(t for t in tools if t.name == "list_agents")
        result = await list_tool.fn()
        assert "A coding bot" in result
        assert "coding" in result
        assert "writer" in result

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# daemon app
# ---------------------------------------------------------------------------

def test_daemon_health(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "main" in data["groups"]


def test_daemon_mcp_unknown_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/mcp/nonexistent/")
        assert resp.status_code == 404


def test_daemon_list_agents(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/api/agents", params={"group_name": "main"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["group"] == "main"
        aliases = {a["alias"] for a in data["agents"]}
        assert aliases == {"coder", "writer"}


def test_daemon_list_agents_all(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        # main:2 + lite:1 + both:1 + direct-only:0 + empty:0 = 4
        assert len(resp.json()["agents"]) == 4


def test_daemon_poll_not_found(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/api/poll/nonexistent-id")
        assert resp.status_code == 404


def test_daemon_reload(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.post("/api/reload")
        assert resp.status_code == 200
        assert "main" in resp.json()["groups"]


def test_daemon_tasks_endpoint(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data


def test_daemon_task_detail_not_found(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/api/tasks/nonexistent")
        assert resp.status_code == 404


def test_daemon_task_stream_not_found(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from astramcp.server import build_daemon_app
    from starlette.testclient import TestClient

    app = build_daemon_app()
    with TestClient(app) as client:
        resp = client.get("/api/tasks/nonexistent/stream")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def test_cli_help():
    from typer.testing import CliRunner
    from astramcp.cli import app
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "daemon" in result.output


def test_cli_mcp_unknown_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save({"servers": {}, "groups": {}})

    from typer.testing import CliRunner
    from astramcp.cli import app
    result = CliRunner().invoke(app, ["mcp", "nonexistent"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_config_list_empty(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save({"servers": {}, "groups": {}})

    from typer.testing import CliRunner
    from astramcp.cli import app
    result = CliRunner().invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "No servers" in result.output


def test_cli_config_list_with_data(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save(_sample_config())

    from typer.testing import CliRunner
    from astramcp.cli import app
    result = CliRunner().invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "local" in result.output
    assert "coder" in result.output
    assert "main" in result.output


def test_cli_config_reload(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save({"servers": {}, "groups": {}})

    from typer.testing import CliRunner
    from astramcp.cli import app
    result = CliRunner().invoke(app, ["config", "reload"])
    assert result.exit_code == 0
    assert "reloaded" in result.output


def test_cli_daemon_unknown_group(tmp_path):
    from astramcp import config
    _patch_config(config, tmp_path)
    config.save({"servers": {}, "groups": {}})

    from typer.testing import CliRunner
    from astramcp.cli import app
    result = CliRunner().invoke(app, ["daemon", "nonexistent"])
    assert result.exit_code != 0
    assert "not found" in result.output
