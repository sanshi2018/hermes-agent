"""Regression tests for background-launch failure propagation."""

import json

import tools.process_registry as process_registry_module
import tools.terminal_tool as terminal_tool_module


def test_remote_background_launcher_without_pid_returns_failure(monkeypatch):
    """A registry failed_start must not become a successful tool response."""

    class FakeRemoteEnv:
        cwd = "/workspace"

        @staticmethod
        def get_temp_dir():
            return "/tmp"

        @staticmethod
        def execute(command, **kwargs):
            return {"output": "launcher syntax error", "returncode": 2}

    registry = process_registry_module.ProcessRegistry()
    config = {
        "env_type": "ssh",
        "cwd": "/workspace",
        "timeout": 60,
        "host_cwd": None,
    }

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_active_environments",
        {"default": FakeRemoteEnv()},
    )
    monkeypatch.setattr(terminal_tool_module, "_last_activity", {"default": 0.0})
    monkeypatch.setattr(process_registry_module, "process_registry", registry)

    result = json.loads(
        terminal_tool_module.terminal_tool(
            command="echo hello",
            background=True,
        )
    )

    assert result == {
        "output": "launcher syntax error",
        "exit_code": 2,
        "error": "Failed to start background process",
        "status": "error",
    }
    assert registry._running == {}
