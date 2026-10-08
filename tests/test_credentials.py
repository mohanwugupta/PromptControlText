import importlib.util
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "credentials.py"
spec = importlib.util.spec_from_file_location("credentials", SCRIPT)
credentials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(credentials)


def test_init_is_private_and_does_not_overwrite(tmp_path):
    path = tmp_path / ".env"
    assert credentials.initialize(path)
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    path.write_text("RUNPOD_API_KEY=existing-test-value\n")
    assert not credentials.initialize(path)
    assert path.read_text() == "RUNPOD_API_KEY=existing-test-value\n"


def test_literal_values_environment_precedence_and_allowlist(tmp_path):
    path = tmp_path / ".env"
    path.write_text("RUNPOD_API_KEY='${HOME}$(do-not-execute)'\nOPENAI_API_KEY=file-value\nPATH=untrusted\n")
    env = credentials.credential_environment(path, {"OPENAI_API_KEY": "inherited", "PATH": "original"})
    assert env["RUNPOD_API_KEY"] == "${HOME}$(do-not-execute)"
    assert env["OPENAI_API_KEY"] == "inherited"
    assert env["PATH"] == "original"


def test_optional_workspace_is_loaded_and_respects_environment(tmp_path):
    path = tmp_path / ".env"
    path.write_text("ANTHROPIC_WORKSPACE_ID=wrkspc_file\n")
    assert credentials.credential_environment(path, {})["ANTHROPIC_WORKSPACE_ID"] == "wrkspc_file"
    assert credentials.credential_environment(path, {"ANTHROPIC_WORKSPACE_ID": "wrkspc_env"})["ANTHROPIC_WORKSPACE_ID"] == "wrkspc_env"


def test_status_does_not_disclose_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text("RUNPOD_API_KEY=private-test-sentinel\n")
    env = {k: v for k, v in os.environ.items() if k not in credentials.KEYS}
    result = subprocess.run([sys.executable, str(SCRIPT), "--env-file", str(path), "status"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "RUNPOD_API_KEY: SET" in result.stdout
    assert "OPENAI_API_KEY: MISSING" in result.stdout
    assert "private-test-sentinel" not in result.stdout + result.stderr


def test_run_passes_key_to_child_without_echoing(tmp_path):
    path = tmp_path / ".env"
    path.write_text("RUNPOD_API_KEY=private-test-sentinel\n")
    env = {k: v for k, v in os.environ.items() if k not in credentials.KEYS}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--env-file", str(path), "run", "--", sys.executable,
         "-c", "import os; assert os.environ['RUNPOD_API_KEY'] == 'private-test-sentinel'; print('child OK')"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "child OK"
    assert "private-test-sentinel" not in result.stdout + result.stderr
