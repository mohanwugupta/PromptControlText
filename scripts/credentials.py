"""Create/check local credentials and pass them to a command without shell expansion."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "RUNPOD_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
)
OPTIONAL_KEYS = ("ANTHROPIC_WORKSPACE_ID",)


def initialize(path):
    """Create an owner-only file; never overwrite existing credentials."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as stream:
        stream.write((ROOT / ".env.example").read_text())
    return True


def credential_environment(path, environ=None):
    from dotenv import dotenv_values

    env = dict(os.environ if environ is None else environ)
    # Treat credential values literally, including dollar signs. Never source
    # this file in a shell or allow it to override PATH or unrelated settings.
    values = dotenv_values(path, interpolate=False) if path.exists() else {}
    for key in KEYS + OPTIONAL_KEYS:
        if not env.get(key) and values.get(key):
            env[key] = values[key]
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("init", help="Create .env with owner-only permissions")
    actions.add_parser("status", help="Show SET/MISSING only; no network calls")
    run = actions.add_parser("run", help="Run a command with credentials in its environment")
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.action == "init":
        try:
            created = initialize(args.env_file)
        except OSError:
            print("Could not create the credential file; check its directory and permissions.", file=sys.stderr)
            return 2
        print("Created local credential file (mode 600)." if created else "Credential file already exists; left unchanged.")
        return 0

    try:
        env = credential_environment(args.env_file)
    except ImportError:
        print("Install the existing setup dependency: python -m pip install python-dotenv", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("Could not read the credential file; check its format and permissions.", file=sys.stderr)
        return 2

    if args.action == "status":
        for key in KEYS:
            print(f"{key}: {'SET' if env.get(key) else 'MISSING'}")
        for key in OPTIONAL_KEYS:
            print(f"{key}: {'SET' if env.get(key) else 'UNSET (optional)'}")
        print("Presence only: validity, billing, permissions, and model access are not verified.")
        return 0 if all(env.get(key) for key in KEYS) else 1

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run requires a command after --")
    try:
        os.execvpe(command[0], command, env)
    except OSError:
        print("Could not start the requested command; check its installation and executable permissions.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
