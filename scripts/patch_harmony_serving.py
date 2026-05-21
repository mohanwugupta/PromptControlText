"""
scripts/patch_harmony_serving.py

Patches the installed vLLM render/serving.py so that HarmonyError (vocab file
unavailable on compute nodes) is caught and falls back to standard chat-template
rendering.

Run this ONCE after activating the PromptControlText conda env, before starting
the gpt-oss vLLM server:

    conda activate PromptControlText
    python scripts/patch_harmony_serving.py

It is idempotent — running it twice is safe.
"""
from __future__ import annotations

import pathlib
import re
import sys

# ---------------------------------------------------------------------------
# Locate the file
# ---------------------------------------------------------------------------
try:
    import vllm  # type: ignore
    vllm_root = pathlib.Path(vllm.__file__).parent
except ImportError:
    print("❌ vllm not found in the current Python environment.")
    sys.exit(1)

target = vllm_root / "entrypoints" / "serve" / "render" / "serving.py"
if not target.exists():
    print(f"❌ Target file not found: {target}")
    print("   vLLM version may differ from expected. Inspect the traceback and")
    print("   adjust the patch script manually.")
    sys.exit(1)

print(f"Patching: {target}")
original = target.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Guard: already patched?
# ---------------------------------------------------------------------------
PATCH_MARKER = "# PCT_HARMONY_PATCH_APPLIED"
if PATCH_MARKER in original:
    print("✅ Already patched — nothing to do.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Patch 1: Add HarmonyError import (near the top, after existing imports)
# ---------------------------------------------------------------------------
IMPORT_SNIPPET = """\
try:
    from openai_harmony import HarmonyError as _HarmonyError  # type: ignore
except ImportError:
    _HarmonyError = Exception  # fallback so the except clause is always valid
"""

# Insert after the last top-level `import` / `from … import` block.
# We look for the first blank line that follows an import statement.
patched, n_import = re.subn(
    r"((?:^(?:import |from )\S.*\n)+)",
    lambda m: m.group(0) + IMPORT_SNIPPET,
    original,
    count=1,
    flags=re.MULTILINE,
)

if n_import == 0:
    # Fallback: just prepend after the module docstring (if any)
    patched = IMPORT_SNIPPET + "\n" + original

# ---------------------------------------------------------------------------
# Patch 2: Wrap `_make_request_with_harmony` in render_chat with try/except
#
# Target pattern (indentation may vary):
#   conversation, engine_inputs = self._make_request_with_harmony(
#       request, should_include_tools
#   )
#
# We want:
#   try:
#       conversation, engine_inputs = self._make_request_with_harmony(
#           request, should_include_tools
#       )
#   except _HarmonyError:
#       conversation, engine_inputs = self._make_request_without_harmony(
#           request, should_include_tools
#       )
# ---------------------------------------------------------------------------
HARMONY_CALL_PATTERN = re.compile(
    r"( +)(conversation, engine_inputs = self\._make_request_with_harmony\(\n"
    r"(?:.*\n)*?.*?\))\n",
)

def _wrap_with_fallback(m: re.Match) -> str:
    indent = m.group(1)
    inner = m.group(2)
    fallback = inner.replace("_make_request_with_harmony", "_make_request_without_harmony")
    return (
        f"{indent}try:\n"
        f"{indent}    {inner.replace(chr(10), chr(10) + indent + '    ')}\n"
        f"{indent}except _HarmonyError:\n"
        f"{indent}    # Harmony vocab unavailable (compute node, no internet). "
        f"Use standard chat-template rendering.\n"
        f"{indent}    {fallback.replace(chr(10), chr(10) + indent + '    ')}\n"
    )

patched2, n_call = HARMONY_CALL_PATTERN.subn(_wrap_with_fallback, patched)

if n_call == 0:
    print("⚠️  Could not find `_make_request_with_harmony` call pattern.")
    print("   The vLLM version may differ. Trying simpler string-based patch...")

    # Simpler fallback: find the assignment line and wrap it
    old_line = "conversation, engine_inputs = self._make_request_with_harmony("
    if old_line in patched:
        lines = patched.splitlines(keepends=True)
        result_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith("conversation, engine_inputs = self._make_request_with_harmony("):
                indent = line[: len(line) - len(line.lstrip())]
                # Collect all lines of this statement (until closing paren)
                stmt_lines = [line]
                depth = line.count("(") - line.count(")")
                j = i + 1
                while depth > 0 and j < len(lines):
                    stmt_lines.append(lines[j])
                    depth += lines[j].count("(") - lines[j].count(")")
                    j += 1
                stmt = "".join(stmt_lines)
                fallback_stmt = stmt.replace(
                    "_make_request_with_harmony", "_make_request_without_harmony"
                )
                result_lines.append(f"{indent}try:\n")
                for sl in stmt.splitlines(keepends=True):
                    result_lines.append(f"{indent}    {sl.lstrip()}")
                result_lines.append(
                    f"{indent}except _HarmonyError:\n"
                    f"{indent}    # Harmony vocab unavailable; fall back to standard rendering.\n"
                )
                for sl in fallback_stmt.splitlines(keepends=True):
                    result_lines.append(f"{indent}    {sl.lstrip()}")
                i = j
                continue
            result_lines.append(line)
            i += 1
        patched2 = "".join(result_lines)
        n_call = 1
    else:
        print("❌ Patch target not found. The vLLM source may have changed.")
        print("   Manual patch required — add a try/except around the")
        print("   _make_request_with_harmony call in render/serving.py:")
        print()
        print("       try:")
        print("           conversation, engine_inputs = self._make_request_with_harmony(...)")
        print("       except HarmonyError:")
        print("           conversation, engine_inputs = self._make_request_without_harmony(...)")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Add the patch marker so we can detect it next run
# ---------------------------------------------------------------------------
patched2 = patched2.rstrip("\n") + f"\n\n{PATCH_MARKER}\n"

# ---------------------------------------------------------------------------
# Write (with backup)
# ---------------------------------------------------------------------------
backup = target.with_suffix(".py.bak")
backup.write_text(original, encoding="utf-8")
target.write_text(patched2, encoding="utf-8")

print(f"✅ Patch applied successfully.")
print(f"   Backup saved to: {backup}")
print(f"   Import block injected: {n_import > 0}")
print(f"   Harmony call wrapped:  {n_call > 0}")
