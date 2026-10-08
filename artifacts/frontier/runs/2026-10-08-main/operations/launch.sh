#!/bin/bash
cd /workspace/PromptControlText || exit 90
.venv/bin/python -u -m frontier.main_run --watch-minutes 120
main_exit=$?
printf '%s\n' "$main_exit" > .local/frontier/main-20261008/exit-code
exit "$main_exit"
