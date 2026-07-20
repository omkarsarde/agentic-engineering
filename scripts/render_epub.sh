#!/bin/bash
# Render the EPUB locally (WSL). Chrome libs are the user-space extraction in
# ~/chrome-deps (no sudo available); execution results come from _freeze.
# Log: scratchpad/repub.log on the Windows side. Run:
#   wsl.exe bash -c "bash /mnt/c/Users/omkar/Projects/genai-senior-prep/newbook/scripts/render_epub.sh"
set -u
BOOK=/mnt/c/Users/omkar/Projects/genai-senior-prep/newbook
LOG=/mnt/c/Users/omkar/AppData/Local/Temp/claude/C--Users-omkar-Projects/e0f454e5-38f2-418a-a86f-a2666518b3b0/scratchpad/repub.log

cd "$BOOK" || exit 1
export HOME=/home/osarde
export LD_LIBRARY_PATH=/home/osarde/chrome-deps/extracted/usr/lib/x86_64-linux-gnu
export QUARTO_PYTHON=/home/osarde/newbook-venv/bin/python
# Quarto's bundled Chromium 91 is too old for its mermaid.js; use a current
# chrome-headless-shell (see setup_modern_chrome.sh).
export QUARTO_CHROMIUM=/home/osarde/chrome-modern/chrome-headless-shell-linux64/chrome-headless-shell

echo "=== epub render start: $(date) ===" > "$LOG"
/home/osarde/.local/bin/quarto render --to epub >> "$LOG" 2>&1
RC=$?
echo "QUARTO_EXIT_CODE=$RC" >> "$LOG"
ls -la "$BOOK"/_book/*.epub >> "$LOG" 2>&1
echo "=== end: $(date) ===" >> "$LOG"
exit $RC
