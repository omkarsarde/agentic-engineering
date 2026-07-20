#!/bin/bash
# Install a current chrome-headless-shell user-space (no sudo) for Quarto's
# mermaid pre-rendering; the quarto-bundled Chromium 91 is too old for the
# mermaid.js Quarto 1.9 ships.
set -eu
export HOME=/home/osarde
LOG=/mnt/c/Users/omkar/AppData/Local/Temp/claude/C--Users-omkar-Projects/e0f454e5-38f2-418a-a86f-a2666518b3b0/scratchpad/chrome-modern.log
{
  echo "=== start: $(date) ==="
  VER=$(curl -s https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE)
  echo "stable version: $VER"
  mkdir -p "$HOME/chrome-modern"
  cd "$HOME/chrome-modern"
  curl -sL -o shell.zip "https://storage.googleapis.com/chrome-for-testing-public/${VER}/linux64/chrome-headless-shell-linux64.zip"
  ls -la shell.zip
  /home/osarde/newbook-venv/bin/python -m zipfile -e shell.zip .
  BIN="$HOME/chrome-modern/chrome-headless-shell-linux64/chrome-headless-shell"
  chmod +x "$BIN" "$HOME/chrome-modern/chrome-headless-shell-linux64/"* 2>/dev/null || true
  export LD_LIBRARY_PATH=/home/osarde/chrome-deps/extracted/usr/lib/x86_64-linux-gnu
  echo "--- missing libs (ldd) ---"
  ldd "$BIN" 2>/dev/null | grep "not found" || echo "none missing"
  echo "--- probe ---"
  "$BIN" --headless --no-sandbox --disable-gpu --dump-dom about:blank 2>&1 | tail -2
  echo "PROBE_RC=$?"
  echo "=== end: $(date) ==="
} > "$LOG" 2>&1
