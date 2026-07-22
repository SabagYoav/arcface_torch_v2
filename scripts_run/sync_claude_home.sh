#!/bin/bash
# Periodically mirrors ~/.claude (session/transcript data, lives on the ephemeral
# container overlay filesystem) onto /DATA (a persistent host bind-mount), so a
# container rebuild doesn't wipe conversation history.
set -u
DEST=/DATA/.claude_persist
while true; do
  cp -a --remove-destination "$HOME/.claude/." "$DEST/" 2>/dev/null
  sleep 120
done
