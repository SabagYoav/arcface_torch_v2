#!/bin/bash
# Runs once via devcontainer postCreateCommand. ~/.claude (Claude Code session/
# transcript data) lives on the container's ephemeral overlay filesystem, so a
# rebuild normally wipes it. /DATA is a persistent host bind-mount; if a prior
# container already backed up its ~/.claude there (see scripts_run/
# sync_claude_home.sh), adopt it here so conversation history survives rebuilds.
set -u
PERSIST=/DATA/.claude_persist
LOCAL="$HOME/.claude"

if [ -L "$LOCAL" ]; then
  exit 0  # already symlinked, nothing to do
fi

if [ -d "$PERSIST" ] && [ -n "$(ls -A "$PERSIST" 2>/dev/null)" ]; then
  rm -rf "$LOCAL"
else
  mkdir -p "$PERSIST"
  if [ -d "$LOCAL" ]; then
    cp -a "$LOCAL/." "$PERSIST/"
  fi
  rm -rf "$LOCAL"
fi

ln -s "$PERSIST" "$LOCAL"
echo "~/.claude restored from $PERSIST"
