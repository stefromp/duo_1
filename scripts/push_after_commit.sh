#!/bin/sh
# Helper script (tracked) used by the README to show what the hook does.
# Usage: ./scripts/push_after_commit.sh

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ -z "$branch" ] || [ "$branch" = "HEAD" ]; then
  echo "Detached HEAD or unknown branch. Nothing to push."
  exit 0
fi

echo "Pushing branch '$branch' to origin..."
git push -u origin "$branch"
