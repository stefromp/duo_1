# Git hooks helpers

This repository includes a simple *local* post-commit hook to automatically push the current branch to `origin` after you run `git commit`.

Files added:

- `.git/hooks/post-commit` (local only, not tracked): calls `git push -u origin <current-branch>` with a few retries.
- `scripts/push_after_commit.sh` (tracked): a helper script that performs the same push command and can be executed manually.

How to enable the hook

1. Make sure the hook file is executable:

```bash
chmod +x .git/hooks/post-commit
```

2. Optionally test the helper script:

```bash
./scripts/push_after_commit.sh
```

Notes and safety

- The hook only runs locally and is not pushed to the remote. This is intentional: hooks live under `.git/hooks` and are not version-controlled.
- The hook only pushes after successful commits. It will not auto-commit changes.
- Be careful: pushing will fail if remote rejects the update (non-fast-forward) or if network/authentication is required. The hook retries three times on transient failures and otherwise prints a helpful message.
- If you prefer a different behavior (auto-committing or skipping certain branches), I can update the hook logic.
