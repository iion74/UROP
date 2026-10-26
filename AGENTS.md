# UROP workspace instructions

## Preserve local research data

- Never delete or overwrite unrelated user changes.
- Never publish `.env`, credentials, private study logs, virtual environments,
  caches, or third-party source documents without verified redistribution rights.
- On 2026-10-26 the user authorized publication of project code, manuscripts,
  figures, review notes, experiment reports and learned policy models. Respect
  the reviewed `.gitignore` publication scope; do not force-add ignored files.
- Keep large model bundles and numerical replay data in GitHub Releases, not
  regular Git. Do not change repository visibility or invent a software license.
- Do not launch experiments, model servers, or GPU workloads merely to sync Git.

## GitHub synchronization

The user requested completed code changes be reflected in
`https://github.com/iion74/UROP` on branch `main`.

- After implementation tasks, run relevant checks, then commit and push only
  task files: `bash scripts/sync_github.sh "Describe the change" path/to/file ...`.
- This is a task-completion workflow, not a background file watcher. Do not
  commit or push for explanation-only or review-only requests.
- Do not stage unrelated changes, consume pre-existing staged changes,
  force-push, rewrite history, or automatically merge/rebase conflicts.
- If authentication, network, permissions, or remote changes block a push,
  preserve local work and report what is unsynchronized. Never claim a
  successful GitHub update until the push succeeds.
- Report verification and commit/push status briefly at handoff.

See `docs/GITHUB_SYNC.md` for initial authentication and manual usage.
