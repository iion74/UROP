# UROP workspace instructions

## Scope

The repository holds only the NCS learning-volume web app and the paper's
final method (code, results, manuscript). Earlier exploratory experiments were
removed on 2026-10-27 at the user's request; they remain in Git history.

## Preserve local research data

- Never delete or overwrite unrelated user changes.
- Never publish `.env`, credentials, private study logs (`data/index/`), NCS
  source PDFs (`data/ncs/`), virtual environments, caches, or third-party
  source documents without verified redistribution rights.
- Respect the reviewed `.gitignore` publication scope; do not force-add ignored files.
- Keep large model bundles and numerical replay data in GitHub Releases, not
  regular Git. Do not change repository visibility or invent a software license.

## GitHub synchronization

The user wants completed changes reflected in `https://github.com/iion74/UROP`
on branch `main`.

- After implementation tasks, run `python -m pytest -q`, then commit only the
  task's files and push. Do not commit for explanation-only requests.
- Commits are authored by `iion74`; do not add `Co-Authored-By` trailers.
- Do not stage unrelated changes or force-push without the user's explicit request.
- If authentication, network, or remote changes block a push, keep local work
  and report what is unsynchronized. Never claim a push that did not succeed.
- Communicate with the user in Korean.
