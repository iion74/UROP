#!/usr/bin/env bash
# Commit explicitly selected task files; never force-push or auto-merge.
set -euo pipefail

if (( $# < 2 )); then
    echo 'Usage: bash scripts/sync_github.sh "Commit message" file-or-directory ...' >&2
    exit 2
fi

sync_message=$1
shift
sync_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sync_root=$(git -C "$sync_script_dir/.." rev-parse --show-toplevel)
cd -- "$sync_root"

sync_remote=$(git config --get remote.origin.url)
case "$sync_remote" in
    https://github.com/iion74/UROP|https://github.com/iion74/UROP.git|git@github.com:iion74/UROP.git) ;;
    *) echo 'Refusing to publish: origin is not iion74/UROP.' >&2; exit 1 ;;
esac

if [[ $(git branch --show-current) != main ]]; then
    echo 'Refusing to publish: switch to main after reviewing your branch.' >&2
    exit 1
fi

for sync_state in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply; do
    if [[ -e $(git rev-parse --git-path "$sync_state") ]]; then
        echo 'Refusing to publish: complete the pending Git operation first.' >&2
        exit 1
    fi
done

if ! git diff --cached --quiet; then
    echo 'Refusing to publish: staged changes already exist; review them first.' >&2
    exit 1
fi

for sync_path in "$@"; do
    case "$sync_path" in
        ''|/*|.|..|../*|*/../*|*/..|:*|.git|.git/*)
            echo "Refusing unsafe path: $sync_path" >&2; exit 1 ;;
    esac
    if git check-ignore -q -- "$sync_path"; then
        echo "Refusing ignored path: $sync_path" >&2
        exit 1
    fi
done

# Do not open credential dialogs or wait for input during agent execution.
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=/bin/false
export GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes'
git fetch origin main
if ! git merge-base --is-ancestor refs/remotes/origin/main HEAD; then
    echo 'GitHub has changes not in this branch. Review and integrate them before publishing.' >&2
    exit 1
fi

git add --all -- "$@"
git diff --cached --check
if ! git diff --cached --quiet; then
    git diff --cached --stat
    git commit -m "$sync_message"
else
    echo 'No new changes in selected files; checking for unpublished commits.'
fi
git push origin HEAD:main
echo 'GitHub synchronization completed.'
