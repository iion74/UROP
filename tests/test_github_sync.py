"""Exercise synchronization against temporary local repositories, without GitHub."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REMOTE_URL = "https://github.com/iion74/UROP.git"


class GitHubSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="urop-sync-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote.git"
        self.work = self.root / "work"
        self.work.mkdir()
        self.git("init", "--bare", "--initial-branch=main", str(self.remote))
        self.git("init", "--initial-branch=main")
        self.configure_identity(self.work)
        self.git("config", f"url.{self.remote}.insteadOf", REMOTE_URL)
        self.git("remote", "add", "origin", REMOTE_URL)
        (self.work / "scripts").mkdir()
        (self.work / "src").mkdir()
        shutil.copyfile(PROJECT / "scripts/sync_github.sh", self.work / "scripts/sync_github.sh")
        shutil.copyfile(PROJECT / ".gitignore", self.work / ".gitignore")
        (self.work / "README.md").write_text("Initial project\n")
        (self.work / "src/unrelated.py").write_text("VALUE = 1\n")
        self.git("add", ".gitignore", "README.md", "scripts", "src")
        self.git("commit", "-m", "Initial commit")
        self.git("push", "--set-upstream", "origin", "main")

    def git(self, *arguments, cwd=None):
        return subprocess.run(
            ["git", *arguments], cwd=cwd or self.work, check=True,
            text=True, capture_output=True,
        ).stdout.strip()

    def configure_identity(self, cwd):
        self.git("config", "user.name", "Sync Test", cwd=cwd)
        self.git("config", "user.email", "sync-test@example.invalid", cwd=cwd)

    def sync(self, *paths):
        return subprocess.run(
            ["bash", "scripts/sync_github.sh", "Task update", *paths],
            cwd=self.work, text=True, capture_output=True,
        )

    def test_selected_files_only_are_committed_and_pushed(self):
        (self.work / "src/task.py").write_text("VALUE = 2\n")
        (self.work / "src/unrelated.py").write_text("VALUE = 99\n")
        (self.work / ".env").write_text("LOCAL_ONLY=example\n")
        result = self.sync("src/task.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        remote_head = self.git("--git-dir", str(self.remote), "rev-parse", "main")
        self.assertEqual(remote_head, self.git("rev-parse", "HEAD"))
        self.assertIn("src/unrelated.py", self.git("status", "--short"))
        names = self.git("--git-dir", str(self.remote), "ls-tree", "-r", "--name-only", "main").splitlines()
        self.assertIn("src/task.py", names)
        self.assertNotIn(".env", names)

    def test_existing_staged_changes_are_not_consumed(self):
        (self.work / "src/unrelated.py").write_text("VALUE = 99\n")
        self.git("add", "src/unrelated.py")
        before = self.git("rev-parse", "HEAD")
        result = self.sync("README.md")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("staged changes already exist", result.stderr)
        self.assertEqual(before, self.git("rev-parse", "HEAD"))

    def test_ignored_credentials_are_rejected(self):
        (self.work / ".env").write_text("LOCAL_ONLY=example\n")
        result = self.sync(".env")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing ignored path", result.stderr)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_wrong_remote_is_rejected(self):
        self.git("remote", "set-url", "origin", "https://github.com/example/other.git")
        result = self.sync("README.md")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("origin is not iion74/UROP", result.stderr)

    def test_remote_changes_are_not_overwritten(self):
        other = self.root / "other"
        self.git("clone", str(self.remote), str(other))
        self.configure_identity(other)
        (other / "README.md").write_text("Someone else's update\n")
        self.git("add", "README.md", cwd=other)
        self.git("commit", "-m", "Remote update", cwd=other)
        self.git("push", cwd=other)
        before = self.git("rev-parse", "HEAD")
        result = self.sync("README.md")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GitHub has changes", result.stderr)
        self.assertEqual(before, self.git("rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main()
