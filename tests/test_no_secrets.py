"""No credential may be committed to this repository.

The repository went public on day 14, and a Gemini API key now exists on the
machine that builds it. Bots scrape GitHub for key-shaped strings within
minutes of a push, so the window between committing a secret and it being
found is not long enough to rely on noticing.

I scanned by hand before flipping the repo public. A check that ran once is a
check that will not run next time, so this is the permanent version, and it
runs in CI as its own named step so a failure is unmissable.

**Scope, stated honestly.** This scans the files git is currently tracking.
It does not scan history: a secret that was committed and later removed is
still in the objects and still compromised, and the only real remedy there is
to revoke the credential. It also cannot catch a key that does not match a
known shape. It is a guard against the ordinary accident — pasting a key into
a file and committing it — not a proof that nothing has ever leaked.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Patterns are written to match a *realistic* credential: the right prefix and
# the right length. A looser pattern finds more and is switched off within a
# week, which protects nothing.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("Google API key", r"AIza[0-9A-Za-z_\-]{35}"),
    ("OpenAI key", r"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    ("Anthropic key", r"sk-ant-[A-Za-z0-9_\-]{40,}"),
    ("GitHub token", r"gh[pousr]_[A-Za-z0-9]{36}"),
    ("GitHub fine-grained token", r"github_pat_[A-Za-z0-9_]{60,}"),
    ("Hugging Face token", r"hf_[A-Za-z0-9]{34,}"),
    ("AWS access key id", r"AKIA[0-9A-Z]{16}"),
    ("Slack token", r"xox[baprs]-[A-Za-z0-9\-]{20,}"),
    ("private key block", r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    ("Google service account", r'"type"\s*:\s*"service_account"'),
)

# A long opaque string assigned to a secret-sounding name. This is the one
# that would produce false positives, so placeholders are excluded explicitly
# rather than by loosening the pattern.
ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(?:api[_-]?key|secret|password|passwd|token|credential)s?\b
    \s*[:=]\s*
    ["'`]([A-Za-z0-9+/_\-]{24,})["'`]
    """
)

PLACEHOLDERS = re.compile(
    r"(?i)^(?:your|my|put|insert|replace|example|sample|dummy|fake|test|"
    r"changeme|xxx+|placeholder|redacted|none|null|todo|abc|foo|bar)"
)

# Binary and generated files: scanning them is slow and they are not where a
# pasted key ends up.
SKIP_SUFFIXES = frozenset(
    {".npz", ".wav", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".lock", ".onnx"}
)


def tracked_files() -> list[Path]:
    """Everything git is tracking. Ignored files are, by definition, not committed."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return [REPO / name for name in out.stdout.split("\0") if name]


def findings_in(text: str) -> list[str]:
    """Every credential-shaped string in `text`, named by what it looks like."""
    found = []
    for label, pattern in PATTERNS:
        if re.search(pattern, text):
            found.append(label)
    for match in ASSIGNMENT.finditer(text):
        value = match.group(1)
        if not PLACEHOLDERS.match(value):
            found.append(f"long opaque value assigned to a secret-sounding name: {value[:8]}…")
    return found


def scan() -> dict[str, list[str]]:
    """Scan every tracked text file. Returns {path: [what was found]}."""
    problems: dict[str, list[str]] = {}
    for path in tracked_files():
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        # This file necessarily contains the patterns it is looking for.
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        found = findings_in(text)
        if found:
            problems[str(path.relative_to(REPO))] = found
    return problems


class TestTheRepositoryIsClean:
    def test_no_tracked_file_contains_a_credential(self):
        problems = scan()
        assert not problems, "credential-shaped strings found:\n" + "\n".join(
            f"  {path}: {', '.join(found)}" for path, found in problems.items()
        )

    def test_no_env_file_is_tracked(self):
        """`.env` is gitignored, but a `git add -f` would override that."""
        names = {p.name for p in tracked_files()}
        assert not {n for n in names if n == ".env" or n.startswith(".env.")} - {".env.example"}

    def test_the_gitignore_still_covers_env_files(self):
        ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
        assert re.search(r"^\.env\b", ignore, re.MULTILINE), ".env is no longer gitignored"


class TestTheScannerActuallyWorks:
    """A scanner that finds nothing is indistinguishable from a broken one.

    Every fixture is built by concatenation so this file does not itself
    contain a key-shaped literal — otherwise the scanner would have to skip
    its own test file for a second reason, and the exclusion would hide real
    findings if these ever moved.
    """

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("Google API key", "AIza" + "a1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r"),
            ("OpenAI key", "sk-" + "A" * 40),
            ("Anthropic key", "sk-ant-" + "B" * 50),
            ("GitHub token", "ghp_" + "c" * 36),
            ("Hugging Face token", "hf_" + "d" * 36),
            ("AWS access key id", "AKIA" + "E" * 16),
            ("private key block", "-----BEGIN RSA PRIVATE KEY-----"),
        ],
    )
    def test_it_finds_a_real_key_shape(self, label, value):
        assert label in findings_in(f"SOME_KEY = {value!r}\n")

    def test_it_finds_a_key_pasted_into_prose(self):
        text = "set it to " + "AIza" + "z" * 35 + " and you are done"
        assert "Google API key" in findings_in(text)

    def test_it_finds_an_opaque_value_behind_a_secret_name(self):
        secret = "q" * 40
        assert findings_in(f'api_key = "{secret}"')

    @pytest.mark.parametrize(
        "line",
        [
            "GOOGLE_API_KEY=your-key-here",
            'api_key = "your-api-key"',
            'token: "REPLACE_ME_WITH_A_REAL_TOKEN"',
            'password = "changeme"',
            'secret = "example-value-goes-right-here"',
        ],
    )
    def test_it_ignores_obvious_placeholders(self, line):
        assert not findings_in(line), line

    @pytest.mark.parametrize(
        "line",
        [
            "model_id = 'cross-encoder/ms-marco-MiniLM-L-6-v2'",
            "chunk_id = 'cancellation-private-events#structural:0'",
            'conversation_id = "d13-asks-for-human-mid-booking"',
            "trace_id = 'a1b2c3d4e5f6'",
            "# the token budget is 256",
        ],
    )
    def test_it_does_not_fire_on_ordinary_repository_strings(self, line):
        assert not findings_in(line), line

    def test_it_reads_the_files_git_is_tracking(self):
        files = tracked_files()
        assert len(files) > 100
        assert (REPO / "README.md") in files
