"""Print the release notes of one version, taken from CHANGELOG.md.

Usage: python .github/scripts/release_notes.py X.Y.Z > notes.md

Prints the section "## [X.Y.Z] - date" of CHANGELOG.md (without its heading, up to
the next "## [" heading or the link definitions), then the compare link of X.Y.Z
and the install command. Used by .github/workflows/publish.yml.
"""

import pathlib
import re
import sys


def notes(version, text):
    head = re.search(r"^## \[" + re.escape(version) + r"\] - .*$", text, re.MULTILINE)
    if head is None:
        raise SystemExit(f"CHANGELOG.md has no section '## [{version}] - date'")
    rest = text[head.end():]
    end = re.search(r"^(## \[|\[[^\]]+\]: )", rest, re.MULTILINE)
    body = (rest[: end.start()] if end else rest).strip()
    if not body:
        raise SystemExit(f"the CHANGELOG.md section of {version} is empty")
    lines = [body, ""]
    link = re.search(r"^\[" + re.escape(version) + r"\]: (\S+)", text, re.MULTILINE)
    if link:
        lines.append(f"Full list of changes: {link.group(1)}")
        lines.append("")
    lines.append(f"Install: `pip install rota-order=={version}`")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    changelog = pathlib.Path(__file__).resolve().parents[2] / "CHANGELOG.md"
    print(notes(sys.argv[1], changelog.read_text(encoding="utf-8")))
