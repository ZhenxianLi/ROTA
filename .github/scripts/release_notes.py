"""Print the release notes of one version, taken from CHANGELOG.md.

Usage: python .github/scripts/release_notes.py X.Y.Z > notes.md

Prints the highlights in .github/release-highlights/X.Y.Z.md if that file exists
(figures, a short tour), then the section "## [X.Y.Z] - date" of CHANGELOG.md
(without its heading, up to the next "## [" heading or the link definitions), then
the compare link of X.Y.Z and the install command. Used by
.github/workflows/publish.yml.
"""

import pathlib
import re
import sys


def notes(version, text, highlights=None):
    head = re.search(r"^## \[" + re.escape(version) + r"\] - .*$", text, re.MULTILINE)
    if head is None:
        raise SystemExit(f"CHANGELOG.md has no section '## [{version}] - date'")
    rest = text[head.end():]
    end = re.search(r"^(## \[|\[[^\]]+\]: )", rest, re.MULTILINE)
    body = (rest[: end.start()] if end else rest).strip()
    if not body:
        raise SystemExit(f"the CHANGELOG.md section of {version} is empty")
    lines = []
    if highlights and highlights.strip():
        lines += [highlights.strip(), "", "## Changes", ""]
    lines += [body, ""]
    link = re.search(r"^\[" + re.escape(version) + r"\]: (\S+)", text, re.MULTILINE)
    if link:
        lines.append(f"Full list of changes: {link.group(1)}")
        lines.append("")
    lines.append(f"Install: `pip install rota-order=={version}`")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = pathlib.Path(__file__).resolve().parents[2]
    highlights = root / ".github" / "release-highlights" / f"{sys.argv[1]}.md"
    print(notes(sys.argv[1], (root / "CHANGELOG.md").read_text(encoding="utf-8"),
                highlights.read_text(encoding="utf-8") if highlights.is_file() else None))
