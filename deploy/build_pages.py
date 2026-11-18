"""Build the public, static landing page without copying the repository."""

import argparse
import os
import re
from html import escape
from pathlib import Path
from shutil import copyfile


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "__CERVIAI_APP_URL__"
RUN_APP_ORIGIN = re.compile(r"https://(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app/?")
ASSETS = ("static/style.css", "static/images/ai_doctor.png")


def build(app_url: str, output: Path) -> None:
    url = app_url.strip()
    if not RUN_APP_ORIGIN.fullmatch(url):
        raise ValueError("Set CERVIAI_APP_URL to the HTTPS *.run.app Service URL, without a path or credentials.")
    source = ROOT / "frontend"
    template = (source / "index.html").read_text(encoding="utf-8")
    if TOKEN not in template:
        raise ValueError("The landing page is missing its Cloud Run link placeholder.")
    # Only upload a new, explicit allowlist. Never publish the repository or a reused output folder.
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.html").write_text(template.replace(TOKEN, escape(url.rstrip("/"))), encoding="utf-8")
    for relative in ASSETS:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source / relative, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        build(os.environ.get("CERVIAI_APP_URL", ""), args.output)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"Pages build failed: {error}\n")
