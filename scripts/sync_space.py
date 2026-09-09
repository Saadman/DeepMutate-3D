#!/usr/bin/env python
"""Push the repository to the Hugging Face Space.

The Space needs a YAML front-matter block at the top of README.md to configure
itself (sdk, sdk_version, app_file, preload_from_hub). GitHub has no use for it
and renders it as a metadata table above the actual README, so the block lives
in space_header.yaml and is prepended only for the Space.

    python scripts/sync_space.py            # sync every tracked file
    python scripts/sync_space.py app.py     # sync just these
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitOperationAdd, CommitOperationDelete

REPO = "ras1992/DeepMutate-3D"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return out.stdout.split()


def main(argv: list[str]) -> int:
    files = argv[1:] or tracked_files()
    header = (ROOT / "space_header.yaml").read_text()

    ops, tmp = [], None
    for name in files:
        if name == "space_header.yaml":
            continue                      # Space-only config, not content
        if name == "README.md":
            tmp = pathlib.Path(tempfile.mkdtemp()) / "README.md"
            tmp.write_text(header + "\n" + (ROOT / "README.md").read_text())
            ops.append(CommitOperationAdd("README.md", str(tmp)))
        else:
            ops.append(CommitOperationAdd(name, str(ROOT / name)))

    api = HfApi()
    # Anything on the Space that is no longer tracked here is removed, so the
    # two stay in genuine sync rather than the Space accumulating stale files.
    if not argv[1:]:
        keep = set(files) | {".gitattributes", "space_header.yaml"}
        stale = sorted(set(api.list_repo_files(REPO, repo_type="space")) - keep)
        ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
        if stale:
            print(f"removing {len(stale)} stale file(s) from the Space")

    info = api.create_commit(
        repo_id=REPO, repo_type="space", operations=ops,
        commit_message=f"Sync {len(ops)} change(s) from the repository")
    print(f"synced {len(ops)} operations -> {REPO}")
    print("commit:", getattr(info, "oid", info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
