"""Download the DeepLure Drive corpus into data/.

The corpus is proprietary. It is gitignored and must not be redistributed.
Delete data/ once the exercise is over.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_FOLDER = "1V_DcXJ50QYV9nEqvq7fqKno_lhbSPdBM"
USER_AGENT = "Mozilla/5.0"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def list_folder(folder_id: str) -> list[tuple[str, str, bool]]:
    """Return (name, id, is_folder) for a publicly shared Drive folder."""
    html = _get(f"https://drive.google.com/embeddedfolderview?id={folder_id}").decode("utf-8", "replace")
    entries = re.findall(
        r'id="entry-([^"]+)"[\s\S]*?class="flip-entry-title"[^>]*>([^<]+)',
        html,
    )
    items = []
    for file_id, name in entries:
        entry_html = html.split(f'id="entry-{file_id}"', 1)[1].split('class="flip-entry"', 1)[0]
        is_folder = f"/folders/{file_id}" in entry_html
        items.append((name.strip(), file_id, is_folder))
    return items


def _download_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    data = _get(f"https://drive.google.com/uc?export=download&id={file_id}")
    if not data.startswith(b"\xff\xd8") and not data.startswith(b"\x89PNG"):
        match = re.search(br"confirm=([0-9A-Za-z_\-]+)", data)
        token = match.group(1).decode() if match else "t"
        data = _get(
            "https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm={token}"
        )
    if not data.startswith(b"\xff\xd8") and not data.startswith(b"\x89PNG"):
        raise RuntimeError(f"Drive did not return an image for {file_id} ({dest.name})")
    dest.write_bytes(data)


def download_corpus(dest_root: Path, workers: int = 6) -> int:
    dest_root.mkdir(parents=True, exist_ok=True)
    top = list_folder(ROOT_FOLDER)
    jobs: list[tuple[str, Path]] = []
    for name, folder_id, is_folder in top:
        if not is_folder:
            jobs.append((folder_id, dest_root / name))
            continue
        children = list_folder(folder_id)
        print(f"{name}: {len(children)} entries")
        for child_name, child_id, child_is_folder in children:
            if child_is_folder:
                continue
            jobs.append((child_id, dest_root / name / child_name))
    print(f"downloading {len(jobs)} files into {dest_root}")
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_file, fid, path): path for fid, path in jobs}
        done = 0
        for future in as_completed(futures):
            done += 1
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed += 1
                print(f"FAIL {path.name}: {exc}")
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}")
    if failed:
        print(f"{failed} downloads failed")
    return len(jobs) - failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/sarees"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    count = download_corpus(args.dest, args.workers)
    print(f"ready: {count} images")


if __name__ == "__main__":
    main()
