from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive a training run's checkpoints into Git LFS."
    )
    parser.add_argument(
        "run_dir",
        help="Training run directory containing checkpoints/ and usually latest.pt.",
    )
    parser.add_argument(
        "--archive-name",
        default="",
        help="Destination name under artifacts/checkpoint-archive/. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--include-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also archive latest.pt when present.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Create a git commit after staging the LFS files and manifest.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push HEAD to the target branch after committing.",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    return parser.parse_args()


def checkpoint_cycle(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("cycle-"):
        return None
    try:
        return int(stem.removeprefix("cycle-"))
    except ValueError:
        return None


def copy_if_changed(source: Path, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size == source.stat().st_size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def main() -> None:
    root = repo_root()
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    archive_name = args.archive_name or run_dir.name
    archive_dir = root / "artifacts" / "checkpoint-archive" / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = sorted(
        checkpoint_dir.glob("cycle-*.pt"),
        key=lambda path: (checkpoint_cycle(path) is None, checkpoint_cycle(path) or 0, path.name),
    )
    if not checkpoint_paths:
        raise RuntimeError(f"no cycle checkpoints found in {checkpoint_dir}")

    manifest_entries: list[dict[str, object]] = []
    copied = 0
    for source in checkpoint_paths:
        dest = archive_dir / source.name
        copied += int(copy_if_changed(source, dest))
        manifest_entries.append(
            {
                "kind": "cycle",
                "cycle": checkpoint_cycle(source),
                "path": str(dest.relative_to(root)).replace("\\", "/"),
                "bytes": source.stat().st_size,
            }
        )

    latest = run_dir / "latest.pt"
    if args.include_latest and latest.exists():
        dest = archive_dir / "latest.pt"
        copied += int(copy_if_changed(latest, dest))
        manifest_entries.append(
            {
                "kind": "latest",
                "cycle": None,
                "path": str(dest.relative_to(root)).replace("\\", "/"),
                "bytes": latest.stat().st_size,
            }
        )

    manifest = {
        "archive_name": archive_name,
        "source_run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(checkpoint_paths),
        "total_checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_paths),
        "entries": manifest_entries,
    }
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    run_git(["lfs", "install"], root)
    run_git(["lfs", "track", "artifacts/checkpoint-archive/**/*.pt"], root)
    run_git(["add", ".gitattributes"], root)
    run_git(["add", "-f", str(archive_dir.relative_to(root))], root)

    total_mb = manifest["total_checkpoint_bytes"] / 1024 / 1024
    print(
        f"archived {len(checkpoint_paths)} cycle checkpoints "
        f"({total_mb:.2f} MB) to {archive_dir}"
    )
    print(f"copied_or_updated={copied}")

    if args.commit:
        run_git(["commit", "-m", f"Archive checkpoints for {archive_name}"], root)
    if args.push:
        if not args.commit:
            raise ValueError("--push requires --commit so the archive is actually recorded")
        run_git(["push", args.remote, f"HEAD:{args.branch}"], root)


if __name__ == "__main__":
    main()
