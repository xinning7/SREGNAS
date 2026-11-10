
import argparse
import hashlib
import json
from pathlib import Path
from artifact import ROOT, read


def renamed(data):
    return data.replace(b"SHARDGNAS", b"SREGNAS").replace(b"ShardGNAS", b"SREGNAS").replace(b"shardgnas", b"sregnas")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--install", action="store_true", help="copy locally after obtaining the necessary permission")
    args = p.parse_args()
    files = read(ROOT / "provenance/external_files.json")["files"]
    verified = []
    for row in files:
        path = args.bundle.resolve() / row["path"]
        if not path.is_file() or path.is_symlink():
            p.error(f'missing regular file: {row["path"]}')
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            p.error(f'bundle version mismatch: {row["path"]}')
        dest = ROOT / row["target"]
        if args.install and dest.exists():
            p.error(f'destination already exists: {row["target"]}')
        verified.append((row, data))
    installed = []
    if args.install:
        for row, data in verified:
            dest = ROOT / row["target"]
            if dest.suffix in {".py", ".yaml", ".yml", ".json"}:
                data = renamed(data)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            installed.append(dict(path=row["target"], sha256=hashlib.sha256(data).hexdigest()))
        target = ROOT / "outputs/external_installation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(installed, indent=2) + "\n")
    print(json.dumps(dict(verified_files=len(files), installed=bool(args.install))))


if __name__ == "__main__":
    main()
