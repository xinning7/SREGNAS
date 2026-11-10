

from functools import lru_cache
import hashlib
import os
import subprocess

from sregnas.paths import PACKAGE_ROOT, SOURCE_ROOT

ROOT = SOURCE_ROOT


def implementation_sha256():
    pass
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".tsv", ".yaml", ".pth"}:
            digest.update(str(path.relative_to(PACKAGE_ROOT)).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _checkout_commit():

    if ROOT is None or not (ROOT / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


@lru_cache(maxsize=1)
def commit_hash():
    return os.environ.get("SREGNAS_RUN_COMMIT") or _checkout_commit()


def freeze_run_commit():
    head = _checkout_commit()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        text=True,
    ) if head is not None else ""
    if dirty.strip():
        from sregnas.icde27.x_authorization import unchanged_preexisting_audit
        if not unchanged_preexisting_audit(ROOT, dirty):
            raise AssertionError("commit all source/config changes before a GPU launch")
    declared = os.environ.get("SREGNAS_RUN_COMMIT")
    if declared and declared != head:
        raise AssertionError("requested run commit differs from checkout HEAD")
    if head is not None:
        os.environ["SREGNAS_RUN_COMMIT"] = head
    else:
        os.environ.pop("SREGNAS_RUN_COMMIT", None)
    commit_hash.cache_clear()
    return head
