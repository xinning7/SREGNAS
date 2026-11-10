
import os
from pathlib import Path


def record_lock(path, *, enabled=False):
    fd = os.environ.get('SREGNAS_X16_LOCK_FD') if enabled else None
    if fd is None:
        return Path(path).open('a')
    fd = int(fd)
    expected, actual = os.stat(path), os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise RuntimeError('X16 inherited descriptor is not the global GPU lock')
    return os.fdopen(os.dup(fd), 'a')
