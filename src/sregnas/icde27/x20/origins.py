
from pathlib import Path


def resolver():
    cache={}
    def resolve(value):
        path=Path(value)
        if path in cache:return cache[path]

        if not path.is_absolute() or '..' in path.parts or path.parent==path:
            result=path.resolve()
        else:
            parent=resolve(path.parent)
            candidate=parent/path.name
            result=candidate.resolve() if candidate.is_symlink() else candidate
        cache[path]=result
        return result
    return resolve
