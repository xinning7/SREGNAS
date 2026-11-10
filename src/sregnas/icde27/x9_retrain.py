
import functools
import json
from pathlib import Path
import sys


def install(module):
    counts = {name: 0 for name in ('enabled', 'attach_degrees', 'forward')}
    for name in counts:
        original = getattr(module, name)
        def wrap(fn, key):
            @functools.wraps(fn)
            def counted(*args, **kwargs):
                counts[key] += 1
                return fn(*args, **kwargs)
            return counted
        setattr(module, name, wrap(original, name))
    return counts


def main():
    from sregnas.models import normalization_fix
    counts = install(normalization_fix)
    from sregnas.comparison.retrain import main as production_main
    job_path = Path(sys.argv[sys.argv.index('--job') + 1])
    job = json.loads(job_path.read_text())
    try:
        production_main()
    finally:
        Path(job['output'] + '.nf_calls.json').write_text(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
