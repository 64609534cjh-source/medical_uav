"""Restore hash-checked research evidence into this checkout's ignored outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {'r1': 'gen4_r1_visibility', 'lookahead': 'gen4_lookahead_feasibility'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(name):
    index = json.loads((ROOT / f'artifacts/{name}_manifest.json').read_text())
    archive = ROOT / 'artifacts' / index['archive']
    if digest(archive.read_bytes()) != index['sha256']:
        raise ValueError('research archive hash mismatch')
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for entry in tar:
            relative = Path(entry.name)
            target = ROOT / relative
            if (not entry.isfile() or relative.is_absolute() or '..' in relative.parts
                    or not target.resolve().is_relative_to(ROOT / 'outputs') or entry.name in seen):
                raise ValueError('unsafe/duplicate research archive member')
            content = tar.extractfile(entry).read()
            expected = index['members'].get(entry.name)
            if expected is None and entry.name != 'outputs/gen4_r1_visibility/manifest.json':
                raise ValueError('unindexed research archive member')
            if expected is not None and digest(content) != expected:
                raise ValueError('research member hash mismatch')
            if target.exists() and digest(target.read_bytes()) != digest(content):
                raise ValueError('refusing to overwrite changed local research evidence')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            seen.add(entry.name)
    if not set(index['members']) <= seen:
        raise ValueError('research archive missing indexed evidence')
    print(name, len(seen), 'verified files restored')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', choices=tuple(BUNDLES)+('all',), default='all')
    args = parser.parse_args()
    for key, name in BUNDLES.items():
        if args.bundle in (key, 'all'):
            restore(name)
