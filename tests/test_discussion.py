"""Artifact extraction must reject unsafe entries and preserve local evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


def setup_release(tmp_path, monkeypatch, name='outputs/record.txt'):
    spec = importlib.util.spec_from_file_location('discussion', Path(__file__).parents[1] / 'scripts/discussion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    archive = artifacts / 'test.tar.gz'
    data = b'original evidence'
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    manifest = {'archives': {'test.tar.gz': module.digest(archive)},
                'files': {name: hashlib.sha256(data).hexdigest()}, 'source_files': {}}
    (artifacts / 'checkpoint_manifest.json').write_text(json.dumps(manifest))
    return module, archive


def test_extraction_idempotent_but_refuses_changed_local_record(tmp_path, monkeypatch):
    module, _ = setup_release(tmp_path, monkeypatch)
    module.unpack()
    module.unpack()
    target = tmp_path / 'outputs/record.txt'
    target.write_text('different local evidence')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        module.unpack()
    assert target.read_text() == 'different local evidence'


@pytest.mark.parametrize('name', ['../escape.txt', '/tmp/escape.txt'])
def test_reject_archive_path_escape(tmp_path, monkeypatch, name):
    module, _ = setup_release(tmp_path, monkeypatch, name)
    with pytest.raises(ValueError, match='unexpected archive entry'):
        module.unpack()


def test_reject_modified_archive(tmp_path, monkeypatch):
    module, archive = setup_release(tmp_path, monkeypatch)
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='archive hash mismatch'):
        module.unpack()
