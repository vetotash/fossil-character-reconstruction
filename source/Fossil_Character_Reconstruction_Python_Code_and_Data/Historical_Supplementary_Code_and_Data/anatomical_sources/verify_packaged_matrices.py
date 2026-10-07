#!/usr/bin/env python3
"""Verify the ten packaged primary matrices; do not claim article verification."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.package_root.resolve()
    register = json.loads((root / 'anatomical_sources/PACKAGED_MATRICES.json').read_text())
    assert len(register['matrices']) == 10
    for row in register['matrices']:
        path = root / row['packaged_path']
        assert path.stat().st_size == row['bytes'], row['source_id']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['source_id']
    print(json.dumps({'primary_matrix_hashes': 'PASS', 'verified_matrix_files': 10,
                      'full_text_article_verification': 'NOT_PERFORMED_BY_THIS_COMMAND',
                      'independent_anatomical_validation': 'NOT_PERFORMED'}, indent=2))


if __name__ == '__main__':
    main()
