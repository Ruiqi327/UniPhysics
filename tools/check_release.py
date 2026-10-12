"""Validate bundled dataset hashes and scan distributable files without printing secrets."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'build', 'dist', 'results'}
PATTERNS = {
    'private key': re.compile(rb'-----BEGIN [A-Z ]*PRIVATE' + rb' KEY-----'),
    'API token': re.compile(rb'\bs' + rb'k-[A-Za-z0-9_-]{16,}'),
    'GitHub token': re.compile(rb'\bgh' + rb'[pousr]_[A-Za-z0-9]{20,}'),
    'AWS access key': re.compile(rb'\bAK' + rb'IA[A-Z0-9]{16}\b'),
    'personal Unix path': re.compile(rb'/' + rb'home/[^\s/]+/'),
    'personal Windows path': re.compile(rb'[A-Za-z]:[\\/]' + rb'Users[\\/][^\s\\/]+'),
    'credential in URL': re.compile(rb'https?://' + rb'[^\s/@:]+:[^\s/@]+@'),
}


def main():
    errors = []
    manifest = json.loads((ROOT / 'Eval/benchmark/manifest.json').read_text(encoding='utf-8'))
    for name, details in manifest['datasets'].items():
        path = ROOT / 'Eval/benchmark' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != details['sha256']:
            errors.append(f'{name}: checksum mismatch')
        with path.open(encoding='utf-8') as source:
            records = [json.loads(line) for line in source if line.strip()]
        if len(records) != details['records'] or len({r['id'] for r in records}) != len(records):
            errors.append(f'{name}: invalid record count or duplicate IDs')
    checked = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in IGNORED or part.endswith('.egg-info') for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix in {'.png', '.jpg'}:
            continue
        relative = str(path.relative_to(ROOT))
        if path.name.startswith('.env') and path.name != '.env.example':
            errors.append(f'{relative}: local environment file')
        if path.suffix in {'.pem', '.key', '.p12', '.pfx'} or path.name.startswith('id_rsa'):
            errors.append(f'{relative}: credential file')
        data = path.read_bytes()
        checked += 1
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                errors.append(f'{relative}: possible {label}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'PASS: dataset checksums/counts and {checked} distributable text files scanned.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
