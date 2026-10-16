"""Build the local S14 data pack through the production N3B extension path."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from tttmem.p2.data import _set_process_name, pack_store, sha256_file
from tttmem.p2.s14_data import prepare_data, verify_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process-name', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--pack-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    _set_process_name(args.process_name)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    expected = repo/'configs/s14/data/expected-sha256.json'
    base, pack, report = map(Path, (args.base, args.pack_dir, args.report_dir))
    report.mkdir(parents=True, exist_ok=True)
    verify_store(base, expected)
    pack.mkdir(parents=True, exist_ok=True)
    for name in json.loads(expected.read_text())['sha256']:
        dest = pack/name
        if not dest.exists():
            shutil.copy2(base/name, dest)
    verification = prepare_data(dict(data_dir=str(pack), repo_root=str(repo),
        data=dict(mode='uploaded', rebuild_c=True, workers=1, c_rows=25600,
                  expected_hashes=str(expected))), report, need_c=True)
    record = json.loads((pack/'extensions.json').read_text())[-1]
    pinned = dict(schema='s14-C-pinned-v1', counts=record['counts'],
                  sha256={n:sha256_file(pack/n) for n in
                    ('C.bin','C.jsonl','extend-C.meta.json','extensions.json')})
    pin_path = repo/'configs/s14/data/expected-c-sha256.json'
    pin_path.write_text(json.dumps(pinned, indent=2, sort_keys=True)+'\n')
    verification = prepare_data(dict(data_dir=str(pack), repo_root=str(repo),
        data=dict(mode='uploaded', rebuild_c=False, workers=1, c_rows=25600,
                  expected_hashes=str(expected), expected_c_hashes=str(pin_path))), report, need_c=True)
    archive = pack.parent/'p2data.tar'
    digest = pack_store(pack, archive)
    result = dict(process_name=args.process_name, pack_dir=str(pack), archive=str(archive),
        archive_sha256=digest, archive_bytes=archive.stat().st_size,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p))
               for p in sorted(pack.iterdir()) if p.is_file()},
        wall_s=time.monotonic()-started, verification=verification,
        pinned_c=pinned, torch_imported='torch' in __import__('sys').modules)
    (report/'pack-build.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('archive','archive_bytes','archive_sha256','wall_s','torch_imported')}), flush=True)


if __name__ == '__main__':
    main()
