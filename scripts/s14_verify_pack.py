"""Independent full CPU verification of the local uploaded S14 pack and archive."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tarfile
import time

from tttmem.p2.data import _set_process_name, sha256_file
from tttmem.p2.s14_data import prepare_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--pack-dir',required=True)
    p.add_argument('--report-dir',required=True);p.add_argument('--process-name',required=True)
    a=p.parse_args();_set_process_name(a.process_name)
    started=time.monotonic();pack=Path(a.pack_dir);report=Path(a.report_dir)
    repo=Path(__file__).resolve().parents[1]
    verification=prepare_data(dict(data_dir=str(pack),repo_root=str(repo),data=dict(
        mode='uploaded',rebuild_c=False,c_rows=25600,workers=1,audit_shingles=True,
        expected_hashes='configs/s14/data/expected-sha256.json',
        expected_c_hashes='configs/s14/data/expected-c-sha256.json')),report,need_c=True)
    archive=pack.parent/'p2data.tar';digest=sha256_file(archive)
    assert digest==archive.with_name(archive.name+'.sha256').read_text().split()[0]
    hashes={p.name:sha256_file(p) for p in pack.iterdir() if p.is_file()}
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert {m.name for m in members}=={'p2data/'+n for n in hashes}
        for member in members:
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for b in iter(lambda:f.read(8<<20),b''):h.update(b)
            assert h.hexdigest()==hashes[Path(member.name).name]
    record=json.loads((pack/'extensions.json').read_text())[-1]
    assert record['counts']=={'C':25600} and record['existing_unchanged']
    assert record['workers']==1 and record['used_shards_are_seeded_prefix']
    result=dict(status='OK',verification=verification,archive_bytes=archive.stat().st_size,
        archive_sha256=digest,archive_members_verified=len(hashes),
        C_record_provenance='OK',base_hashes_verified=13,C_rows=25600,
        wall_s=time.monotonic()-started,torch_imported='torch' in sys.modules,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    (report/'pack-full-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','wall_s','base_hashes_verified','C_rows',
                                         'archive_sha256','torch_imported','peak_rss_bytes')}),flush=True)


if __name__=='__main__':main()
