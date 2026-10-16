import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tttmem.p2.data import (BOS, CODE_EXCLUDED_LANGS, SEQ, iter_codeparrot,
                            prepare, select_code_eval)
from tttmem.p2.train import mix_order


def _parquet(path, rows):
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), path)


def test_codeparrot_is_deterministic_and_filters_language(tmp_path):
    rows = [
        {"code": "z" * 20, "repo_name": "r2", "path": "z.py", "language": "Python"},
        {"code": "a" * 20, "repo_name": "r1", "path": "a.html", "language": "HTML"},
        {"code": "b" * 20, "repo_name": "r1", "path": "b.py", "language": "Python"},
    ]
    path = tmp_path / "train-00000.parquet"
    _parquet(path, rows)
    got = list(iter_codeparrot("rev", [str(path)], CODE_EXCLUDED_LANGS))
    assert [x["language"] for x in got] == ["Python", "Python"]
    got_sha = [hashlib.sha1(r["code"].encode()).hexdigest() for r in rows if r["language"] == "Python"]
    assert [x["repo_name"] for x in got] == [r["repo_name"] for _, r in sorted(zip(got_sha, [rows[0], rows[2]]))]
    assert got[0]["url"] == "github://r2/z.py"
    sha = hashlib.sha1(rows[2]["code"].encode()).hexdigest()
    got2 = list(iter_codeparrot("rev", [str(path)], CODE_EXCLUDED_LANGS,
                                exclude_sha1=(sha,), exclude_repos=("r2",)))
    assert len(got2) == 0


def test_prepare_preserves_code_manifest_fields_and_quantiles(tmp_path):
    ids = list(range(3, 3 + SEQ - 1))
    doc = {"text": "x" * 12000, "_ids": ids, "url": "github://r/p.py",
           "shard": "00000", "row": 7, "language": "Python", "repo_name": "r", "path": "p.py"}
    out = prepare([doc], lambda _: [ids], tmp_path, {"code_eval_pool": 1}, min_chars=12000, log_every=0)
    assert out["counts"] == {"code_eval_pool": 1}
    rec = json.loads((tmp_path / "code_eval_pool.jsonl").read_text())
    assert rec["language"] == "Python" and rec["repo_name"] == "r" and rec["path"] == "p.py"
    assert "chars_per_token_quantiles" in out["acceptance"]


def _probe_row(offset):
    row = np.arange(3 + offset, 3 + offset + (SEQ - 1), dtype=np.uint32)
    row[:1] = 1
    for i in range(5):
        key = [11 + 4 * i, 12 + 4 * i, 13 + 4 * i, 14 + 4 * i]
        row[3 + 6 * i:7 + 6 * i] = key
        row[35 + 6 * i:39 + 6 * i] = key
    return np.r_[np.array([BOS], dtype=np.uint32), row]


def test_code_eval_selection_is_reproducible(tmp_path):
    rows = np.stack([_probe_row(i * 1000) for i in range(3)])
    rows.tofile(tmp_path / "code_eval_pool.bin")
    with (tmp_path / "code_eval_pool.jsonl").open("w") as f:
        for i in range(3):
            f.write(json.dumps({"row": i, "sha1": str(i), "language": "Python",
                                "repo_name": f"r{i}", "path": f"{i}.py",
                                "chars_per_token": 2.0}) + "\n")
    a = select_code_eval(tmp_path, quota=2, chunk=8, barriers=(2, 4, 6))
    first = (tmp_path / "code_eval.jsonl").read_bytes()
    b = select_code_eval(tmp_path, quota=2, chunk=8, barriers=(2, 4, 6))
    assert a == b and first == (tmp_path / "code_eval.jsonl").read_bytes()
    assert a["eval_count"] == 2 and a["eligible_count"] >= 2


def test_mix_order_exactly_eight_plus_eight_and_seeded():
    order = mix_order(128, 128, steps=3, batch=16, data_seed=41)
    assert order.shape == (48, 2)
    for i in range(3):
        batch = order[i * 16:(i + 1) * 16]
        assert np.all(batch[:8, 0] == 0) and np.all(batch[8:, 0] == 1)
    assert not np.array_equal(order, mix_order(128, 128, steps=3, batch=16, data_seed=42))
