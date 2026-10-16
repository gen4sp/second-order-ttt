import json
import re
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tttmem.p2 import passkey
from tttmem.p2.meta import InnerCfg, stream_eval
from tttmem.p2.mini import _write_tiny_data
from tttmem.p2.model import P2Config, PRESETS, build_model
from tttmem.p2.train import TrainCfg


class TinyTokenizer:
    """Separate space plus five digit tokens, to test independent continuations."""
    def encode(self, text, add_special_tokens=False):
        if text == ' ':
            return type('Encoded', (), {'ids': [53]})()
        words = re.findall(r'\d|[A-Za-z]+|[^\w\s]', text)
        ids = [3+int(w) if w.isdigit() else 13+sum(map(ord, w)) % 40 for w in words]
        if re.search(r' \d{5}$', text):
            ids.insert(-5, 53)
        return type('Encoded', (), {'ids': ids})()


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    torch.set_num_threads(4)
    preset = P2Config(vocab_size=97, hidden_size=16, intermediate_size=24,
                      num_layers=3, num_heads=2, chunk=32, window=256,
                      suffix_len=1, bos_id=1, eos_id=2, max_len=256)
    monkeypatch.setitem(PRESETS, 'passkey-tiny', preset)
    cfg = TrainCfg(preset='passkey-tiny', device='cpu', dtype='fp32',
                   seq_len=256, segment_len=64, barriers=[2, 4, 6], inner_clip=.4)
    model = build_model(preset, compute_dtype=torch.float32, seed=3)
    model.requires_grad_(False)
    return model, cfg, TinyTokenizer()


def test_multitoken_keys_unique_length_matched_and_paired_documents(tiny):
    model, cfg, tok = tiny
    hosts = np.random.default_rng(5).integers(3, 97, (2, 257), dtype=np.int64)
    p = passkey._documents(hosts, tok, cfg, model.cfg, 1014)
    q = passkey._documents(hosts, tok, cfg, model.cfg, 1014)
    for i, (a, b) in enumerate(zip(p, q)):
        assert a['keys'] == b['keys'] and len(set(a['keys'])) == 9
        assert a['key_ids'].shape == (9, 6)
        s, e = a['source_span']
        assert 0 < s < e <= model.cfg.chunk
        for position in ('beyond', 'in_window'):
            np.testing.assert_array_equal(a['docs'][position], b['docs'][position])
        # The positive-control answer is absent in the cross layout's query chunk.
        q0, a0, e0 = a['spans']['in_window']
        np.testing.assert_array_equal(a['docs']['beyond'][q0:e0], hosts[i, q0:e0])
        assert a['spans']['beyond'][0] // cfg.segment_len == 2
        assert a['spans']['in_window'][0] // cfg.segment_len == 0
        assert a['spans']['in_window'][2] == cfg.segment_len
        assert a['spans']['in_window'][0] > model.cfg.chunk+2


@pytest.mark.parametrize('position', ['beyond', 'in_window'])
@pytest.mark.parametrize('adapt', [True, False])
def test_optimized_scores_match_full_stream_eval_for_each_multitoken_foil(tiny, position, adapt):
    model, cfg, tok = tiny
    host = np.random.default_rng(9).integers(3, 97, (1, 257), dtype=np.int64)
    p = passkey._documents(host, tok, cfg, model.cfg, 1014)[0]
    doc, span = p['docs'][position], p['spans'][position]
    inner = InnerCfg(lr=.6, clip=cfg.inner_clip)
    context = passkey._context(model, doc, cfg, inner, position, adapt)
    got = passkey._score(model, doc, p['key_ids'], span, context, cfg)
    _, a, end = span
    expected = []
    for ids in p['key_ids']:
        d = doc.copy(); d[a:end] = ids
        x = torch.from_numpy(d[:-1]).unsqueeze(0)
        y = torch.from_numpy(d[1:]).unsqueeze(0)
        mask = y != model.cfg.bos_id
        im = torch.zeros_like(mask)
        im[:, :model.cfg.chunk*(1 if position == 'in_window' else 4)] = True
        read = torch.zeros_like(mask); read[:, a-1:end-1] = True
        r = stream_eval(model, x, y, mask, inner, adapt=adapt, barriers=cfg.barriers,
                        inner_mask=im, fused_attn=False, nll_positions=read, return_greedy=True)
        expected.append(r['nll'][0, a:end-1].numpy())
        if not len(expected)-1:
            assert got['exact'] == bool(r['greedy'][0, a:end-1].all())
    np.testing.assert_allclose(got['nll_token'], expected, rtol=1e-5, atol=2e-6)
    assert got['lift'] == pytest.approx(np.mean(expected[1:])-np.mean(expected[0]), abs=2e-6)


def test_barrier_blocks_attention_and_query_cannot_write_answer(tiny):
    model, cfg, tok = tiny
    host = np.random.default_rng(12).integers(3, 97, (1, 257), dtype=np.int64)
    p = passkey._documents(host, tok, cfg, model.cfg, 7)[0]
    d = p['docs']['beyond']
    changed = d.copy(); s, e = p['source_span']; changed[s:e] = 55
    inner = InnerCfg(lr=1, clip=cfg.inner_clip)
    score = lambda doc, adapt: passkey._score(model, doc, p['key_ids'], p['spans']['beyond'],
        passkey._context(model, doc, cfg, inner, 'beyond', adapt), cfg)['nll_true']
    assert score(d, False) == score(changed, False)
    assert abs(score(d, True)-score(changed, True)) > 1e-7


def test_public_api_files_summaries_and_noadapt(tiny, tmp_path, monkeypatch):
    model, cfg, tok = tiny
    data, out = tmp_path/'data', tmp_path/'reads'
    _write_tiny_data(data, rows=4, seed=8)
    np.random.default_rng(10).integers(3, 97, (4, 8193), dtype=np.uint32).tofile(data/'ho.bin')
    (data/'ho.jsonl').write_text(''.join(json.dumps({'row': i, 'sha1': f'ho-{i}'})+'\n' for i in range(4)))
    meta = json.loads((data/'meta.json').read_text()); meta['quotas']['ho'] = 4
    (data/'meta.json').write_text(json.dumps(meta))
    (data/'B.bin').write_bytes((data/'A.bin').read_bytes())
    (data/'B.jsonl').write_bytes((data/'A.jsonl').read_bytes())
    hosts = tmp_path/'hosts.json'
    passkey.freeze_hosts(data, hosts)
    monkeypatch.setattr(passkey, 'HOSTS_FILE', hosts)
    monkeypatch.setattr(passkey, '_tokenizer', lambda *args: tok)
    config = tmp_path/'cell.json'; config.write_text(json.dumps(asdict(cfg)))
    ckpt = tmp_path/'model.pt'; torch.save({'model': model.state_dict()}, ckpt)
    r = passkey.passkey_read(str(ckpt), str(config), [1, 2], str(out), n_docs=4, device='cpu', data_dir=str(data))
    assert set(r['conditions']) == {'ce', 'eta200', 'noadapt'}
    assert r['torch_threads'] == 4 and r['dtype'] == 'fp32'
    assert r['n_docs'] == 4 and r['leak_check'] == 'PASS' and r['leak_check_n_docs'] == 4
    assert r['max_abs_delta_nll'] <= 1e-5
    for condition in r['conditions']:
        with np.load(out/f'passkey_{condition}.npz', allow_pickle=False) as z:
            summary = json.loads(str(z['summary']))
            assert str(z['leak_check']) == 'PASS'
            for pos in ('in_window', 'beyond'):
                assert z[pos+'_nll_foils'].shape == (4, 8)
                assert z[pos+'_nll_token'].shape == (4, 9, 5)
                assert summary[pos]['lift'] == pytest.approx(z[pos+'_lift'].mean())
                assert summary[pos]['top1'] == np.mean(z[pos+'_rank'] == 1)
    with pytest.raises(ValueError, match='positive'):
        passkey.passkey_read('', '', [-1], str(out))
    with pytest.raises(ValueError, match='collide'):
        passkey.passkey_read('', '', [1, 1], str(out))
    with pytest.raises(ValueError, match='batch'):
        passkey.passkey_read('', '', [1], str(out), batch=0)
    # Exercise the public API's real batching, all candidate digit NLLs and
    # both adapted/noadapt conditions; metadata/timings need not be equal.
    r4 = passkey.passkey_read(str(ckpt), str(config), [1, 2], str(tmp_path/'batch4'),
        n_docs=4, device='cpu', data_dir=str(data), batch=4)
    assert r4['batch'] == 4
    for condition in r['conditions']:
        with np.load(out/f'passkey_{condition}.npz') as one, np.load(tmp_path/'batch4'/f'passkey_{condition}.npz') as four:
            for pos in ('in_window', 'beyond'):
                for field in ('nll_token', 'nll_true', 'nll_foils', 'lift', 'rank', 'exact'):
                    assert np.max(np.abs(one[pos+'_'+field].astype(float)-four[pos+'_'+field].astype(float))) <= 1e-5


@pytest.mark.parametrize('position', ['in_window', 'beyond'])
def test_changing_query_and_answer_cannot_change_adapted_fast_weights(tiny, position):
    model, cfg, tok = tiny
    host = np.random.default_rng(12).integers(3, 97, (1, 257), dtype=np.int64)
    p = passkey._documents(host, tok, cfg, model.cfg, 7)[0]
    doc = p['docs'][position]
    changed = doc.copy()
    q, a, end = p['spans'][position]
    changed[q:end] = np.random.default_rng(44).integers(3, 97, end-q)
    inner = InnerCfg(lr=1, clip=cfg.inner_clip)
    before = passkey._context(model, doc, cfg, inner, position, True)[0]
    after = passkey._context(model, changed, cfg, inner, position, True)[0]
    for w, v in zip(before, after):
        torch.testing.assert_close(w, v, rtol=0, atol=0)


def test_leak_check_detects_source_information_and_checks_each_token(tiny, monkeypatch):
    model, cfg, tok = tiny
    hosts = np.random.default_rng(5).integers(3, 97, (10, 257), dtype=np.int64)
    plans = passkey._documents(hosts, tok, cfg, model.cfg, 1014)
    got = passkey._leak_check(model, plans, hosts, cfg)
    assert got['leak_check'] == 'PASS' and got['leak_check_n_docs'] == 8
    original = passkey._score
    def leaking(model, doc, ids, span, context, cfg):
        r = original(model, doc, ids, span, context, cfg)
        # Inject a source-dependent effect into just one foil digit token.
        r['nll_token'][3, 2] += doc[plans[0]['source_span'][0]]*.001
        return r
    monkeypatch.setattr(passkey, '_score', leaking)
    assert passkey._leak_check(model, plans, hosts, cfg)['leak_check'] == 'FAIL'


def test_frozen_hosts_ignore_C_and_fail_closed_on_ids_tokens_manifests(tmp_path, monkeypatch):
    _write_tiny_data(tmp_path, rows=2)
    (tmp_path/'B.bin').write_bytes((tmp_path/'A.bin').read_bytes())
    (tmp_path/'B.jsonl').write_bytes((tmp_path/'A.jsonl').read_bytes())
    rows = np.memmap(tmp_path/'ho.bin', mode='r+', dtype=np.uint32, shape=(2, 8193))
    train = np.memmap(tmp_path/'A.bin', mode='r', dtype=np.uint32, shape=(2, 8193))
    rows[0, :256] = train[0, :256]; rows.flush()
    hosts = tmp_path/'hosts.json'
    frozen = passkey.freeze_hosts(tmp_path, hosts)
    monkeypatch.setattr(passkey, 'HOSTS_FILE', hosts)
    _, selected, split = passkey._host_rows(tmp_path, 1, 1014)
    assert split == 'ho' and selected.tolist() == [1]
    # Presence of C, even with a matching prefix, cannot change the frozen set.
    (tmp_path/'C.bin').write_bytes((tmp_path/'ho.bin').read_bytes())
    (tmp_path/'C.jsonl').write_bytes((tmp_path/'ho.jsonl').read_bytes())
    np.testing.assert_array_equal(passkey._host_rows(tmp_path, 1, 1014)[1], selected)
    with pytest.raises(ValueError, match='only 1'):
        passkey._host_rows(tmp_path, 2, 1014)
    with pytest.raises(ValueError, match='ids/seed'):
        passkey._host_rows(tmp_path, 1, 4)
    frozen['host_rows'] = [0]
    hosts.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='documents_sha256'):
        passkey._host_rows(tmp_path, 1, 1014)
    frozen['host_rows'] = [1]
    hosts.write_text(json.dumps(frozen))
    rows[1, 1000] += 1; rows.flush()
    with pytest.raises(ValueError, match='documents_sha256'):
        passkey._host_rows(tmp_path, 1, 1014)
