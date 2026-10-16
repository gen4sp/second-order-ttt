"""Held-out five-digit passkey reads through the MINI/N3B fast-weight path.

The two positions use separate, paired documents: a positive-control answer
never becomes a write before the cross-barrier read. Only source/filler chunks
write; the query chunk and subsequent chunks are frozen. Candidate continuations
are scored independently, with their own teacher-forced prefix (digits are NOT
assumed to be single tokens). Digit-exact greedy match is conditional on the
common space prefix: the conjunction of correct digit predictions along the
true sequence, equivalent to free digit generation when query writes are frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import tomllib

import numpy as np
import torch
import torch.nn.functional as F

from .data import open_split
from .meta import InnerCfg, _as_past, _fast_values, stream_eval
from .mini import _load_checkpoint, _sha256
from .train import TrainCfg
from .s14_protocol import PASSKEY_ETAS, PASSKEY_PROTOCOL

QUERY = 'What is the pass key? The pass key is'
K = 8
HOSTS_FILE = Path(__file__).with_name('s14_passkey_hosts.json')


def _config(path, device, data_dir):
    path = Path(path)
    raw = (tomllib.loads(path.read_text()) if path.suffix == '.toml' else json.loads(path.read_text()))
    raw = raw.get('p2', raw)
    extra = {k: v for k, v in raw.items() if k.startswith('passkey_')}
    raw = {k: v for k, v in raw.items() if not k.startswith('passkey_')}
    # The trained cell's config stays authoritative for geometry/inner rule. ``init``
    # is cleared because the read loads the trained checkpoint, not the cell's init.
    raw.update(device=device, data_dir=data_dir, init='', resume=False)
    if device == 'cpu':
        raw.update(dtype='fp32', fused_attn=False)
    return TrainCfg.from_dict(raw), extra


def _tokenizer(data_dir, extra):
    from tokenizers import Tokenizer
    if 'passkey_tokenizer_path' in extra:
        return Tokenizer.from_file(extra['passkey_tokenizer_path'])
    from .data import load_tokenizer, TOKENIZER_REPO, TOKENIZER_REVISION
    meta = Path(data_dir) / 'meta.json'
    info = json.loads(meta.read_text()).get('tokenizer', {}) if meta.exists() else {}
    repo = info.get('repo', TOKENIZER_REPO)
    revision = info.get('commit') or info.get('revision', TOKENIZER_REVISION)
    return load_tokenizer(repo, revision)


def _encode(tok, text):
    return np.asarray(tok.encode(text, add_special_tokens=False).ids, dtype=np.int64)


def candidates(tok, rng):
    """Distinct uniform five-digit integers, conditioned on completion length."""
    key = str(int(rng.integers(10000, 100000)))
    prefix = _encode(tok, QUERY)
    def completion(k):
        full = _encode(tok, QUERY+' '+k)
        if not np.array_equal(full[:len(prefix)], prefix):
            raise ValueError('tokenizer changes query prefix at completion boundary')
        return full[len(prefix):]
    target = completion(key)
    space = _encode(tok, ' ')
    if len(space) != 1 or len(target) < 2 or target[0] != space[0]:
        raise ValueError('passkey requires a separate leading space token before digit tokens')
    keys, ids = [key], [target]
    for _ in range(100000):
        f = str(int(rng.integers(10000, 100000)))
        if f in keys:
            continue
        fi = completion(f)
        if len(fi) == len(target) and fi[0] == space[0]:
            keys.append(f); ids.append(fi)
        if len(keys) == K+1:
            return keys, np.stack(ids), prefix
    raise ValueError('could not draw eight same-length foil keys')


def _host_digest(ids, rows):
    h = hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes())
    h.update(np.asarray(rows, dtype='<u4').tobytes())
    return h.hexdigest()


def freeze_hosts(data_dir, path, seed=1014):
    """One-time registration tool, never called by a read. Excludes A+B only.

    C is checked independently against ho/eval/calib by training preflight.
    Hash includes ordered int64 ids followed by uint32 host tokens, little endian.
    """
    root = Path(data_dir)
    split = 'ho'
    rows, manifest = open_split(root, split)
    train_sha = set()
    train_paths = ['A', 'B']
    for s in train_paths:
        train_sha.update(json.loads(line)['sha1'] for line in (root/f'{s}.jsonl').open())
    valid = [i for i, m in enumerate(manifest) if not m.get('near_dup') and m.get('sha1') not in train_sha]
    # Compare candidate held-out prefixes only; bound memory independently of
    # the size of the training store.
    digest = lambda row: hashlib.sha1(np.asarray(row[:256], np.uint32).tobytes()).digest()
    prefixes = {}
    for i in valid:
        prefixes.setdefault(digest(rows[i]), set()).add(i)
    bad = set()
    for s in train_paths:
        tr, _ = open_split(root, s)
        for row in tr:
            bad.update(prefixes.get(digest(row), ()))
        del tr
    valid = np.array([i for i in valid if i not in bad], dtype=np.int64)
    chosen = np.random.default_rng(seed).permutation(valid)
    result = dict(schema='s14-passkey-hosts-v1', split=split, seed=seed,
                  excluded_against=train_paths, host_rows=chosen.tolist(),
                  documents_sha256=_host_digest(chosen, rows[chosen]),
                  manifests_sha256={s: _sha256(root/f'{s}.jsonl') for s in ('A', 'B', 'ho')})
    Path(path).write_text(json.dumps(result, indent=2)+'\n')
    return result


def _host_rows(data_dir, n_docs, seed):
    """Verify the committed host set; presence of C never changes selection."""
    root = Path(data_dir)
    frozen = json.loads(HOSTS_FILE.read_text())
    chosen = np.asarray(frozen['host_rows'], np.int64)
    rows, _ = open_split(root, frozen['split'])
    if (seed != frozen['seed'] or len(np.unique(chosen)) != len(chosen)
            or not len(chosen) or chosen.min() < 0 or chosen.max() >= len(rows)):
        raise ValueError('frozen passkey host ids/seed mismatch; do not regenerate it')
    for s, digest in frozen['manifests_sha256'].items():
        if _sha256(root/f'{s}.jsonl') != digest:
            raise ValueError(f'frozen passkey host manifest mismatch: {s}')
    if _host_digest(chosen, rows[chosen]) != frozen['documents_sha256']:
        raise ValueError('frozen passkey host documents_sha256 mismatch; do not regenerate it')
    if n_docs > len(chosen) or n_docs < 1:
        raise ValueError(f'n_docs={n_docs}; only {len(chosen)} frozen clean ho rows available')
    chosen = chosen[:n_docs]
    split = frozen['split']
    return np.asarray(rows[chosen], np.int64), chosen, split


def _documents(hosts, tok, cfg, model_cfg, seed):
    C, T, S = model_cfg.chunk, cfg.seq_len, cfg.segment_len
    if T % C or T < 3*S or S != 2*C or list(cfg.barriers) != [2, 4, 6][:len(cfg.barriers)]:
        raise ValueError('passkey requires MINI geometry: two-chunk segments and barriers 2,4,6')
    if len(cfg.barriers) < 2 or cfg.barriers[1]*C != 2*S:
        raise ValueError('query region must start at the second attention barrier')
    if hosts.shape[1] < T+1:
        raise ValueError('held-out rows shorter than configured sequence')
    rng = np.random.default_rng(seed)
    plans = []
    for host in hosts:
        keys, ids, query = candidates(tok, rng)
        source = _encode(tok, f'The pass key is {keys[0]}. Remember it. {keys[0]} is the pass key.')
        offset = max(2, C//128)
        source_start = C-len(source)-offset
        if source_start < 1 or len(query)+ids.shape[1]+offset >= C:
            raise ValueError('source/query do not fit their registered chunks')
        base = np.array(host[:T+1], copy=True)
        base[source_start:source_start+len(source)] = source
        docs, spans = {}, {}
        for position, q0 in (('in_window', S-len(query)-ids.shape[1]), ('beyond', 2*S+offset)):
            doc = base.copy()
            doc[q0:q0+len(query)] = query
            a = q0+len(query)
            doc[a:a+ids.shape[1]] = ids[0]
            docs[position] = doc
            spans[position] = (q0, a, a+ids.shape[1])
        plans.append(dict(keys=keys, key_ids=ids, docs=docs, spans=spans,
                          source_span=(source_start, source_start+len(source))))
    return plans


def _context(model, doc, cfg, inner, position, adapt):
    return _context_batch(model, np.asarray(doc)[None, :], cfg, inner, position, adapt)


def _context_batch(model, docs, cfg, inner, position, adapt):
    """Use the same stream_eval updates, normalization, clip and barriers as G0.

    A non-writing sentinel chunk lets stream_eval perform the last source
    update. Causal attention makes pruning after it numerically equivalent to
    the full 8192-token document. Suffix K/V resets beyond the barrier.
    """
    C = model.cfg.chunk
    write_end = C if position == 'in_window' else 4*C
    stop = write_end+C
    ids = torch.from_numpy(docs[:, :stop].copy()).to(cfg.device)
    tgt = torch.from_numpy(docs[:, 1:stop+1].copy()).to(cfg.device)
    mask = tgt != model.cfg.bos_id
    imask = torch.zeros_like(mask)
    imask[:, :write_end] = True
    r = stream_eval(model, ids, tgt, mask, inner, adapt=adapt, barriers=list(cfg.barriers),
                    inner_mask=imask, fused_attn=cfg.fused_attn, return_fast=True,
                    nll_positions=torch.zeros_like(mask))
    W = r['fast']
    if position == 'beyond':
        past = _as_past([None]*model.cfg.suffix_len, W[0], model)
        segment_start, chunk_start = 4*C, 4*C
    else:
        # Only chunk 0 precedes the in-window query. Its K/V was produced at
        # W0 (pre-update), not at W1. Rebuild that exact cache once, without CE.
        with torch.no_grad():
            h = model.prefix_forward(ids[:, :2*C])
            w0 = _fast_values(model, len(docs))
            _, past = model.suffix_chunk(h[:, :C], w0,
                                         _as_past([None]*model.cfg.suffix_len, h, model),
                                         math_backend=not cfg.fused_attn, segment_pos=0)
        segment_start, chunk_start = 0, C
    return W, past, segment_start, chunk_start


@torch.no_grad()
def _score(model, doc, key_ids, span, context, cfg):
    return _score_batch(model, np.asarray(doc)[None, :], np.asarray(key_ids)[None, :], span, context, cfg)[0]


@torch.no_grad()
def _score_batch(model, docs, key_ids, span, context, cfg):
    W, past, segment_start, chunk_start = context
    _, a, end = span
    # For key at input index a, its predictive hidden state is at a-1.
    tok_losses, exact = [], None
    for j in range(key_ids.shape[1]):
        completion = key_ids[:, j]
        branch = docs[:, segment_start:end-1].copy()
        branch[:, a-segment_start:end-1-segment_start] = completion[:, :-1]
        ids = torch.from_numpy(branch).to(cfg.device)
        h = model.prefix_forward(ids)
        hc = h[:, chunk_start-segment_start:]
        x, _ = model.suffix_chunk(hc, W, past, math_backend=not cfg.fused_attn,
                                  segment_pos=chunk_start-segment_start)
        logits = model.head(x[:, a-1-chunk_start:end-1-chunk_start])
        target = torch.from_numpy(completion.copy()).to(cfg.device)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none')
        tok_losses.append(loss.reshape(target.shape).cpu().numpy())
        if j == 0:
            # Common space is given; conjunction equals free greedy digit match.
            exact = logits[:, 1:].argmax(-1).eq(target[:, 1:]).all(-1).cpu().numpy()
    # Leading space is shared by every candidate and is always unscored.
    losses = np.stack(tok_losses, axis=1)[:, :, 1:]
    means = losses.mean(axis=2)
    # Conservative ties: a foil tied with true prevents top1.
    ranks = 1+np.count_nonzero(means[:, 1:] <= means[:, :1], axis=1)
    return [dict(nll_true=float(m[0]), nll_foils=m[1:], nll_token=l,
                 lift=float(m[1:].mean()-m[0]), rank=int(r), exact=bool(e))
            for m, l, r, e in zip(means, losses, ranks, exact)]


def _read_plans(model, plans, cfg, inner, position, adapt, batch):
    """Batch documents with matching token geometry; preserve frozen ordering."""
    read = []
    start = 0
    while start < len(plans):
        span = plans[start]['spans'][position]
        end = start+1
        while (end < min(start+batch, len(plans))
               and plans[end]['spans'][position] == span
               and plans[end]['key_ids'].shape == plans[start]['key_ids'].shape):
            end += 1
        group = plans[start:end]
        docs = np.stack([p['docs'][position] for p in group])
        context = _context_batch(model, docs, cfg, inner, position, adapt)
        read.extend(_score_batch(model, docs, np.stack([p['key_ids'] for p in group]), span, context, cfg))
        del context
        if cfg.device == 'cpu' and _peak_gb() >= 6:
            raise MemoryError('CPU passkey peak RSS exceeded 6 GB')
        start = end
    return read


def _leak_check(model, plans, hosts, cfg):
    """Deterministic checkpoint audit: erase only the source, fp32, no writes.

    Audit up to the first eight docs (smokes may have fewer), all candidate digit
    tokens. Batch one and math attention keep the audit independent of read batch.
    Parameters are fp32 in the MINI model; only compute_dtype needs switching.
    """
    started = time.perf_counter()
    old_dtype = model.compute_dtype
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    deltas = []
    try:
        model.compute_dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False
        check_cfg = replace(cfg, dtype='fp32', fused_attn=False)
        inner = InnerCfg(lr=0, clip=cfg.inner_clip)
        for p, host in zip(plans[:8], hosts[:8]):
            doc = p['docs']['beyond']
            erased = doc.copy()
            s, e = p['source_span']
            erased[s:e] = host[s:e]
            def score(d):
                context = _context(model, d, check_cfg, inner, 'beyond', False)
                return _score(model, d, p['key_ids'], p['spans']['beyond'], context, check_cfg)['nll_token']
            deltas.append(float(np.max(np.abs(score(doc)-score(erased)))))
    finally:
        model.compute_dtype = old_dtype
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    maximum = max(deltas)
    return dict(leak_check='PASS' if np.isfinite(maximum) and maximum <= 1e-5 else 'FAIL',
                max_abs_delta_nll=maximum, per_doc_max_abs_delta_nll=deltas,
                leak_check_n_docs=len(deltas), leak_check_tolerance=1e-5,
                leak_check_dtype='fp32', leak_check_batch=1,
                leak_check_wall_s=time.perf_counter()-started)


def _peak_gb():
    unit = 1 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit/1e9


def passkey_read(model_path: str, cell_config_path: str, eta_mults: list[float], out_dir: str, *,
                 n_docs: int = 256, seed: int = 1014, device: str = 'cuda',
                 data_dir: str = 'data/p2data', noadapt: bool = True, batch: int = 1) -> dict:
    """Write passkey_<read>.npz and return JSON-serializable condition summaries.

    NLL and lift are mean nats per digit token, excluding the leading space.
    Llama-3 encodes space + 3-digit + 2-digit chunks; per-token arrays omit space.
    Eight distinct length-matched foils plus true give a 1/9 rank chance level.
    A five-digit vocabulary-wide exact match has a different chance baseline.
    """
    if not all(np.isfinite(e) and e > 0 for e in eta_mults):
        raise ValueError('eta_mults must be positive finite numbers')
    names = ['ce' if e == 1 else 'eta%03d' % round(e*100) for e in eta_mults]
    if len(set(names)) != len(names):
        raise ValueError('eta names collide')
    if not names and not noadapt:
        raise ValueError('no conditions requested')
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError('batch must be a positive integer')
    process_name = os.environ.get('PROCESS_NAME', 'TTM-s14-32')
    import setproctitle
    setproctitle.setproctitle(process_name)
    torch.set_num_threads(4)
    if device == 'cpu':
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:  # caller may already have initialized the pool
            pass
    cfg, extra = _config(cell_config_path, device, data_dir)
    tok = _tokenizer(data_dir, extra)
    hosts, host_ids, split = _host_rows(data_dir, n_docs, seed)
    model = _load_checkpoint(Path(model_path), cfg)
    model.requires_grad_(False)
    plans = _documents(hosts, tok, cfg, model.cfg, seed)
    if max(int(p['key_ids'].max()) for p in plans) >= model.cfg.vocab_size:
        raise ValueError('tokenizer vocabulary does not match model')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result = dict(model_path=model_path, model_sha256=_sha256(Path(model_path)),
                  config_path=cell_config_path, n_docs=n_docs, seed=seed, device=device,
                  dtype=cfg.dtype, torch_threads=torch.get_num_threads(), host_split=split,
                  host_rows=host_ids.tolist(), seq_len=cfg.seq_len, segment_len=cfg.segment_len,
                  barriers=list(cfg.barriers), K=K, rank_chance=1/(K+1), conditions={})
    result.update(host_set_sha256=json.loads(HOSTS_FILE.read_text())['documents_sha256'],
                  protocol=PASSKEY_PROTOCOL, batch=batch, requested_batch=batch,
                  tokenization='space + 3-digit + 2-digit chunks; space excluded')
    result.update(_leak_check(model, plans, hosts, cfg))
    print('PASSKEY_LEAK_CHECK '+json.dumps({k: result[k] for k in
          ('leak_check', 'max_abs_delta_nll', 'per_doc_max_abs_delta_nll', 'leak_check_n_docs')}), flush=True)
    if batch > 1:
        from .s14_batch import passkey_comparison, save_check
        subset = plans[:32]
        inner = InnerCfg(rule='ce',lr=cfg.rule_lr('ce'),clip=cfg.inner_clip,seed=seed)
        reference = {pos:_read_plans(model,subset,cfg,inner,pos,True,1)
                     for pos in ('in_window','beyond')}
        candidate = {pos:_read_plans(model,subset,cfg,inner,pos,True,batch)
                     for pos in ('in_window','beyond')}
        check = passkey_comparison(reference,candidate)
        check.update(requested_batch=batch,reference_batch=1,
                     effective_batch=batch if check['status']=='PASS' else 1,
                     checkpoint_sha256=result['model_sha256'],cell_config_sha256=_sha256(Path(cell_config_path)))
        save_check(out,'passkey',check)
        print(('PASSKEY_BATCH_CHECK' if check['status']=='PASS' else 'PASSKEY_BATCH_FALLBACK')
              +' '+json.dumps(check),flush=True)
        batch = check['effective_batch']
        result.update(batch=batch,batch_check=check)
    manifest_hash = hashlib.sha256()
    for p in plans:
        manifest_hash.update(p['docs']['beyond'].tobytes())
        manifest_hash.update(p['docs']['in_window'].tobytes())
        manifest_hash.update(p['key_ids'].tobytes())
    result['documents_sha256'] = manifest_hash.hexdigest()
    max_key_len = max(p['key_ids'].shape[1]-1 for p in plans)
    conditions = list(zip(names, eta_mults)) + ([('noadapt', 0.)] if noadapt else [])
    for label, eta in conditions:
        started = time.perf_counter()
        payload = dict(host_rows=host_ids, key=np.array([p['keys'][0] for p in plans]),
                       foil_keys=np.array([p['keys'][1:] for p in plans]),
                       key_token_count=np.array([p['key_ids'].shape[1]-1 for p in plans]),
                       eta=np.asarray(eta), seed=np.asarray(seed),
                       protocol=np.asarray(result['protocol']),
                       leak_check=np.asarray(result['leak_check']),
                       leak_check_max_abs_delta_nll=np.asarray(result['max_abs_delta_nll']),
                       leak_check_per_doc_max_abs_delta_nll=np.asarray(result['per_doc_max_abs_delta_nll']),
                       leak_check_n_docs=np.asarray(len(result['per_doc_max_abs_delta_nll'])),
                       leak_check_tolerance=np.asarray(1e-5),
                       leak_check_dtype=np.asarray('fp32'),
                       host_set_sha256=np.asarray(result['host_set_sha256']),
                       documents_sha256=np.asarray(result['documents_sha256']))
        summary = {}
        for position in ('in_window', 'beyond'):
            inner = InnerCfg(rule='ce', lr=cfg.rule_lr('ce')*eta, clip=cfg.inner_clip, seed=seed)
            read = _read_plans(model, plans, cfg, inner, position, label != 'noadapt', batch)
            for i, r in enumerate(read):
                print('PASSKEY_DOC '+json.dumps(dict(condition=label, position=position, doc=i,
                    lift=r['lift'], rank=r['rank'], exact=r['exact'], peak_gb=_peak_gb())), flush=True)
                if device == 'cpu' and _peak_gb() >= 6:
                    raise MemoryError('CPU passkey peak RSS exceeded 6 GB')
            for key in ('nll_true', 'nll_foils', 'lift', 'rank', 'exact'):
                payload[position+'_'+key] = np.asarray([r[key] for r in read])
            token = np.full((n_docs, K+1, max_key_len), np.nan, np.float32)
            for i, r in enumerate(read):
                token[i, :, :r['nll_token'].shape[1]] = r['nll_token']
            payload[position+'_nll_token'] = token
            payload[position+'_query_span'] = np.array([p['spans'][position] for p in plans])
            summary[position] = dict(nll_true=float(np.mean(payload[position+'_nll_true'])),
                nll_foil_mean=float(np.mean(payload[position+'_nll_foils'])),
                lift=float(np.mean(payload[position+'_lift'])),
                top1=float(np.mean(payload[position+'_rank'] == 1)),
                exact_match=float(np.mean(payload[position+'_exact'])))
        rec = dict(summary=summary, wall_s=time.perf_counter()-started, peak_rss_gb=_peak_gb())
        payload['summary'] = np.asarray(json.dumps(summary, sort_keys=True))
        payload['metadata'] = np.asarray(json.dumps({k: v for k, v in result.items() if k != 'conditions'}, sort_keys=True))
        np.savez_compressed(out/f'passkey_{label}.npz', **payload)
        result['conditions'][label] = rec
        print('PASSKEY_READ '+json.dumps(dict(condition=label, **rec)), flush=True)
    (out/'passkey_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--cell-config-path', required=True)
    parser.add_argument('--eta-mults', type=float, nargs='*')
    parser.add_argument('--arm', choices=['so', 'fo'], default='so')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--n-docs', type=int, default=256)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1014)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-dir', default='data/p2data')
    parser.add_argument('--no-noadapt', action='store_true')
    parser.add_argument('--process-name', default=os.environ.get('PROCESS_NAME', 'TTM-s14-32'))
    args = parser.parse_args(argv)
    os.environ['PROCESS_NAME'] = args.process_name
    passkey_read(args.model_path, args.cell_config_path, args.eta_mults or PASSKEY_ETAS[args.arm], args.out_dir,
                 n_docs=args.n_docs, seed=args.seed, device=args.device,
                 data_dir=args.data_dir, noadapt=not args.no_noadapt, batch=args.batch)


if __name__ == '__main__':
    main()
