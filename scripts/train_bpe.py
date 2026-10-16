"""Train the frozen 16k byte-level BPE used by the micro16k testbed."""
from __future__ import annotations
import argparse
import hashlib
import os
import json
from pathlib import Path
from tttmem.p2.data import DCLM_REPO, DCLM_REVISION, iter_parquet, _set_process_name

def train(output: str | Path, *, revision=DCLM_REVISION, max_bytes=100_000_000,
          max_shards=8, seed=991, process_name="TTM-tbb-01-bpe"):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from huggingface_hub import HfApi
    _set_process_name(process_name)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(
        DCLM_REPO, repo_type="dataset", revision=revision) if f.endswith(".parquet"))
    if len(files) < max_shards:
        raise RuntimeError(f"need {max_shards} parquet shards, found {len(files)}")
    reserved = files[-max_shards:]
    texts, total = [], 0
    scratch = output.parent / ".bpe-shards"
    scratch.mkdir(exist_ok=True)
    for doc in iter_parquet(DCLM_REPO, revision, seed, scratch, shards=reserved):
        text = doc.get("text") or ""
        if not text:
            continue
        texts.append(text)
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break
    if not texts:
        raise RuntimeError("no DCLM text was available for BPE training")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=16384, min_frequency=2,
                                  special_tokens=["<|bos|>", "<|eos|>", "<unk>"])
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(output))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    output.with_suffix(output.suffix + ".json").write_text(json.dumps({
        "vocab_size": tok.get_vocab_size(), "bos_id": tok.token_to_id("<|bos|>"),
        "eos_id": tok.token_to_id("<|eos|>"), "bytes": total,
        "dataset_revision": revision, "sha256": digest, "shards": reserved
    }, indent=2) + "\n")
    print(f"BPE path={output} vocab={tok.get_vocab_size()} bytes={total} sha256={digest}", flush=True)
    return digest

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=DCLM_REVISION)
    ap.add_argument("--max-bytes", type=int, default=100_000_000)
    ap.add_argument("--max-shards", type=int, default=8)
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-tbb-01-bpe"))
    args = ap.parse_args(argv)
    train(args.out, revision=args.revision, max_bytes=args.max_bytes,
          max_shards=args.max_shards, process_name=args.process_name)

if __name__ == "__main__":
    main()
