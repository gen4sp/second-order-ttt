import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tttmem.p2 import data
from tttmem.p2.model import PRESETS, build_model
from tttmem.p2.probe import find_probes
from tttmem.p2.tbb_analysis import tbb_analysis

def local_tok(path):
    tok=Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer=pre_tokenizers.ByteLevel()
    tok.train_from_iterator(["alpha beta gamma delta "*100], trainers.BpeTrainer(vocab_size=32,special_tokens=["<|bos|>","<|eos|>","<unk>"]))
    tok.save(str(path)); return tok

def test_local_tokenizer_drives_ids_and_exclusions(tmp_path):
    path=tmp_path/"tokenizer.json"; tok=local_tok(path)
    loaded=data.load_tokenizer(path=str(path))
    info=data.tokenizer_special_ids(loaded)
    mask=data.excluded_vocab(loaded, info["vocab_size"])
    assert info["bos_id"] == loaded.token_to_id("<|bos|>")
    assert info["eos_id"] == loaded.token_to_id("<|eos|>")
    assert len(mask) == loaded.get_vocab_size()

def test_micro16k_preset_and_probe_layout():
    cfg=PRESETS["micro16k"]
    assert (cfg.vocab_size,cfg.chunk,cfg.window,cfg.suffix_len)==(16384,512,4096,2)
    assert build_model("micro16k", compute_dtype=torch.float32).cfg == cfg
    x=np.tile(np.arange(10,dtype=np.int64),1000)
    p=find_probes(x[:-1],x[1:],512,barriers=[2,4,6],probe_start=4,excluded=np.zeros(16384,bool))
    assert p.pos.size >= 1 and np.all(p.pos//512 >= 4)

def test_tbb_analysis_synthetic_labels(tmp_path):
    root=tmp_path
    for seed in range(3):
        for arm,m in (("e2e2",.30),("e2e1-ce",.10),("e2e12-p125",.20),("e2e1-rp",.105)):
            for step in (0,300,600,1200):
                d=root/"micro16k"/arm/f"s{seed}"; d.mkdir(parents=True,exist_ok=True)
                np.savez(d/f"eval_step{step:04d}.npz", doc_fP_focal=np.array([m+.2]), doc_fP_nonfocal=np.array([m]),
                         doc_noadapt=np.array([1.0]), doc_adapt=np.array([.98]),
                         probe_stratum=np.array([0,1,2,3,4]), probe_dist=np.array([256,512,1024,2048,4096]),
                         probe_delta=np.array([-.05,-.01,.02,.04,.08]),
                         src_cot_adapt=np.array([1.0]), ratio_cot=np.array([1.0]))
        np.savez(root/"micro16k"/"e2e2"/f"s{seed}"/"eval_meta.npz", probe_stratum=np.array([2]),probe_pos=np.array([1]),probe_dist=np.array([1]),focal_offset=np.array([0,1]),focal_pos=np.array([0]))
    result=tbb_analysis(root)
    assert result["label"] in ("QUALIFIED","NOT QUALIFIED")
    assert Path(root/"tbb_analysis.json").exists() and Path(root/"tbb_analysis.md").exists()
