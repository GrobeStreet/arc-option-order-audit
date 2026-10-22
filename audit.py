#!/usr/bin/env python3
"""Option-order robustness audit of ARC-Challenge (test split).

For every 4-option question, the answer options are cyclically rotated (4 rotations)
and a small open model is scored on the label-token logits (A-D) after a raw-completion
prompt (no chat template). Everything is pinned: model revision, dataset revision, seed.

Usage:  python audit.py --model Qwen/Qwen2.5-0.5B-Instruct --out results_qwen05b.parquet
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers, datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["A", "B", "C", "D"]
DATASET = "allenai/ai2_arc"
DATASET_REV = "210d026faf9955653af8916fad021475a3f00453"
MODELS = {  # pinned revisions
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "a10cc1512eabd3dde888204e902eca88bddb4951",
}


def rotate(seq, r):
    """Cyclic left rotation by r: displayed position i shows underlying option (i+r) % 4."""
    return [seq[(i + r) % 4] for i in range(4)]


def display_to_underlying(pos, r):
    return (pos + r) % 4


def underlying_to_display(u, r):
    return (u - r) % 4


def prompt(question, choices):
    lines = [question.strip(), ""] + [f"{l}. {c}" for l, c in zip(LABELS, choices)] + ["", "Answer:"]
    return "\n".join(lines)


def label_ids(tok):
    ids = []
    for l in LABELS:
        sp = tok.encode(" " + l, add_special_tokens=False)
        ids.append(sp[0] if len(sp) == 1 else tok.encode(l, add_special_tokens=False)[0])
    assert len(set(ids)) == 4
    return ids


def sha256(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=0, help="0 = all eligible questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--out", default="results.parquet")
    a = ap.parse_args()
    rev = MODELS[a.model]
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    ds = load_dataset(DATASET, "ARC-Challenge", split="test", revision=DATASET_REV)
    items = [(i, x) for i, x in enumerate(ds) if len(x["choices"]["text"]) == 4
             and x["answerKey"] in x["choices"]["label"]]
    idx = list(range(len(items)))
    random.Random(a.seed).shuffle(idx)
    if a.n:
        idx = idx[: a.n]
    print(f"eligible 4-option questions: {len(items)} of {len(ds)}; using {len(idx)}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, dtype=getattr(torch, a.dtype), low_cpu_mem_usage=True).eval()
    torch.set_num_threads(2)
    ids = label_ids(tok)

    rows = []
    part = Path(a.out + ".partial.parquet")
    done = 0
    if part.exists():
        prev = pd.read_parquet(part)
        rows = prev.to_dict("records")
        done = prev.question_rank.nunique()
        print(f"resuming from checkpoint: {done} questions already done", flush=True)
    for rank, j in enumerate(idx):
        if rank < done:
            continue
        di, x = items[j]
        texts = x["choices"]["text"]
        ans = x["choices"]["label"].index(x["answerKey"])
        for r in range(4):
            ch = rotate(texts, r)
            inp = tok(prompt(x["question"], ch), return_tensors="pt")
            with torch.no_grad():
                lg = model(**inp).logits[0, -1][ids]
            p = torch.softmax(lg.float(), 0).numpy()
            pos = int(np.argmax(p))
            srt = np.sort(p)[::-1]
            rows.append(dict(
                question_rank=rank, dataset_index=di, id=x["id"], rotation=r,
                answer_underlying=ans, answer_display=underlying_to_display(ans, r),
                pred_display=pos, pred_underlying=display_to_underlying(pos, r),
                correct=int(display_to_underlying(pos, r) == ans), confidence=float(p[pos]),
                p0=float(p[0]), p1=float(p[1]), p2=float(p[2]), p3=float(p[3]),
                margin=float(srt[0] - srt[1]), exact_tie=bool(srt[0] == srt[1]),
                model=a.model, model_revision=rev, dataset_revision=DATASET_REV, seed=a.seed))
        if (rank + 1) % 5 == 0:
            pd.DataFrame(rows).to_parquet(part, index=False)
        if (rank + 1) % 50 == 0:
            print(f"  {rank+1}/{len(idx)} questions", flush=True)

    out = Path(a.out)
    pd.DataFrame(rows).to_parquet(out, index=False)
    prov = dict(model=a.model, model_revision=rev, dataset=DATASET, dataset_revision=DATASET_REV,
                config="ARC-Challenge/test", eligible=len(items), n_used=len(idx), seed=a.seed,
                dtype=a.dtype, device="cpu", python=sys.version.split()[0], platform=platform.platform(),
                torch=torch.__version__, transformers=transformers.__version__, datasets=datasets.__version__,
                numpy=np.__version__, pandas=pd.__version__, output=out.name, output_sha256=sha256(out),
                prompt="question + blank + A-D options + blank + 'Answer:'; raw completion, no chat template")
    Path(str(out) + ".provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
