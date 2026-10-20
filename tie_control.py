#!/usr/bin/env python3
"""Control: bf16 produces exact probability ties; the harness breaks them by slot order (np.argmax).
Re-decide every tied row by uniform random choice among the tied slots (200 reseeds) and recompute headline numbers.
Usage: python tie_control.py results.parquet"""
import sys, json, numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1]); nq = df.question_rank.nunique()
P = df[["p0", "p1", "p2", "p3"]].to_numpy(); mx = P.max(1, keepdims=True); tied = (P == mx)
ties = tied.sum(1) > 1
rot = df.rotation.to_numpy(); ans = df.answer_underlying.to_numpy(); qr = df.question_rank.to_numpy()
rng = np.random.default_rng(0); acc, flip = [], []
for _ in range(200):
    d = df.pred_display.to_numpy().copy()
    for i in np.where(ties)[0]:
        d[i] = rng.choice(np.where(tied[i])[0])
    u = (d + rot) % 4
    c = (u == ans).astype(int)
    pu = pd.DataFrame({"q": qr, "u": u}).groupby("q").u.nunique()
    acc.append(c.mean()); flip.append((pu > 1).mean())
base_flip = (df.groupby("question_rank").pred_underlying.nunique() > 1).mean()
out = dict(tied_rows=int(ties.sum()), acc_positional=float(df.correct.mean()), acc_random_ties_mean=float(np.mean(acc)),
           acc_random_ties_range=[float(min(acc)), float(max(acc))], flip_positional=float(base_flip),
           flip_random_ties_mean=float(np.mean(flip)), flip_random_ties_range=[float(min(flip)), float(max(flip))])
print(json.dumps(out, indent=2))
