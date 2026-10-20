#!/usr/bin/env python3
"""Compute every headline number from a raw results parquet. Nothing is hand-typed downstream.
Usage: python analyze.py results.parquet [--out summary.json] [--boot 5000]
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--out", default=None); ap.add_argument("--boot", type=int, default=5000)
a = ap.parse_args()
df = pd.read_parquet(a.path)
nq = df.question_rank.nunique()
assert len(df) == 4 * nq and (df.groupby("question_rank").size() == 4).all()
rng = np.random.default_rng(0)

C = df.pivot(index="question_rank", columns="rotation", values="correct").to_numpy()          # nq x 4
P = df.pivot(index="question_rank", columns="rotation", values="pred_underlying").to_numpy()  # nq x 4
D = df.pivot(index="question_rank", columns="rotation", values="pred_display").to_numpy()
conf = df.pivot(index="question_rank", columns="rotation", values="confidence").to_numpy()


def stats(idx):
    c, p = C[idx], P[idx]
    acc_rot = c.mean(0)
    flip = (p != p[:, [0]]).any(1)          # underlying prediction not identical across 4 rotations
    stable = ~flip
    return dict(
        acc_mean=c.mean(), acc_min=acc_rot.min(), acc_max=acc_rot.max(), acc_range=acc_rot.max() - acc_rot.min(),
        acc_all_correct=c.all(1).mean(), acc_any_correct=c.any(1).mean(),
        flip_rate=flip.mean(),
        acc_on_flip=c[flip].mean() if flip.any() else np.nan,
        acc_on_stable=c[stable].mean() if stable.any() else np.nan,
    )


base = stats(np.arange(nq))
boots = {k: [] for k in base}
for _ in range(a.boot):
    s = stats(rng.integers(0, nq, nq))
    for k, v in s.items():
        boots[k].append(v)
ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boots.items()}
# The bootstrap of a max-min statistic is biased upward (its interval can exclude the point estimate),
# so the rotation-range CI is NOT reported. Use per-rotation CIs and Cochran's Q instead.
for k in ("acc_min", "acc_max", "acc_range"):
    ci.pop(k, None)
from scipy.stats import chi2
k_, N_ = 4, nq
Tj, Ti = C.sum(0), C.sum(1)
q_denom = k_ * Ti.sum() - (Ti ** 2).sum()
cochran_q = float((k_ - 1) * (k_ * (Tj ** 2).sum() - Tj.sum() ** 2) / q_denom) if q_denom > 0 else float("nan")
cochran_p = float(chi2.sf(cochran_q, k_ - 1))
acc_rot_ci = []
for r in range(4):
    bs = [C[rng.integers(0, nq, nq), r].mean() for _ in range(a.boot)]
    acc_rot_ci.append([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])

# --- baselines so the flip rate is interpretable ---
# (1) pure position-bias predictor: picks display slot ~ observed marginal, independent of content
marg = np.bincount(D.ravel(), minlength=4) / D.size
sim_flip = []
for _ in range(2000):
    d = rng.choice(4, size=(nq, 4), p=marg)
    u = (d + np.arange(4)[None, :]) % 4
    sim_flip.append((u != u[:, [0]]).any(1).mean())
# (2) content-only predictor with the model's own accuracy would give flip ~ (1-acc_all) bounded below by 0
# (3) shuffled control: break question-option link by pairing each question's rotations with other questions' (model p)
perm_flip = []
for _ in range(2000):
    # keep each rotation's underlying prediction marginal but shuffle across questions independently per rotation
    p = np.stack([rng.permutation(P[:, r]) for r in range(4)], 1)
    perm_flip.append((p != p[:, [0]]).any(1).mean())

# position bias
disp_dist = {int(k): float(v) for k, v in zip(range(4), marg)}
acc_by_ans_pos = {int(k): float(g.correct.mean()) for k, g in df.groupby("answer_display")}
n_by_ans_pos = {int(k): int(len(g)) for k, g in df.groupby("answer_display")}

from scipy.stats import chi2_contingency
ct=pd.crosstab(df.answer_display, df.correct).to_numpy()
slot_chi2=chi2_contingency(ct)
slot_test=dict(chi2=float(slot_chi2[0]),df=int(slot_chi2[2]),p=float(slot_chi2[1]))

# calibration (10 equal-width bins, top-label confidence)
cf, cr = df.confidence.to_numpy(), df.correct.to_numpy()
bins = np.clip((cf * 10).astype(int), 0, 9)
ece = sum(abs(cf[bins == b].mean() - cr[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())

out = dict(
    n_questions=int(nq), n_predictions=int(len(df)), model=df.model.iloc[0], model_revision=df.model_revision.iloc[0],
    dataset_revision=df.dataset_revision.iloc[0], seed=int(df.seed.iloc[0]),
    metrics={k: float(v) for k, v in base.items()}, ci95_bootstrap_over_questions=ci,
    acc_by_rotation=[float(x) for x in C.mean(0)], acc_by_rotation_ci95=acc_rot_ci,
    cochran_q_rotations=dict(Q=cochran_q, df=3, p=cochran_p),
    chance_floor=0.25,
    flip_rate_if_position_only_predictor=dict(mean=float(np.mean(sim_flip)), ci95=[float(np.percentile(sim_flip, 2.5)), float(np.percentile(sim_flip, 97.5))]),
    flip_rate_if_rotation_predictions_shuffled_across_questions=dict(mean=float(np.mean(perm_flip)), ci95=[float(np.percentile(perm_flip, 2.5)), float(np.percentile(perm_flip, 97.5))]),
    display_position_pick_distribution=disp_dist,
    accuracy_by_correct_answer_display_position=acc_by_ans_pos, n_by_correct_answer_display_position=n_by_ans_pos,
    slot_dependence_of_correctness_chi2=slot_test,
    ece_10bin=float(ece), mean_confidence=float(cf.mean()), exact_ties=int(df.exact_tie.sum()),
)
print(json.dumps(out, indent=2))
if a.out:
    open(a.out, "w").write(json.dumps(out, indent=2) + "\n")
