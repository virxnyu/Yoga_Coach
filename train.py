"""Train YogaNet.

    python train.py                       # synthetic bootstrap only
    python train.py --real-dir data/real  # + your recorded clips (recommended)

Outputs: models/yogacoach.pt, outputs/metrics.json, outputs/*.png, outputs/train_log.txt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from yogacoach import dataio, rules, synth
from yogacoach.config import ERRORS, NUM_ERR, POSE_ERRORS, POSES, WINDOW
from yogacoach.engine import FLAG_THRESHOLD, fuse
from yogacoach.geometry import measure
from yogacoach.model import YogaNet, applicability_matrix, count_params, multitask_loss

OTHER = POSES.index("other")


# ----------------------------------------------------------------------------- helpers
def concat(a, b):
    if a is None:
        return b
    return {k: np.concatenate([a[k], b[k]]) for k in ("kp", "X", "pose", "errors", "is_target")}


def take(d, idx):
    return {k: d[k][idx] for k in ("kp", "X", "pose", "errors", "is_target")}


@torch.no_grad()
def predict(net, d, bs=1024):
    net.eval()
    P, C, Er = [], [], []
    for i in range(0, len(d["X"]), bs):
        x = torch.from_numpy(d["X"][i:i + bs])
        pl, cl, el = net(x)
        P.append(torch.softmax(pl, -1).numpy())
        C.append(torch.sigmoid(cl).numpy())
        Er.append(torch.sigmoid(el).numpy())
    return np.concatenate(P), np.concatenate(C), np.concatenate(Er)


def rule_matrices(d, window_stat="median"):
    """Rule-only predictions using the ground-truth pose. Returns (labels, confidences) (N,E)."""
    N = len(d["pose"])
    lab = np.zeros((N, NUM_ERR), np.float32)
    conf = np.zeros((N, NUM_ERR), np.float32)
    for pi, pose in enumerate(POSES):
        if pose == "other":
            continue
        sel = np.where(d["pose"] == pi)[0]
        if len(sel) == 0:
            continue
        kp = d["kp"][sel]
        kp = np.median(kp, axis=1) if window_stat == "median" else kp[:, -1]
        m = measure(kp)
        lab[sel] = rules.labels(pose, m)
        conf[sel] = rules.confidence_matrix(pose, m)
    return lab, conf


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def error_metrics(d, pred):
    """Multi-label metrics on target-pose samples, restricted to applicable errors."""
    A = applicability_matrix().numpy()
    sel = d["is_target"] > 0
    gt = d["errors"][sel]
    pr = pred[sel] * A[d["pose"][sel]]
    ap = A[d["pose"][sel]] > 0
    per = {}
    TP = FP = FN = 0
    f1s = []
    for ei, code in enumerate(ERRORS):
        m = ap[:, ei]
        if m.sum() == 0:
            continue
        tp = int(((pr[m, ei] == 1) & (gt[m, ei] == 1)).sum())
        fp = int(((pr[m, ei] == 1) & (gt[m, ei] == 0)).sum())
        fn = int(((pr[m, ei] == 0) & (gt[m, ei] == 1)).sum())
        p, r, f = prf(tp, fp, fn)
        per[code] = dict(precision=p, recall=r, f1=f, support=int(gt[m, ei].sum()))
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        f1s.append(f)
    gi, pi_ = gt.sum(1) > 0, pr.sum(1) > 0
    tp = int((gi & pi_).sum()); fp = int((~gi & pi_).sum()); fn = int((gi & ~pi_).sum())
    p, r, f = prf(tp, fp, fn)
    return dict(
        micro_f1=prf(TP, FP, FN)[2], macro_f1=float(np.mean(f1s)),
        exact_match=float((pr == gt * 1.0).all(1).mean()) if False else float(((pr > 0) == (gt > 0)).all(1).mean()),
        correct_vs_incorrect=dict(accuracy=float((gi == pi_).mean()), precision_incorrect=p,
                                  recall_incorrect=r, f1_incorrect=f),
        per_error=per)


def pose_metrics(d, pose_p):
    pred = pose_p.argmax(1)
    cm = np.zeros((len(POSES), len(POSES)), int)
    for t, p in zip(d["pose"], pred):
        cm[t, p] += 1
    f1 = [prf(cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i])[2] for i in range(len(POSES))]
    return dict(accuracy=float((pred == d["pose"]).mean()), macro_f1=float(np.mean(f1)), confusion=cm.tolist())


def full_eval(net, d, name):
    pose_p, corr_p, err_p = predict(net, d)
    # judge errors using the GROUND-TRUTH pose so error quality is separated from pose recognition
    A = applicability_matrix().numpy()[d["pose"]]
    dl = ((err_p * A) >= FLAG_THRESHOLD).astype(np.float32)
    rl_lab, rl_conf = rule_matrices(d, "median")
    rl_last, _ = rule_matrices(d, "last")
    hy = ((fuse(err_p, rl_conf) * A) >= FLAG_THRESHOLD).astype(np.float32)
    out = dict(name=name, n=int(len(d["pose"])), pose=pose_metrics(d, pose_p),
               errors_dl=error_metrics(d, dl), errors_rules_window=error_metrics(d, rl_lab),
               errors_rules_single_frame=error_metrics(d, rl_last), errors_hybrid=error_metrics(d, hy))
    return out


def log_line(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=5000, help="synthetic sequences per pose (train)")
    ap.add_argument("--val-n", type=int, default=600)
    ap.add_argument("--test-n", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--real-dir", default=None, help="folder of recorded .npz clips (scripts/collect.py)")
    ap.add_argument("--real-repeat", type=int, default=5, help="oversampling factor for real windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model-out", default="models/yogacoach.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    fh = open(os.path.join(args.out, "train_log.txt"), "w")

    t0 = time.time()
    log_line(f"== generating synthetic data (seed {args.seed}) ==", fh)
    train = concat(synth.generate(args.train_n, seed=args.seed + 1, noise=1.0),
                   synth.generate(args.train_n // 2, seed=args.seed + 2, noise=1.8))
    val = synth.generate(args.val_n, seed=args.seed + 3, noise=1.4)
    test = synth.generate(args.test_n, seed=args.seed + 4, noise=1.4)
    test_noisy = synth.generate(args.test_n, seed=args.seed + 5, noise=3.0)
    log_line(f"train {len(train['pose'])}  val {len(val['pose'])}  test {len(test['pose'])}  "
             f"test_noisy {len(test_noisy['pose'])}  ({time.time() - t0:.1f}s)", fh)
    log_line("train label balance:\n" + synth.describe(train), fh)

    real_test = None
    if args.real_dir and os.path.isdir(args.real_dir):
        real = dataio.load_real(args.real_dir)
        if real is not None:
            nfiles = len(real["files"])
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(nfiles)
            n_test = max(1, int(round(0.15 * nfiles))) if nfiles >= 8 else 0
            n_val = max(1, int(round(0.15 * nfiles))) if nfiles >= 8 else 0
            test_f, val_f = set(order[:n_test]), set(order[n_test:n_test + n_val])
            f = real["file"]
            tr = np.array([i for i in range(len(f)) if f[i] not in test_f and f[i] not in val_f])
            log_line(f"real clips: {nfiles} files, {len(f)} windows (train {len(tr)})", fh)
            rt = take(real, tr)
            for _ in range(args.real_repeat):
                train = concat(train, rt)
            if n_val:
                val = concat(val, take(real, np.array([i for i in range(len(f)) if f[i] in val_f])))
            if n_test:
                real_test = take(real, np.array([i for i in range(len(f)) if f[i] in test_f]))

    net = YogaNet()
    log_line(f"model parameters: {count_params(net):,}", fh)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(train["X"]) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    applic = net.applic

    Xtr = torch.from_numpy(train["X"]); ptr = torch.from_numpy(train["pose"])
    etr = torch.from_numpy(train["errors"]); ttr = torch.from_numpy(train["is_target"])
    Xv = torch.from_numpy(val["X"]); pv = torch.from_numpy(val["pose"])
    ev = torch.from_numpy(val["errors"]); tv = torch.from_numpy(val["is_target"])

    hist = dict(train_loss=[], val_loss=[], val_pose_acc=[], val_err_f1=[])
    best, best_state, bad = 1e9, None, 0
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        run, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            out = net(Xtr[idx])
            loss, _ = multitask_loss(out, ptr[idx], etr[idx], ttr[idx], applic)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); nb += 1
        net.eval()
        with torch.no_grad():
            vo = net(Xv)
            vloss, _ = multitask_loss(vo, pv, ev, tv, applic)
        pose_acc = float((vo[0].argmax(1) == pv).float().mean())
        pred = ((torch.sigmoid(vo[2]) * applic[pv]) >= 0.5).float().numpy()
        f1 = error_metrics(val, pred)["micro_f1"]
        hist["train_loss"].append(run / nb); hist["val_loss"].append(vloss.item())
        hist["val_pose_acc"].append(pose_acc); hist["val_err_f1"].append(f1)
        log_line(f"epoch {ep:02d}  train {run / nb:.4f}  val {vloss.item():.4f}  "
                 f"pose-acc {pose_acc:.3f}  err-microF1 {f1:.3f}  ({time.time() - t0:.0f}s)", fh)
        if vloss.item() < best - 1e-4:
            best, bad = vloss.item(), 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log_line(f"early stop at epoch {ep}", fh)
                break
    net.load_state_dict(best_state)

    results = {"synthetic_test": full_eval(net, test, "synthetic_test (noise x1.4)"),
               "synthetic_test_noisy": full_eval(net, test_noisy, "synthetic_test_noisy (noise x3.0, unseen)")}
    if real_test is not None and len(real_test["pose"]):
        results["real_test"] = full_eval(net, real_test, "real_test (held-out clips)")
    for k, r in results.items():
        log_line(f"\n[{k}] n={r['n']}  pose acc={r['pose']['accuracy']:.3f}  pose macroF1={r['pose']['macro_f1']:.3f}", fh)
        for name in ("errors_dl", "errors_rules_single_frame", "errors_rules_window", "errors_hybrid"):
            e = r[name]
            log_line(f"   {name:27s} microF1={e['micro_f1']:.3f} macroF1={e['macro_f1']:.3f} "
                     f"correct/incorrect acc={e['correct_vs_incorrect']['accuracy']:.3f} "
                     f"F1(incorrect)={e['correct_vs_incorrect']['f1_incorrect']:.3f}", fh)

    torch.save(dict(state=net.state_dict(), cfg=net.cfg, poses=POSES, errors=ERRORS, window=WINDOW,
                    trained_on=dict(synthetic=len(train['pose']), real_dir=args.real_dir)), args.model_out)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(history=hist, results=results, args=vars(args), params=count_params(net)), f, indent=2)
    make_plots(hist, results, args.out)
    log_line(f"\nsaved {args.model_out}, {args.out}/metrics.json and plots  (total {time.time() - t0:.0f}s)", fh)


def make_plots(hist, results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ep = range(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="validation")
    ax[0].set_title("Multi-task loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["val_pose_acc"], color="tab:green"); ax[1].set_title("Validation pose accuracy")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.02)
    ax[2].plot(ep, hist["val_err_f1"], color="tab:red"); ax[2].set_title("Validation error micro-F1")
    ax[2].set_xlabel("epoch"); ax[2].set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out, "training_curves.png"), dpi=140); plt.close()

    r = results["synthetic_test"]
    cm = np.array(r["pose"]["confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(POSES))); ax.set_xticklabels(POSES, rotation=30)
    ax.set_yticks(range(len(POSES))); ax.set_yticklabels(POSES)
    for i in range(len(POSES)):
        for j in range(len(POSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Pose confusion (synthetic test)")
    plt.colorbar(im, fraction=.046); plt.tight_layout()
    plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for a, key in zip(ax, ("synthetic_test", "synthetic_test_noisy")):
        rr = results[key]
        names = ["errors_rules_single_frame", "errors_rules_window", "errors_dl", "errors_hybrid"]
        labels = ["rules (1 frame)", "rules (window)", "YogaNet", "hybrid"]
        vals = [rr[n]["micro_f1"] for n in names]
        bars = a.bar(labels, vals, color=["#bbb", "#999", "tab:blue", "tab:green"])
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
        a.set_title(rr["name"], fontsize=9); a.set_ylim(0, 1.08)
    ax[0].set_ylabel("error micro-F1")
    plt.tight_layout(); plt.savefig(os.path.join(out, "method_comparison.png"), dpi=140); plt.close()


if __name__ == "__main__":
    main()
