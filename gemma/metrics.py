"""Massive-STEPS-compatible metrics.  NDCG uses 1/log2(rank+2) (repo convention)."""
import numpy as np


def ranking_metrics(ranks, eval_ks=(1, 5, 10), ndcg_ks=(5, 10)):
    ranks = np.asarray(ranks)
    m = {f"acc@{k}": float((ranks < k).mean()) for k in eval_ks}
    for k in ndcg_ks:
        hit = ranks < k
        m[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).mean())
    m["mrr"] = float((1.0 / (ranks + 1)).mean())
    return m


def classification_metrics(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}
