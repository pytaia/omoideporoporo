import json
from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC_DIR = Path(__file__).resolve().parent / "public"

COARSE_GRID = np.arange(0.5, 4.01, 0.01)
BACKDOOR_HEAD = 6
BACKDOOR_THRESH = 2.0


def freq_spectrum(t, feats, freq_grid):
    ang = 2 * np.pi * np.outer(freq_grid, t)
    C = np.cos(ang)
    S = np.sin(ang)
    Cc = np.sum(C * C, axis=1)
    Ss = np.sum(S * S, axis=1)
    Cs = np.sum(C * S, axis=1)
    det = Cc * Ss - Cs * Cs
    Py = C @ feats
    Qy = S @ feats
    c_coef = (Ss[:, None] * Py - Cs[:, None] * Qy) / det[:, None]
    s_coef = (Cc[:, None] * Qy - Cs[:, None] * Py) / det[:, None]
    mag = np.sqrt(c_coef ** 2 + s_coef ** 2)
    return mag.mean(axis=1)


def regress_single(t, feats, freq):
    ang = 2 * np.pi * freq * t
    C = np.cos(ang)
    S = np.sin(ang)
    Cc = np.sum(C * C)
    Ss = np.sum(S * S)
    Cs = np.sum(C * S)
    det = Cc * Ss - Cs * Cs
    Py = C @ feats
    Qy = S @ feats
    c_coef = (Ss * Py - Cs * Qy) / det
    s_coef = (Cc * Qy - Cs * Py) / det
    return c_coef + 1j * s_coef


def load_passes(df, group_col):
    passes = []
    for sid, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("t_s")
        t = g["t_s"].to_numpy()
        feats = g[[f"feat_{j}" for j in range(20)]].to_numpy()
        label = g[group_col].iloc[0]
        passes.append((sid, label, t, feats))
    return passes


def main():
    train_df = pd.read_csv(PUBLIC_DIR / "train.csv")
    test_df = pd.read_csv(PUBLIC_DIR / "task.csv")
    id_to_char = json.load(open(PUBLIC_DIR / "employee_number.json"))

    train_passes = load_passes(train_df, "identity_index")
    test_passes = load_passes(test_df, "slot")
    test_passes.sort(key=lambda x: x[1])
    n_slots = len(test_passes)

    coarse_freq = {}
    for sid, label, t, feats in train_passes:
        spec = freq_spectrum(t, feats, COARSE_GRID)
        coarse_freq[sid] = COARSE_GRID[np.argmax(spec)]

    test_coarse_freq = {}
    for sid, slot, t, feats in test_passes:
        spec = freq_spectrum(t, feats, COARSE_GRID)
        test_coarse_freq[sid] = COARSE_GRID[np.argmax(spec)]

    kept = []
    dropped_backdoor = []
    for sid, label, t, feats in train_passes:
        head = feats[:min(BACKDOOR_HEAD, feats.shape[0]), 0:3]
        if np.mean(np.abs(head)) > BACKDOOR_THRESH:
            dropped_backdoor.append(sid)
        else:
            kept.append((sid, label, t, feats))

    freqs = np.array([coarse_freq[sid] for sid, *_ in kept])
    order = np.argsort(freqs)
    sorted_sids = [kept[i][0] for i in order]
    sorted_labels = [kept[i][1] for i in order]
    sorted_freqs = freqs[order]

    n_identities = train_df["identity_index"].nunique()
    gaps = np.diff(sorted_freqs)
    boundary_idx = np.sort(np.argsort(gaps)[-(n_identities - 1):])

    cluster_id = np.zeros(len(sorted_freqs), dtype=int)
    b, cur = 0, 0
    for i in range(1, len(sorted_freqs)):
        if b < len(boundary_idx) and i - 1 == boundary_idx[b]:
            cur += 1
            b += 1
        cluster_id[i] = cur
    n_clusters = cur + 1

    cluster_label_counts = {}
    for cl in range(n_clusters):
        mask = cluster_id == cl
        counts = {}
        for lbl in np.array(sorted_labels)[mask]:
            counts[lbl] = counts.get(lbl, 0) + 1
        cluster_label_counts[cl] = counts

    label_home_cluster = {}
    for cl, counts in cluster_label_counts.items():
        total = sum(counts.values())
        for lbl, n in counts.items():
            purity = n / total
            prev = label_home_cluster.get(lbl)
            if prev is None or purity > prev[1]:
                label_home_cluster[lbl] = (cl, purity, n)

    kept_clean = []
    dropped_label_noise = 0
    for sid, lbl, cl in zip(sorted_sids, sorted_labels, cluster_id):
        home_cl, purity, n = label_home_cluster[lbl]
        if cl == home_cl:
            kept_clean.append(sid)
        else:
            dropped_label_noise += 1

    kept_by_sid = {sid: (label, t, feats) for sid, label, t, feats in kept}
    clean_by_label = {}
    for sid in kept_clean:
        label, t, feats = kept_by_sid[sid]
        clean_by_label.setdefault(label, []).append((sid, t, feats))

    ref_freq = {}
    rel_phase_template = {}
    for label, items in clean_by_label.items():
        coarse = float(np.mean([coarse_freq[sid] for sid, t, feats in items]))
        fine_grid = np.arange(coarse - 0.05, coarse + 0.05, 0.0005)
        acc = np.zeros(len(fine_grid))
        for sid, t, feats in items:
            acc += freq_spectrum(t, feats, fine_grid)
        f_hat = fine_grid[np.argmax(acc)]
        ref_freq[label] = f_hat

        unit_deltas = []
        for sid, t, feats in items:
            z = regress_single(t, feats, f_hat)
            d = np.angle(z * np.conj(z[0]))
            unit_deltas.append(np.exp(1j * d))
        rel_phase_template[label] = np.angle(np.mean(np.array(unit_deltas), axis=0))

    naive_ref = {}
    by_label_all = {}
    for sid, label, t, feats in train_passes:
        by_label_all.setdefault(label, []).append(coarse_freq[sid])
    for label, vals in by_label_all.items():
        naive_ref[label] = float(np.mean(vals))

    decoded = []
    rescued_slots = []
    broken_slots = []
    for sid, slot, t, feats in test_passes:
        f_naive = test_coarse_freq[sid]
        naive_pred = min(naive_ref, key=lambda c: abs(naive_ref[c] - f_naive))

        best_c, best_score = None, -1.0
        for c, f_c in ref_freq.items():
            z = regress_single(t, feats, f_c)
            u = z * np.exp(-1j * rel_phase_template[c])
            score = abs(np.sum(u)) / len(u)
            if score > best_score:
                best_score, best_c = score, c

        if best_c != naive_pred:
            rescued_slots.append((slot, naive_pred, best_c))

        ch = id_to_char.get(str(best_c))
        if ch is None:
            broken_slots.append(slot)
            ch = "?"
        decoded.append(ch)

    flag = "".join(decoded)
    print(flag)


if __name__ == "__main__":
    main()
