"""Runs every experiment and writes raw results + figures into results/.

    py run_experiments.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import export_text

from explain import explain, confidence_word
from features import FEATURE_NAMES
from schedulers import SCHEDULERS, LEFT, RIGHT, TIME_PER_TRACK_NS
from selector import Selector, make_dataset, movements
from workloads import GENERATORS

OUT = Path("results")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_SEED, TEST_SEED, SHIFT_SEED = 307, 2026, 4
N_TRAIN, N_TEST = 400, 200            # windows per workload type
PHASES = [("sequential", 30), ("random", 30), ("bursty", 30)]
NAMES = list(SCHEDULERS)


def evaluate_test_set(sel, X, costs, kinds):
    rows = []
    for x, c, kind in zip(X, costs, kinds):
        proba = sel.tree.predict_proba([x])[0]
        k = int(np.argmax(proba))
        choice = sel.tree.classes_[k]
        best = min(c.values())
        rows.append({"workload": kind, "choice": choice, "confidence": proba[k],
                     "correct": c[choice] == best, "selector": c[choice],
                     "oracle": best, **c})
    return pd.DataFrame(rows)


def calibration_table(df, bins=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    df = df.assign(bin=pd.cut(df.confidence, bins, right=False))
    table = df.groupby("bin", observed=True).agg(
        windows=("correct", "size"),
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
    ).reset_index()
    table["bin"] = table["bin"].astype(str)
    ece = (table.windows * (table.accuracy - table.mean_confidence).abs()).sum() / len(df)
    return table, ece


def run_shift(selectors):
    """One long trace whose workload type changes at fixed points.
    Each policy keeps its own head position and direction across windows."""
    rng = np.random.default_rng(SHIFT_SEED)
    windows = [(kind, GENERATORS[kind](rng)) for kind, n in PHASES for _ in range(n)]

    policies = NAMES + list(selectors) + ["Oracle"]
    state = {p: (100, LEFT) for p in policies}
    rows, notes = [], []
    for i, (kind, reqs) in enumerate(windows):
        for p in policies:
            head, direction = state[p]
            costs = movements(reqs, head, direction)
            best = min(costs.values())
            if p in SCHEDULERS:
                choice, conf = p, np.nan
            elif p == "Oracle":
                choice, conf = min(costs, key=costs.get), np.nan
            else:
                sel = selectors[p]
                choice, conf, x = sel.choose(reqs, head, direction)
                notes.append({"window": i, "workload": kind, "selector": p,
                              "choice": choice, "confidence": round(conf, 3),
                              "confidence_word": confidence_word(conf),
                              "correct": costs[choice] == best,
                              "explanation": explain(sel.tree, x, choice, conf)})
            s = SCHEDULERS[choice](reqs, head, direction)
            state[p] = (s.order[-1], s.direction)
            rows.append({"window": i, "workload": kind, "policy": p, "choice": choice,
                         "confidence": conf, "movement": s.movement,
                         "best_possible": best, "correct": s.movement == best})
    return pd.DataFrame(rows), pd.DataFrame(notes)


def plot_static(static):
    cols = NAMES + ["Selector", "Oracle"]
    ax = static.set_index("workload")[cols].plot.bar(figsize=(7, 3.4), rot=0, width=0.8)
    ax.set_ylabel("Mean head movement per window")
    ax.set_xlabel("")
    ax.set_title("Test windows (20 requests each)")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "static_by_workload.png", dpi=200)
    plt.close()


def plot_shift(timeline, notes):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for p in NAMES + ["Selector-A", "Oracle"]:
        d = timeline[timeline.policy == p]
        style = "--" if p == "Oracle" else "-"
        width = 2.2 if p == "Selector-A" else 1.2
        a1.plot(d.window, d.movement.rolling(5, min_periods=1).mean(), style,
                lw=width, label=p)
    a1.set_yscale("log")
    a1.set_ylabel("Head movement, log scale\n(5-window mean)")
    a1.legend(ncol=3, fontsize=8, loc="lower right")

    for name, marker in [("Selector-A", "o"), ("Selector-B", "s")]:
        d = notes[notes.selector == name]
        ok, bad = d[d.correct], d[~d.correct]
        a2.scatter(ok.window, ok.confidence, s=14, marker=marker, label=f"{name} correct")
        a2.scatter(bad.window, bad.confidence, s=22, marker="x", label=f"{name} wrong")
    a2.set_ylabel("Confidence")
    a2.set_xlabel("Window (20 requests each)")
    a2.set_ylim(0, 1.05)
    a2.legend(ncol=2, fontsize=7)

    edge = 0
    for kind, n in PHASES:
        for ax in (a1, a2):
            ax.axvline(edge - 0.5, color="grey", lw=0.8)
        a1.text(edge + n / 2, a1.get_ylim()[1] * 0.97, kind, ha="center", va="top")
        edge += n
    fig.tight_layout()
    fig.savefig(FIG / "shift_timeline.png", dpi=200)
    plt.close(fig)


def plot_calibration(table):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(table.mean_confidence, table.accuracy, "o-", label="selector")
    for _, r in table.iterrows():
        ax.annotate(str(r.windows), (r.mean_confidence, r.accuracy),
                    textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Fraction correct")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Reliability (test windows)")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=200)
    plt.close(fig)


def main():
    report = []

    # Training data. Selector-B never sees bursty windows, to test an unseen shift.
    X, y, _, kinds = make_dataset(np.random.default_rng(TRAIN_SEED), GENERATORS, N_TRAIN)
    no_bursty = np.array([k != "bursty" for k in kinds])
    selectors = {
        "Selector-A": Selector().fit(X, y),
        "Selector-B": Selector().fit(X[no_bursty], y[no_bursty]),
    }
    tree = selectors["Selector-A"].tree
    (OUT / "decision_tree.txt").write_text(export_text(tree, feature_names=FEATURE_NAMES))
    importances = pd.Series(tree.feature_importances_, FEATURE_NAMES).sort_values(ascending=False)
    report.append("Label counts (train): " + str(pd.Series(y).value_counts().to_dict()))
    report.append("Feature importance (Selector-A):\n" + importances.round(3).to_string())

    # Test set: fresh windows from a different seed.
    Xt, _, costs_t, kinds_t = make_dataset(np.random.default_rng(TEST_SEED), GENERATORS, N_TEST)
    test = evaluate_test_set(selectors["Selector-A"], Xt, costs_t, kinds_t)
    test.to_csv(OUT / "test_predictions.csv", index=False)

    static = test.groupby("workload")[NAMES + ["selector", "oracle"]].mean().round(1)
    static = static.rename(columns={"selector": "Selector", "oracle": "Oracle"})
    static = static.reindex([k for k, _ in PHASES]).reset_index()
    static.to_csv(OUT / "static_comparison.csv", index=False)
    plot_static(static)
    report.append("Mean movement per test window:\n" + static.to_string(index=False))
    report.append(f"Selector-A test accuracy (picked a lowest-movement scheduler): "
                  f"{test.correct.mean():.3f}")
    for name in NAMES:
        report.append(f"  'always {name}' would be best in {(test[name] == test.oracle).mean():.3f} of windows")

    table, ece = calibration_table(test)
    table.to_csv(OUT / "calibration_bins.csv", index=False)
    plot_calibration(table)
    right, wrong = test[test.correct].confidence.mean(), test[~test.correct].confidence.mean()
    report.append("Calibration bins:\n" + table.round(3).to_string(index=False))
    report.append(f"Expected calibration error = {ece:.3f}; mean confidence when correct = "
                  f"{right:.3f}, when wrong = {wrong:.3f}")
    extra = (test.selector - test.oracle)[~test.correct]
    report.append(f"When wrong, the selector moved {extra.mean():.1f} extra cylinders on average "
                  f"({(extra / test.oracle[~test.correct]).mean() * 100:.1f}% above the best)")

    # Shift experiment.
    timeline, notes = run_shift(selectors)
    timeline.to_csv(OUT / "shift_timeline.csv", index=False)
    notes.to_csv(OUT / "explanations.csv", index=False)
    plot_shift(timeline, notes)

    phase = timeline.pivot_table(index="policy", columns="workload", values="movement",
                                 aggfunc="sum")[[k for k, _ in PHASES]]
    phase["total"] = phase.sum(axis=1)
    phase["seek_time_us"] = phase.total * TIME_PER_TRACK_NS / 1000
    first = PHASES[0][0]
    for kind, _ in PHASES[1:]:
        phase[f"x_{kind}_vs_{first}"] = (phase[kind] / phase[first]).round(2)
    phase = phase.sort_values("total")
    phase.to_csv(OUT / "shift_summary.csv")
    report.append("Shift trace, total movement per phase (30 windows each):\n" + phase.to_string())

    acc = notes.groupby(["selector", "workload"]).agg(
        accuracy=("correct", "mean"), mean_confidence=("confidence", "mean")
    ).round(3)
    report.append("Selector accuracy / confidence per phase:\n" + acc.to_string())

    bonus = notes.groupby(["selector", "confidence_word"]).agg(
        windows=("correct", "size"), accuracy=("correct", "mean")).round(3)
    report.append("Bonus: explanation confidence word vs. correctness:\n" + bonus.to_string())

    text = "\n\n".join(report)
    (OUT / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
