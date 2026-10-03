"""Bonus track: rule-based (template) explanation of each selector decision.

The explanation is built from the decision-tree path that produced the
prediction, plus one sentence about why that scheduler suits such a window
(taken from the lecture's advantages/best-use slides). The self-rated
confidence is the leaf probability, reported as a number and a word.
"""
from features import FEATURE_NAMES

READABLE = {
    "spread": "request spread",
    "mean_jump": "average jump between arrivals",
    "monotonic": "share of arrivals moving one way",
    "zones_used": "share of disk zones in use",
    "ahead_frac": "share of requests ahead of the head",
    "first_gap": "distance to the first request",
}

WHY = {
    "FCFS": "requests already arrive in near track order, so arrival order adds little movement",
    "SSTF": "requests sit in tight clusters, so the nearest request is usually in the same cluster",
    "SSTF-spread": "the queue is dense, so the nearest pending request is always a short jump away",
    "SCAN": "requests are spread on both sides, so one sweep each way covers them",
    "C-SCAN": "a one-direction sweep with a single wrap covers the requests",
}


def confidence_word(p):
    return "high" if p >= 0.8 else "medium" if p >= 0.6 else "low"


def explain(tree, x, choice, confidence):
    t = tree.tree_
    node, steps = 0, []
    while t.children_left[node] != -1:
        f, thr = t.feature[node], t.threshold[node]
        name = FEATURE_NAMES[f]
        if x[f] <= thr:
            steps.append(f"{READABLE[name]} {x[f]:.2f} <= {thr:.2f}")
            node = t.children_left[node]
        else:
            steps.append(f"{READABLE[name]} {x[f]:.2f} > {thr:.2f}")
            node = t.children_right[node]
    reason = WHY[choice]
    if choice == "SSTF" and x[FEATURE_NAMES.index("zones_used")] > 0.5:
        reason = WHY["SSTF-spread"]
    return (
        f"Chose {choice} ({confidence_word(confidence)} confidence, {confidence:.2f}) "
        f"because {'; '.join(steps)}. In class terms: {reason}."
    )
