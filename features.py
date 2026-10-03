"""Features describing one window of pending requests.

Only information the OS has *before* serving the window is used:
the pending requests (in arrival order), the head position and its direction.
All features are scaled to roughly 0..1.
"""
import numpy as np

from schedulers import LEFT

FEATURE_NAMES = [
    "spread",       # std of cylinder numbers / disk size
    "mean_jump",    # mean |gap| between consecutive arrivals / disk size
    "monotonic",    # fraction of consecutive arrivals moving the same way
    "zones_used",   # fraction of the 10 equal disk zones that hold a request
    "ahead_frac",   # fraction of requests on the side the head is moving toward
    "first_gap",    # |head - first request| / disk size
]


def extract(requests, head, direction, disk_size=200):
    r = np.asarray(requests, dtype=float)
    jumps = np.diff(r) if len(r) > 1 else np.zeros(1)
    ahead = (r <= head) if direction == LEFT else (r >= head)
    return [
        r.std() / disk_size,
        np.abs(jumps).mean() / disk_size,
        max((jumps > 0).mean(), (jumps < 0).mean()),
        len(np.unique(r * 10 // disk_size)) / 10,
        ahead.mean(),
        abs(head - r[0]) / disk_size,
    ]
