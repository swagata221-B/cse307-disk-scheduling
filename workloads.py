"""Synthetic disk-request workloads (cylinder numbers 0 .. DISK_SIZE-1).

Every generator returns one window of WINDOW requests in arrival order.
"""
import numpy as np

DISK_SIZE = 200
WINDOW = 20


def sequential(rng, n=WINDOW, disk_size=DISK_SIZE):
    """A file read block after block: small steps in one direction."""
    steps = rng.integers(1, 6, size=n)
    start = rng.integers(0, disk_size - steps.sum())
    reqs = start + np.cumsum(steps)
    if rng.random() < 0.3:                      # some files are read backward
        reqs = disk_size - 1 - reqs
    return reqs.tolist()


def random_uniform(rng, n=WINDOW, disk_size=DISK_SIZE):
    """Independent requests spread over the whole disk."""
    return rng.integers(0, disk_size, size=n).tolist()


def bursty(rng, n=WINDOW, disk_size=DISK_SIZE):
    """2-3 hot regions of the disk; requests from different regions arrive
    interleaved, as when a few processes hammer their own files at once."""
    centers = rng.integers(10, disk_size - 10, size=rng.integers(2, 4))
    picks = rng.choice(centers, size=n)
    reqs = np.rint(picks + rng.normal(0, 4, size=n)).astype(int)
    return np.clip(reqs, 0, disk_size - 1).tolist()


GENERATORS = {"sequential": sequential, "random": random_uniform, "bursty": bursty}
