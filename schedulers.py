"""Classical disk-scheduling algorithms: FCFS, SSTF, SCAN and C-SCAN.

Conventions follow the CSE-307 lecture slides:
- cylinders are numbered 0 .. disk_size-1 (0-199 in class)
- "left" means moving toward cylinder 0, "right" toward cylinder disk_size-1
- SCAN goes to the physical end before reversing
- C-SCAN goes to the physical end, jumps to the other end without serving,
  and the jump is counted as head movement (0 -> 199 = 199 cylinders)
- head movement is counted only until the last pending request is served
"""
from dataclasses import dataclass

LEFT, RIGHT = "left", "right"
TIME_PER_TRACK_NS = 2


@dataclass
class Schedule:
    order: list      # requests in the order they are served
    path: list       # every head position visited, including start, ends and wraps
    movement: int    # total head movement in cylinders
    direction: str   # direction of the head after the last move

    @property
    def seek_time_ns(self):
        return self.movement * TIME_PER_TRACK_NS


def _opposite(direction):
    return RIGHT if direction == LEFT else LEFT


def _make(order, path, direction):
    movement = 0
    for a, b in zip(path, path[1:]):
        movement += abs(b - a)
        if b != a:
            direction = LEFT if b < a else RIGHT
    return Schedule(list(order), list(path), movement, direction)


def fcfs(requests, head, direction=LEFT, disk_size=200):
    """Serve requests exactly in arrival order."""
    return _make(requests, [head] + list(requests), direction)


def sstf(requests, head, direction=LEFT, disk_size=200):
    """Always serve the pending request closest to the head.

    Tie-break: if two requests are equally close, keep moving in the
    current direction (the request on the side the head is moving toward).
    """
    pending = list(requests)
    order, pos = [], head
    while pending:
        def key(r):
            behind = (r > pos) if direction == LEFT else (r < pos)
            return (abs(r - pos), behind)
        nxt = min(pending, key=key)
        pending.remove(nxt)
        if nxt != pos:
            direction = LEFT if nxt < pos else RIGHT
        order.append(nxt)
        pos = nxt
    return _make(order, [head] + order, direction)


def scan(requests, head, direction=LEFT, disk_size=200):
    """Elevator: serve everything in the current direction, touch the
    physical end, then reverse and serve the rest."""
    if direction == LEFT:
        first = sorted((r for r in requests if r <= head), reverse=True)
        second = sorted(r for r in requests if r > head)
        end = 0
    else:
        first = sorted(r for r in requests if r >= head)
        second = sorted((r for r in requests if r < head), reverse=True)
        end = disk_size - 1

    path = [head] + first
    if second:
        path += [end] + second
    return _make(first + second, path, direction)


def cscan(requests, head, direction=LEFT, disk_size=200):
    """Circular SCAN: serve in one direction only. At the physical end,
    jump to the opposite end (no service during the jump) and continue."""
    if direction == LEFT:
        first = sorted((r for r in requests if r <= head), reverse=True)
        second = sorted((r for r in requests if r > head), reverse=True)
        ends = [0, disk_size - 1]
    else:
        first = sorted(r for r in requests if r >= head)
        second = sorted(r for r in requests if r < head)
        ends = [disk_size - 1, 0]

    path = [head] + first
    if second:
        path += ends + second
    result = _make(first + second, path, direction)
    # After the wrap the head keeps its original service direction.
    result.direction = direction
    return result


SCHEDULERS = {"FCFS": fcfs, "SSTF": sstf, "SCAN": scan, "C-SCAN": cscan}
