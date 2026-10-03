"""Drawing helpers for the simulator window.

Head-movement charts follow the lecture diagrams: cylinders on the x-axis and
one row per head move, so the path reads from top to bottom.
"""
from matplotlib.lines import Line2D

SERVE, END, WRAP, HEAD = "#2b6cb0", "#8a96a8", "#c53030", "#2f855a"

STYLE = {
    "serve": dict(color=SERVE, linestyle="-", lw=2),
    "end": dict(color=END, linestyle="--", lw=1.4),
    "wrap": dict(color=WRAP, linestyle=":", lw=1.8),
}

LEGEND = [
    Line2D([], [], marker="s", color=HEAD, linestyle="", label="start"),
    Line2D([], [], **STYLE["serve"], marker="o", markersize=4, label="serve request"),
    Line2D([], [], **STYLE["end"], label="travel to end"),
    Line2D([], [], **STYLE["wrap"], label="C-SCAN wrap"),
]


def segment_kinds(s, last=199, circular=False):
    """Label every head move of schedule s as 'serve', 'end' or 'wrap'.

    In C-SCAN (circular=True) the head only ever moves straight between the two
    physical ends during the return jump, so such a move is always the wrap.
    """
    kinds, nxt = [], 0
    for a, b in zip(s.path, s.path[1:]):
        if circular and {a, b} == {0, last}:
            kinds.append("wrap")
        elif nxt < len(s.order) and b == s.order[nxt]:
            kinds.append("serve")
            nxt += 1
        else:
            kinds.append("end")
    return kinds


def style_axes(ax, s, queue, head, last=199):
    ax.set_xlim(-0.03 * last, last * 1.03)
    ax.set_ylim(len(s.path) - 0.5, -0.7)
    ticks = sorted(set(queue + [0, head, last]))
    if len(ticks) <= 25:
        ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=7, rotation=90, top=True, labeltop=True,
                   bottom=False, labelbottom=False)
    if len(s.path) <= 30:
        ax.set_yticks(range(len(s.path)))
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("step", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    for side in ("right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.plot(head, 0, "s", color=HEAD, ms=7)


def draw_upto(ax, name, s, kinds, k):
    """Draw the first k head moves of schedule s; return the movement so far."""
    moved = 0
    for i in range(min(k, len(s.path) - 1)):
        a, b = s.path[i], s.path[i + 1]
        ax.plot([a, b], [i, i + 1], **STYLE[kinds[i]])
        ax.plot(b, i + 1, "o" if kinds[i] == "serve" else "x", color="#1a202c", ms=4)
        moved += abs(b - a)
    if 0 < k < len(s.path):
        ax.plot(s.path[k], k, "o", ms=9, mfc="none", mec=HEAD, mew=2)
    done = k >= len(s.path) - 1
    ax.set_title(name, fontsize=10, pad=26, fontweight="bold" if done else "normal")
    return moved


def draw_disk_strip(ax, s, kinds, k, last=199):
    """A horizontal strip of the disk: pending requests, served requests and the arm."""
    served_count = sum(1 for kind in kinds[:k] if kind == "serve")
    served, pending = s.order[:served_count], s.order[served_count:]
    pos = s.path[min(k, len(s.path) - 1)]

    ax.set_xlim(-0.03 * last, last * 1.03)
    ax.set_ylim(-1, 1.6)
    ax.hlines(0, 0, last, color="#d5dde8", lw=10, zorder=1)
    ax.scatter(pending, [0] * len(pending), marker="|", s=420, color=SERVE, zorder=2)
    ax.scatter(served, [0] * len(served), marker="|", s=420, color="#b8c2cf", zorder=2)
    ax.plot([pos, pos], [0, 0.9], color=HEAD, lw=2, zorder=3)
    ax.plot(pos, 0.9, "v", color=HEAD, ms=11, zorder=4)
    ax.text(pos, 1.25, f"head {pos}", ha="center", fontsize=8, color=HEAD, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([0, last])
    ax.tick_params(axis="x", labelsize=7)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Disk (cylinders 0-{last})   blue = pending, grey = served",
                 fontsize=8, loc="left", color="#4a5568")
