"""Disk scheduling simulator (CSE-307).

Choose an algorithm, enter the request queue and the starting head, press Run
and watch the read/write head move. The step table and the final answer use
the same format as the lecture slides.

    py app.py
"""
import random
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from schedulers import SCHEDULERS, LEFT, RIGHT, TIME_PER_TRACK_NS
from visualize import LEGEND, segment_kinds, style_axes, draw_upto, draw_disk_strip

ALL = "Compare all four"
NAVY, ACCENT, BG, CARD, MUTED, GOOD = "#1f3a5f", "#2b6cb0", "#eef2f7", "#ffffff", "#5a6b7f", "#e6f4ea"
FONT = "Segoe UI"
NOTE = {"serve": "", "end": "physical end", "wrap": "wrap (no service)"}


# ---------------- input checking ----------------

def parse_int(text, what):
    text = text.strip()
    if not text:
        raise ValueError(f"{what} is empty.")
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{what} must be a whole number, got '{text}'.") from None


def parse_inputs(queue_text, head_text, last_text):
    last = parse_int(last_text, "Last cylinder")
    if last < 1:
        raise ValueError("Last cylinder must be at least 1 (the disk needs 2 or more cylinders).")

    tokens = [t for t in re.split(r"[,\s]+", queue_text.strip()) if t]
    if not tokens:
        raise ValueError("The request queue is empty. Enter at least one cylinder number.")
    queue = [parse_int(t, f"Request '{t}'") for t in tokens]
    bad = [r for r in queue if not 0 <= r <= last]
    if bad:
        raise ValueError(f"Requests must be between 0 and {last}. Out of range: {bad}")

    head = parse_int(head_text, "Starting head")
    if not 0 <= head <= last:
        raise ValueError(f"Starting head must be between 0 and {last}, got {head}.")
    return queue, head, last


# ---------------- answer text ----------------

def trace_rows(s, kinds):
    """(move index, from, to, distance, note) for every head move worth listing."""
    rows = []
    for i, ((a, b), kind) in enumerate(zip(zip(s.path, s.path[1:]), kinds)):
        if kind != "serve" and a == b:
            continue
        rows.append((i, a, b, abs(b - a), NOTE[kind]))
    return rows


def sstf_ties(queue, head, order):
    notes, pos, pending = [], head, list(queue)
    for step, nxt in enumerate(order, 1):
        d = abs(nxt - pos)
        rivals = [r for r in pending if r != nxt and abs(r - pos) == d]
        if rivals and d > 0:
            way = "left" if nxt < pos else "right"
            notes.append(f"Step {step}: tie between {nxt} and {rivals[0]} (both {d} away). "
                         f"Tie-break keeps the current direction, so the head went {way} to {nxt}.")
        pending.remove(nxt)
        pos = nxt
    return notes


def queue_notes(queue, head):
    notes = []
    if head in queue:
        notes.append(f"The head is already on cylinder {head}; that request is served with 0 movement.")
    dups = sorted({r for r in queue if queue.count(r) > 1})
    if dups:
        notes.append(f"Cylinder(s) {dups} requested more than once; repeats cost 0 extra movement.")
    return notes


def algorithm_notes(name, queue, head, direction, last, s, kinds):
    notes = []
    if name == "FCFS":
        notes.append("Direction does not matter: requests are served in arrival order.")
    if name == "SSTF":
        notes += sstf_ties(queue, head, s.order)
    if name in ("SCAN", "C-SCAN"):
        ahead = [r for r in queue if (r <= head if direction == LEFT else r >= head)]
        behind = [r for r in queue if r not in ahead]
        end = 0 if direction == LEFT else last
        if not behind:
            notes.append("No requests behind the head, so it stops at the last request "
                         f"and does not travel to the physical end ({end}).")
        elif not ahead:
            notes.append(f"No requests in the {direction} direction, but {name} still sweeps "
                         f"to the physical end ({end}) first.")
    if "wrap" in kinds:
        notes.append(f"The return jump between 0 and {last} is counted as {last} cylinders; "
                     "nothing is served during the jump.")
    return notes


def answer_text(name, s, kinds, queue, head, direction, last):
    rows = trace_rows(s, kinds)
    lines = [f"{name}", f"Head = {head}, direction = {direction}, cylinders 0-{last}",
             f"Service order: {', '.join(map(str, s.order))}",
             f"Head path: {' → '.join(map(str, s.path))}", ""]
    deltas = [d for _, _, _, d, _ in rows]
    total = " + ".join(map(str, deltas)) + " = " if len(deltas) > 1 else ""
    lines += [f"Total head movement = {total}{s.movement} cylinders",
              f"Seek time = {s.movement} × {TIME_PER_TRACK_NS} ns = {s.seek_time_ns} ns"]
    notes = queue_notes(queue, head) + algorithm_notes(name, queue, head, direction, last, s, kinds)
    if notes:
        lines += ["", "Notes"] + [f"  • {n}" for n in notes]
    return "\n".join(lines)


# ---------------- the window ----------------

class Simulator:
    def __init__(self, root):
        self.root = root
        self.selector = None
        self.job = None
        self.after_id = None
        self.playing = False

        root.title("Disk Scheduling Simulator - CSE-307")
        root.geometry("1360x860")
        root.minsize(1000, 640)
        root.state("zoomed")
        root.configure(bg=BG)
        self._styles()

        self._header()
        self.status = tk.Label(root, text="Training the learned selector...", anchor="w",
                               bg="#dfe6ef", fg=MUTED, font=(FONT, 9), padx=10, pady=4)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)
        body = tk.Frame(root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 6))
        self._sidebar(body)
        self._main_area(body)

        root.bind("<Return>", lambda e: self.run())
        root.bind("<space>", self._space)
        threading.Thread(target=self._train_selector, daemon=True).start()
        self.root.after(300, self._wait_for_selector)
        self._welcome()

    # ----- layout -----

    def _styles(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background=BG, font=(FONT, 10))
        st.configure("TLabelframe", background=BG, bordercolor="#c9d3df")
        st.configure("TLabelframe.Label", background=BG, foreground=NAVY, font=(FONT, 10, "bold"))
        st.configure("TRadiobutton", background=BG)
        st.configure("Hint.TLabel", foreground=MUTED, font=(FONT, 8))
        st.configure("Run.TButton", background=ACCENT, foreground="white",
                     font=(FONT, 10, "bold"), padding=(14, 6), borderwidth=0)
        st.map("Run.TButton", background=[("disabled", "#9db4cf"), ("active", "#245a93")])
        st.configure("TButton", padding=(8, 4))
        st.configure("Treeview", rowheight=22, font=(FONT, 9), fieldbackground=CARD)
        st.configure("Treeview.Heading", font=(FONT, 9, "bold"), foreground=NAVY)
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(12, 4), font=(FONT, 9, "bold"))

    def _header(self):
        bar = tk.Frame(self.root, bg=NAVY, padx=16, pady=10)
        bar.pack(fill=tk.X)
        tk.Label(bar, text="Disk Scheduling Simulator", bg=NAVY, fg="white",
                 font=(FONT, 17, "bold")).pack(side=tk.LEFT)

    def _sidebar(self, body):
        side = tk.Frame(body, bg=BG, width=320)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        box = ttk.LabelFrame(side, text="1  Algorithm", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.algo = tk.StringVar(value="FCFS")
        for name in list(SCHEDULERS) + [ALL]:
            ttk.Radiobutton(box, text=name, value=name, variable=self.algo).pack(anchor="w")

        box = ttk.LabelFrame(side, text="2  Requests and head", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(box, text="Request queue").pack(anchor="w")
        self.queue = ttk.Entry(box, width=36)
        self.queue.pack(fill=tk.X, pady=(0, 6))
        row = tk.Frame(box, bg=BG)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Starting head").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Last cylinder").grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.head = ttk.Entry(row, width=10)
        self.head.grid(row=1, column=0, sticky="w")
        self.last = ttk.Entry(row, width=10)
        self.last.grid(row=1, column=1, sticky="w", padx=(12, 0))

        self.dir_box = tk.Frame(box, bg=BG)
        ttk.Label(self.dir_box, text="Direction").pack(anchor="w", pady=(8, 0))
        self.direction = tk.StringVar(value=LEFT)
        ttk.Radiobutton(self.dir_box, text="Left, toward 0", value=LEFT,
                        variable=self.direction).pack(anchor="w")
        ttk.Radiobutton(self.dir_box, text="Right, toward the last cylinder", value=RIGHT,
                        variable=self.direction).pack(anchor="w")
        self.dir_hint = ttk.Label(self.dir_box, style="Hint.TLabel", wraplength=280)
        self.dir_hint.pack(anchor="w")
        self.algo.trace_add("write", lambda *_: self._update_direction())
        self._update_direction()

        self.example_row = tk.Frame(box, bg=BG)
        self.example_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(self.example_row, text="Random queue", command=self.random_queue).pack(side=tk.LEFT)

        box = ttk.LabelFrame(side, text="3  Simulation", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.run_btn = ttk.Button(box, text="Run simulation", style="Run.TButton", command=self.run)
        self.run_btn.pack(fill=tk.X)
        row = tk.Frame(box, bg=BG)
        row.pack(fill=tk.X, pady=(6, 0))
        self.pause_btn = ttk.Button(row, text="Pause", command=self.pause_resume, state="disabled")
        self.pause_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.step_btn = ttk.Button(row, text="Step", command=self.step_once, state="disabled")
        self.step_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(row, text="Reset", command=self.reset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Label(box, text="Speed").pack(anchor="w", pady=(8, 0))
        self.speed = tk.DoubleVar(value=700)
        ttk.Scale(box, from_=1500, to=120, variable=self.speed, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(box, text="slow  ←→  fast      Enter = run, Space = pause",
                  style="Hint.TLabel").pack(anchor="w")

    def _main_area(self, body):
        main = tk.Frame(body, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cards_frame = tk.Frame(main, bg=BG)
        self.cards_frame.pack(fill=tk.X)
        self.cards = {}

        # The tabs are packed before the chart so that, on a small screen,
        # the chart shrinks instead of the tabs being pushed off the window.
        self.tabs = ttk.Notebook(main, height=165)
        self.tabs.pack(side=tk.BOTTOM, fill=tk.X)

        chart = tk.Frame(main, bg=CARD, highlightbackground="#c9d3df", highlightthickness=1)
        chart.pack(fill=tk.BOTH, expand=True, pady=8)
        self.fig = Figure(figsize=(8, 4), facecolor=CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        frame = tk.Frame(self.tabs, bg=CARD)
        cols = ("algo", "step", "move", "dist", "total", "note")
        self.trace = ttk.Treeview(frame, columns=cols, show="headings", height=7)
        for col, title, width in zip(cols, ("Algorithm", "Step", "R/W head movement", "Δ",
                                            "Running total", "Note"), (90, 50, 170, 60, 100, 160)):
            self.trace.heading(col, text=title)
            self.trace.column(col, width=width, anchor="center")
        scroll = ttk.Scrollbar(frame, command=self.trace.yview)
        self.trace.configure(yscrollcommand=scroll.set)
        self.trace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tabs.add(frame, text="Step trace")

        frame = tk.Frame(self.tabs, bg=CARD)
        self.answer = tk.Text(frame, wrap="word", font=("Consolas", 10), relief="flat",
                              padx=10, pady=8, bg=CARD)
        scroll = ttk.Scrollbar(frame, command=self.answer.yview)
        self.answer.configure(yscrollcommand=scroll.set, state="disabled")
        self.answer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tabs.add(frame, text="Answer")

        frame = tk.Frame(self.tabs, bg=CARD)
        cols = ("rank", "algo", "moves", "seek")
        self.compare = ttk.Treeview(frame, columns=cols, show="headings", height=4)
        for col, title, width in zip(cols, ("Rank", "Algorithm", "Head movement (cylinders)",
                                            "Seek time (ns)"), (50, 90, 170, 110)):
            self.compare.heading(col, text=title)
            self.compare.column(col, width=width, anchor="center")
        self.compare.tag_configure("best", background=GOOD)
        self.compare.pack(side=tk.LEFT, anchor="n", padx=8, pady=8)
        self.selector_note = tk.Label(frame, text="", bg=CARD, fg="#2d3748", justify="left",
                                      anchor="nw", wraplength=420, font=(FONT, 9))
        self.selector_note.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 10), pady=8)
        self.tabs.add(frame, text="Comparison")

    def _set_cards(self, values):
        if list(values) != list(self.cards):
            for child in self.cards_frame.winfo_children():
                child.destroy()
            self.cards = {}
            for title in values:
                card = tk.Frame(self.cards_frame, bg=CARD, highlightbackground="#c9d3df",
                                highlightthickness=1, padx=12, pady=6)
                card.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
                tk.Label(card, text=title.upper(), bg=CARD, fg=MUTED, font=(FONT, 8, "bold")).pack(anchor="w")
                value = tk.Label(card, bg=CARD, fg=NAVY, font=(FONT, 15, "bold"))
                value.pack(anchor="w")
                self.cards[title] = value
        for title, text in values.items():
            self.cards[title].config(text=text)

    def _welcome(self):
        self._set_cards({"Algorithm": "-", "Head at": "-", "Head movement": "-",
                         "Requests served": "-", "Seek time": "-"})
        self.fig.clear()
        self.fig.text(0.5, 0.55, "Choose an algorithm, enter the request queue and the starting head,",
                      ha="center", color=MUTED, fontsize=11)
        self.fig.text(0.5, 0.48, "then press Run simulation.", ha="center", color=MUTED, fontsize=11)
        self.canvas.draw()

    # ----- learned selector -----

    def _train_selector(self):
        from selector import Selector, make_dataset
        from workloads import GENERATORS
        X, y, _, _ = make_dataset(np.random.default_rng(307), GENERATORS, 400)
        self.selector = Selector().fit(X, y)

    def _wait_for_selector(self):
        # Tk widgets may only be touched from the main thread, so poll here.
        if self.selector is None:
            self.root.after(300, self._wait_for_selector)
        elif self.job is None:
            self.status.config(text="Ready.")

    # ----- buttons -----

    def _update_direction(self):
        hints = {
            "SSTF": "Used only if two requests are equally close.",
            "SCAN": "SCAN sweeps this way first, then reverses at the end.",
            "C-SCAN": "C-SCAN serves only this way and wraps at the end.",
            ALL: "Used by SCAN and C-SCAN. SSTF uses it only to break a tie. FCFS ignores it.",
        }
        algo = self.algo.get()
        if algo == "FCFS":
            self.dir_box.pack_forget()
            return
        self.dir_hint.config(text=hints[algo])
        self.dir_box.pack(fill=tk.X, before=self.example_row)

    def _space(self, event):
        if not isinstance(event.widget, tk.Entry):   # typing a space in a box is not "pause"
            self.pause_resume()

    def _fill(self, queue, head, last, direction):
        for entry, value in ((self.queue, queue), (self.head, head), (self.last, last)):
            entry.delete(0, tk.END)
            entry.insert(0, value)
        self.direction.set(direction)

    def random_queue(self):
        try:
            last = parse_int(self.last.get(), "Last cylinder")
        except ValueError as e:
            messagebox.showerror("Check your input", str(e))
            return
        last = max(last, 1)
        queue = [random.randint(0, last) for _ in range(random.randint(6, 10))]
        self._fill(", ".join(map(str, queue)), str(random.randint(0, last)), str(last),
                   random.choice([LEFT, RIGHT]))

    def run(self):
        try:
            queue, head, last = parse_inputs(self.queue.get(), self.head.get(), self.last.get())
        except ValueError as e:
            messagebox.showerror("Check your input", str(e))
            return
        self._cancel()
        direction = self.direction.get()
        results = {n: f(queue, head, direction, last + 1) for n, f in SCHEDULERS.items()}
        kinds = {n: segment_kinds(s, last, n == "C-SCAN") for n, s in results.items()}
        shown = list(SCHEDULERS) if self.algo.get() == ALL else [self.algo.get()]
        self.job = dict(queue=queue, head=head, last=last, direction=direction,
                        results=results, kinds=kinds, shown=shown,
                        rows={n: trace_rows(results[n], kinds[n]) for n in shown},
                        listed={n: 0 for n in shown}, totals={n: 0 for n in shown})
        self.frames = max(len(results[n].path) for n in shown)
        self.k = 0

        self.trace.delete(*self.trace.get_children())
        self.compare.delete(*self.compare.get_children())
        self.selector_note.config(text="")
        self._set_answer("The answer appears here when the simulation finishes.")
        self.tabs.select(0)
        self.playing = True
        self.pause_btn.config(state="normal", text="Pause")
        self.step_btn.config(state="normal")
        self.status.config(text=f"Simulating {', '.join(shown)}...")
        self._tick()

    def pause_resume(self):
        if self.job is None or self.k >= self.frames - 1:
            return
        self.playing = not self.playing
        self.pause_btn.config(text="Pause" if self.playing else "Resume")
        if self.playing:
            self._tick()
        else:
            self._cancel()
            self.status.config(text="Paused. Press Step to move one request at a time.")

    def step_once(self):
        if self.job is None or self.k >= self.frames - 1:
            return
        self._cancel()
        self.playing = False
        self.pause_btn.config(text="Resume")
        self.k += 1
        self._render()
        if self.k >= self.frames - 1:
            self._finish()

    def reset(self):
        self._cancel()
        self.job = None
        self.playing = False
        self.queue.delete(0, tk.END)
        self.head.delete(0, tk.END)
        self.trace.delete(*self.trace.get_children())
        self.compare.delete(*self.compare.get_children())
        self.selector_note.config(text="")
        self._set_answer("")
        self.pause_btn.config(state="disabled", text="Pause")
        self.step_btn.config(state="disabled")
        self.status.config(text="Ready.")
        self._welcome()

    # ----- animation -----

    def _cancel(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _tick(self):
        self.after_id = None
        self._render()
        if self.k >= self.frames - 1:
            self._finish()
        elif self.playing:
            self.after_id = self.root.after(int(self.speed.get()), self._advance)

    def _advance(self):
        self.k += 1
        self._tick()

    def _render(self):
        j, k = self.job, self.k
        self.fig.clear()
        if len(j["shown"]) == 1:
            name = j["shown"][0]
            s, kinds = j["results"][name], j["kinds"][name]
            grid = self.fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.9)
            draw_disk_strip(self.fig.add_subplot(grid[0]), s, kinds, k, j["last"])
            ax = self.fig.add_subplot(grid[1])
            style_axes(ax, s, j["queue"], j["head"], j["last"])
            moved = draw_upto(ax, name, s, kinds, k)
            served = sum(1 for kind in kinds[:k] if kind == "serve")
            pos = s.path[min(k, len(s.path) - 1)]
            self._set_cards({"Algorithm": name, "Head at": pos, "Head movement": f"{moved} cyl",
                             "Requests served": f"{served} / {len(s.order)}",
                             "Seek time": f"{moved * TIME_PER_TRACK_NS} ns"})
        else:
            moved = {}
            for i, name in enumerate(j["shown"]):
                ax = self.fig.add_subplot(2, 2, i + 1)
                s = j["results"][name]
                style_axes(ax, s, j["queue"], j["head"], j["last"])
                moved[name] = draw_upto(ax, name, s, j["kinds"][name], k)
            self._set_cards({name: f"{m} cyl" for name, m in moved.items()})
        self.fig.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=8, frameon=False)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.canvas.draw_idle()
        self._list_moves()

    def _list_moves(self):
        j = self.job
        for name in j["shown"]:
            rows = j["rows"][name]
            while j["listed"][name] < len(rows) and rows[j["listed"][name]][0] < self.k:
                _, a, b, d, note = rows[j["listed"][name]]
                j["listed"][name] += 1
                j["totals"][name] += d
                item = self.trace.insert("", tk.END, values=(name, j["listed"][name], f"{a} → {b}",
                                                             d, j["totals"][name], note))
                self.trace.see(item)

    def _finish(self):
        self._cancel()
        self.playing = False
        self.pause_btn.config(state="disabled", text="Pause")
        self.step_btn.config(state="disabled")
        j = self.job
        texts = [answer_text(n, j["results"][n], j["kinds"][n], j["queue"], j["head"],
                             j["direction"], j["last"]) for n in j["shown"]]
        self._set_answer(("\n\n" + "─" * 60 + "\n\n").join(texts))
        self._fill_comparison()
        self.tabs.select(2 if len(j["shown"]) > 1 else 1)
        self.status.config(text="Done. Change the inputs and run again, or open the other tabs.")

    def _fill_comparison(self):
        j = self.job
        ranked = sorted(j["results"].items(), key=lambda item: item[1].movement)
        best = ranked[0][1].movement
        for rank, (name, s) in enumerate(ranked, 1):
            self.compare.insert("", tk.END, values=(rank, name, s.movement, s.seek_time_ns),
                                tags=("best",) if s.movement == best else ())
        winners = [n for n, s in j["results"].items() if s.movement == best]

        if self.selector is None:
            text = "Learned selector: still training, run again in a few seconds."
        elif len(j["queue"]) < 2:
            text = "Learned selector: needs at least two requests."
        else:
            from explain import explain
            choice, conf, x = self.selector.choose(j["queue"], j["head"], j["direction"])
            verdict = "a lowest-movement choice" if choice in winners else "not the best for this queue"
            text = (f"Learned selector picks {choice} ({verdict}).\n"
                    f"{explain(self.selector.tree, x, choice, conf)}")
            if len(j["queue"]) < 10 or j["last"] != 199:
                text += ("\nNote: it was trained on 20-request windows on a 0-199 disk, "
                         "so this input is outside its training data.")
        self.selector_note.config(text=text)

    def _set_answer(self, text):
        self.answer.configure(state="normal")
        self.answer.delete("1.0", tk.END)
        self.answer.insert(tk.END, text)
        self.answer.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    Simulator(root)
    root.mainloop()
