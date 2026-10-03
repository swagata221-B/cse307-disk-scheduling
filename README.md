# CSE-307 Track 2: Disk Scheduling

FCFS, SSTF, SCAN and C-SCAN, and a small decision tree that picks one of them for a window of requests. Workloads are sequential, random and bursty. There is also a run where the pattern changes in the middle.

The simulator is input based. You type the request queue, starting head and last cylinder yourself.

## Run

```
pip install -r requirements.txt
py -m pytest -q
py run_experiments.py
py app.py
```

## Files

- `schedulers.py` — FCFS, SSTF, SCAN, C-SCAN
- `workloads.py` — request generators
- `selector.py` — the tree
- `run_experiments.py` — writes `results/`
- `app.py` — GUI

## Results

On the test windows the selector was right 85.2% of the time. Confidence was 0.87 when it was right and 0.58 when it was wrong.

After the shift FCFS gets a lot worse. SCAN and C-SCAN stay almost the same. SSTF is still the cheapest on movement. Tables and plots are in `results/`.

Used Cursor for some of the code.
