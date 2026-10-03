"""Basic checks for the four schedulers. Run with:  py -m pytest -q
"""
from schedulers import fcfs, sstf, scan, cscan, LEFT, RIGHT

QUEUE = [98, 183, 37, 122, 14, 124, 65, 67]
HEAD = 53


def test_fcfs_sample_queue():
    s = fcfs(QUEUE, HEAD)
    assert s.order == QUEUE
    assert s.movement == 640


def test_sstf_sample_queue():
    s = sstf(QUEUE, HEAD)
    assert s.order == [65, 67, 37, 14, 98, 122, 124, 183]
    assert s.movement == 236


def test_scan_sample_queue_left():
    s = scan(QUEUE, HEAD, LEFT)
    assert s.path == [53, 37, 14, 0, 65, 67, 98, 122, 124, 183]
    assert s.movement == 236


def test_cscan_sample_queue_left():
    s = cscan(QUEUE, HEAD, LEFT)
    assert s.path == [53, 37, 14, 0, 199, 183, 124, 122, 98, 67, 65]
    assert s.movement == 386


def test_scan_toward_199():
    s = scan(QUEUE, HEAD, RIGHT)
    assert s.path == [53, 65, 67, 98, 122, 124, 183, 199, 37, 14]
    assert s.movement == (199 - 53) + (199 - 14)


def test_cscan_toward_199():
    s = cscan(QUEUE, HEAD, RIGHT)
    assert s.path == [53, 65, 67, 98, 122, 124, 183, 199, 0, 14, 37]
    assert s.movement == (199 - 53) + 199 + 37


def test_seek_time_uses_time_per_track():
    assert fcfs(QUEUE, HEAD).seek_time_ns == 640 * 2


# ---------------- corner cases ----------------

def test_empty_queue():
    for f in (fcfs, sstf, scan, cscan):
        s = f([], 50)
        assert s.movement == 0 and s.order == []


def test_head_already_on_request():
    for f in (fcfs, sstf, scan, cscan):
        assert f([50], 50).movement == 0


def test_requests_at_physical_ends():
    assert scan([0, 199], 100, LEFT).movement == 100 + 199
    assert cscan([0, 199], 100, LEFT).movement == 100 + 199
    # 199 is 99 away and 0 is 100 away, so SSTF goes right first.
    assert sstf([0, 199], 100, LEFT).order == [199, 0]
    assert sstf([0, 199], 100, LEFT).movement == 99 + 199


def test_all_requests_on_one_side_no_end_travel():
    # Nothing is pending behind the head, so SCAN/C-SCAN stop at the last request.
    assert scan([40, 30, 10], 50, LEFT).movement == 40
    assert cscan([40, 30, 10], 50, LEFT).movement == 40


def test_all_requests_behind_head():
    # SCAN still has to finish its sweep to the end before reversing.
    assert scan([60, 80], 50, LEFT).movement == 50 + 80
    assert cscan([60, 80], 50, LEFT).movement == 50 + 199 + (199 - 60)


def test_duplicate_requests_are_all_served():
    for f in (fcfs, sstf, scan, cscan):
        s = f([70, 70, 30], 50, LEFT)
        assert sorted(s.order) == [30, 70, 70]


def test_sstf_tie_keeps_current_direction():
    assert sstf([40, 60], 50, LEFT).order == [40, 60]
    assert sstf([40, 60], 50, RIGHT).order == [60, 40]
