from __future__ import annotations

import json
import threading

from coscientist.coevo.store import RunStore


def test_run_store_resume_uses_max_valid_suffix_and_ignores_gaps_and_malformed(tmp_path):
    run = tmp_path / "run"
    candidates = run / "solver" / "candidates"
    probes = run / "supervisor" / "probes"
    candidates.mkdir(parents=True)
    probes.mkdir(parents=True)
    for name in (
        "cand_00000.json",
        "cand_00007.json",
        "cand_00003.json",
        "cand_bad.json",
        "cand_99999.json.tmp",
        "cand_12.json",
        "cand_100000.json",
    ):
        (candidates / name).write_text("{}", encoding="utf-8")
    for name in (
        "probe_0000.json",
        "probe_0042.json",
        "probe_0004.json",
        "probe_bad.json",
        "probe_9999.json.bak",
        "probe_7.json",
    ):
        (probes / name).write_text("{}", encoding="utf-8")

    store = RunStore(run)

    assert store.candidate({"x": 1}, {}, score=1.0) == "cand_100001"
    assert store.probe(
        description="resume", payload={"x": 1}, score=0.0,
        expected_low=True, fooled=False,
    ) == "probe_0043"
    assert json.loads((candidates / "cand_00007.json").read_text()) == {}
    assert json.loads((probes / "probe_0042.json").read_text()) == {}


def test_two_run_store_instances_reserve_unique_sequences_under_concurrency(tmp_path):
    run = tmp_path / "run"
    candidates = run / "solver" / "candidates"
    probes = run / "supervisor" / "probes"
    candidates.mkdir(parents=True)
    probes.mkdir(parents=True)
    (candidates / "cand_00011.json").write_text("{}", encoding="utf-8")
    (probes / "probe_0015.json").write_text("{}", encoding="utf-8")
    stores = (RunStore(run), RunStore(run))
    candidate_ids: list[str] = []
    probe_ids: list[str] = []
    result_lock = threading.Lock()

    def write_pair(index: int) -> None:
        store = stores[index % 2]
        candidate_id = store.candidate({"x": index}, {}, score=float(index))
        probe_id = store.probe(
            description=str(index), payload={"x": index}, score=0.0,
            expected_low=True, fooled=False,
        )
        with result_lock:
            candidate_ids.append(candidate_id)
            probe_ids.append(probe_id)

    threads = [threading.Thread(target=write_pair, args=(i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(candidate_ids) == len(set(candidate_ids)) == 40
    assert len(probe_ids) == len(set(probe_ids)) == 40
    assert min(candidate_ids) == "cand_00012"
    assert max(candidate_ids) == "cand_00051"
    assert min(probe_ids) == "probe_0016"
    assert max(probe_ids) == "probe_0055"
