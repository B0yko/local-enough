from local_enough.bench.rundir import PREDICTIONS, RunDir


def test_records_append_as_members_and_resume_keys(tmp_path):
    run = RunDir(tmp_path / "run")
    run.append_records(PREDICTIONS, [{"model_id": "m", "task": "t", "split": "calib", "item_id": "1", "pass": "A"}])
    run.append_records(PREDICTIONS, [{"model_id": "m", "task": "t", "split": "calib", "item_id": 2, "pass": "B"}])
    assert [r["item_id"] for r in run.iter_records(PREDICTIONS)] == ["1", 2]
    assert len(run.predictions()) == 1
    assert run.completed_keys() == {("m", "t", "calib", "1", "A"), ("m", "t", "calib", "2", "B")}


def test_gzip_output_is_reproducible(tmp_path):
    a, b = RunDir(tmp_path / "a"), RunDir(tmp_path / "b")
    for run in (a, b):
        run.append_records("x.jsonl.gz", [{"k": 1, "a": "é"}])
    assert a.file("x.jsonl.gz").read_bytes() == b.file("x.jsonl.gz").read_bytes()


def test_json_merge(tmp_path):
    run = RunDir(tmp_path)
    run.write_json("env.json", {"a": {"x": 1}, "b": 2})
    assert run.merge_json("env.json", {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}, "b": 2}
    assert run.read_json("missing.json", {}) == {}
