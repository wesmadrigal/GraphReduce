import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import relbench_dataset_utils as dataset_utils  # noqa: E402


def test_dataset_db_recovers_from_stale_registry_hash(monkeypatch):
    calls = []
    fallback_downloads = []

    class FakeDataset:
        def get_db(self, upto_test_timestamp):
            return {"upto_test_timestamp": upto_test_timestamp}

    def fake_get_dataset(name, download=True):
        calls.append((name, download))
        if download:
            raise ValueError(
                "SHA256 hash of downloaded file (db.zip) does not match "
                "the known hash"
            )
        return FakeDataset()

    monkeypatch.setattr(dataset_utils, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        dataset_utils,
        "_download_relbench_archive_without_stale_hash",
        lambda name: fallback_downloads.append(name),
    )

    dataset, db = dataset_utils.get_relbench_dataset_db(
        "rel-stack",
        download=True,
        upto_test_timestamp=True,
    )

    assert isinstance(dataset, FakeDataset)
    assert db == {"upto_test_timestamp": True}
    assert calls == [("rel-stack", True), ("rel-stack", False)]
    assert fallback_downloads == ["rel-stack/db.zip"]


def test_dataset_db_does_not_hide_unrelated_value_errors(monkeypatch):
    def fake_get_dataset(name, download=True):
        raise ValueError("dataset metadata is invalid")

    monkeypatch.setattr(dataset_utils, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(
        dataset_utils,
        "_download_relbench_archive_without_stale_hash",
        lambda name: pytest.fail("unexpected stale-hash fallback"),
    )

    with pytest.raises(ValueError, match="dataset metadata is invalid"):
        dataset_utils.get_relbench_dataset_db("rel-stack")


def test_task_recovers_from_stale_registry_hash(monkeypatch):
    calls = []
    fallback_downloads = []
    task = object()

    def fake_get_task(dataset_name, task_name, download=False):
        calls.append((dataset_name, task_name, download))
        if download:
            raise ValueError(
                "SHA256 hash of downloaded file (user-badge.zip) does not match "
                "the known hash"
            )
        return task

    monkeypatch.setattr(dataset_utils, "get_task", fake_get_task)
    monkeypatch.setattr(
        dataset_utils,
        "_download_relbench_archive_without_stale_hash",
        lambda resource: fallback_downloads.append(resource),
    )

    assert dataset_utils.get_relbench_task("rel-stack", "user-badge") is task
    assert calls == [
        ("rel-stack", "user-badge", True),
        ("rel-stack", "user-badge", False),
    ]
    assert fallback_downloads == ["rel-stack/tasks/user-badge.zip"]
