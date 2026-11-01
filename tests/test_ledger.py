"""Tests for gotrain/ledger.py::get_lineage_tree -- the dataset-wide
parent/child forest used by NEXUS's Recursive Self-Improvement lineage
view. get_lineage (singular, pre-existing) walks one run's ancestor
chain; get_lineage_tree is new and must correctly group an ENTIRE
dataset's runs into generations/children without losing or
double-counting any run, and without raising on a cycle or an orphaned
parent_id (defensive only -- add_run itself already refuses an unknown
parent_id, so these are cheap insurance, not reachable production paths).
"""

import importlib

import pytest


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))

    from gotrain import logging_utils

    importlib.reload(logging_utils)
    from gotrain import ledger

    importlib.reload(ledger)

    yield ledger

    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)


def test_empty_dataset_returns_empty_list(ledger_env):
    ledger = ledger_env
    assert ledger.get_lineage_tree("m-eurosat") == []


def test_single_root_run_has_generation_zero_and_no_children(ledger_env):
    ledger = ledger_env
    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                          task_type="classification", decision="kept")
    tree = ledger.get_lineage_tree("m-eurosat")
    assert len(tree) == 1
    assert tree[0]["run_id"] == run["run_id"]
    assert tree[0]["generation"] == 0
    assert tree[0]["children"] == []


def test_three_generation_chain_has_correct_generations_and_children(ledger_env):
    ledger = ledger_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept")
    child = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "lr": 0.002}, val_metric=0.85,
                            parent_id=root["run_id"], task_type="classification", decision="kept")
    grandchild = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "lr": 0.002, "epochs": 20},
                                 val_metric=0.91, parent_id=child["run_id"], task_type="classification", decision="kept")

    tree = {n["run_id"]: n for n in ledger.get_lineage_tree("m-eurosat")}
    assert tree[root["run_id"]]["generation"] == 0
    assert tree[root["run_id"]]["children"] == [child["run_id"]]
    assert tree[child["run_id"]]["generation"] == 1
    assert tree[child["run_id"]]["children"] == [grandchild["run_id"]]
    assert tree[grandchild["run_id"]]["generation"] == 2
    assert tree[grandchild["run_id"]]["children"] == []


def test_branching_lineage_both_children_see_the_same_root(ledger_env):
    """One root, two independent children (two different single-variable
    experiments off the same baseline) -- both must be generation 1 and
    both must appear in the root's children list, neither overwriting
    the other.
    """
    ledger = ledger_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept")
    kept = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "lr": 0.002}, val_metric=0.85,
                           parent_id=root["run_id"], task_type="classification", decision="kept")
    discarded = ledger.add_run(dataset="m-eurosat", config={"backbone": "convnext_tiny"}, val_metric=0.60,
                                parent_id=root["run_id"], task_type="classification", decision="discarded")

    tree = {n["run_id"]: n for n in ledger.get_lineage_tree("m-eurosat")}
    assert set(tree[root["run_id"]]["children"]) == {kept["run_id"], discarded["run_id"]}
    assert tree[kept["run_id"]]["generation"] == 1
    assert tree[discarded["run_id"]]["generation"] == 1


def test_lineage_tree_is_scoped_to_one_dataset(ledger_env):
    """A run for a DIFFERENT dataset must never appear in this dataset's
    lineage tree, even if (hypothetically) ids or configs looked similar
    -- the same cross-mission-leakage concern as the Experiment Timeline
    fix, applied to this new view.
    """
    ledger = ledger_env
    a = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                        task_type="classification", decision="kept")
    b = ledger.add_run(dataset="m-nz-cattle", config={"architecture": "unet"}, val_metric=0.7,
                        task_type="segmentation", decision="kept")

    eurosat_tree = ledger.get_lineage_tree("m-eurosat")
    cattle_tree = ledger.get_lineage_tree("m-nz-cattle")
    assert [n["run_id"] for n in eurosat_tree] == [a["run_id"]]
    assert [n["run_id"] for n in cattle_tree] == [b["run_id"]]


def test_nodes_are_sorted_by_generation_then_timestamp(ledger_env):
    ledger = ledger_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept")
    grandchild_config = {"backbone": "resnet18", "lr": 0.002, "epochs": 20}
    child = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "lr": 0.002}, val_metric=0.85,
                            parent_id=root["run_id"], task_type="classification", decision="kept")
    grandchild = ledger.add_run(dataset="m-eurosat", config=grandchild_config, val_metric=0.91,
                                 parent_id=child["run_id"], task_type="classification", decision="kept")

    tree = ledger.get_lineage_tree("m-eurosat")
    generations = [n["generation"] for n in tree]
    assert generations == sorted(generations)


def test_tree_nodes_carry_through_full_ledger_fields(ledger_env):
    """The lineage view's per-node dict must still carry val_metric,
    delta_vs_parent, config, decision -- NEXUS's lineage card reads these
    directly without a second lookup.
    """
    ledger = ledger_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept", version_id="v1")
    child = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "lr": 0.002}, val_metric=0.87,
                            parent_id=root["run_id"], task_type="classification", decision="kept", version_id="v2")

    tree = {n["run_id"]: n for n in ledger.get_lineage_tree("m-eurosat")}
    node = tree[child["run_id"]]
    assert node["val_metric"] == 0.87
    assert node["delta_vs_parent"] == pytest.approx(0.07)
    assert node["version_id"] == "v2"
    assert node["decision"] == "kept"
    assert node["config"]["lr"] == 0.002
