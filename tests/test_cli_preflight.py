"""``clover preflight`` -- fails a bad config before any data loads (SPEC §9)."""

from __future__ import annotations

import yaml

from clover.cli import main


def _write_config(tmp_path, stream_extra=None, method_name="simplecil"):
    stream = {"dataset": "synthetic", "init_cls": 5, "increment": 5}
    stream.update(stream_extra or {})
    config = {
        "run": {"output_dir": str(tmp_path / "runs")},
        "stream": stream,
        "method": {"name": method_name},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return str(path)


def test_preflight_passes_on_valid_config(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    exit_code = main(["preflight", config_path])
    assert exit_code == 0
    assert "preflight OK" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()


def test_preflight_fails_on_typoed_key_naming_it(tmp_path, capsys):
    config_path = _write_config(tmp_path, stream_extra={"incrament": 5})
    exit_code = main(["preflight", config_path])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "incrament" in err
    assert not (tmp_path / "runs").exists()


def test_preflight_fails_on_unknown_scenario_name(tmp_path, capsys):
    config_path = _write_config(tmp_path, stream_extra={"scenario": "exact_repaly"})
    exit_code = main(["preflight", config_path])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "unknown scenario" in err
    assert not (tmp_path / "runs").exists()


def test_preflight_fails_on_unknown_method_name(tmp_path, capsys):
    config_path = _write_config(tmp_path, method_name="l2p_not_built_yet")
    exit_code = main(["preflight", config_path])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "unknown method" in err
    assert not (tmp_path / "runs").exists()


def test_preflight_fails_on_infeasible_revisit_placement(tmp_path, capsys):
    config_path = _write_config(
        tmp_path,
        stream_extra={
            "revisits": [
                {
                    "classes": "task0",
                    "placement": "end_of_stream",
                    "min_gap": 100,
                }
            ]
        },
    )
    exit_code = main(["preflight", config_path])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Cannot place" in err
