"""``clover smoke`` -- runs every registered method (SPEC §9)."""

from __future__ import annotations

from clover.cli import main


def test_smoke_runs_every_registered_method_successfully(capsys):
    exit_code = main(["smoke"])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "simplecil OK" in out
    assert "l2p OK" in out
    assert "dualprompt OK" in out
    assert "coda_prompt OK" in out


def test_smoke_reports_failure_and_nonzero_exit_for_a_broken_method(monkeypatch, capsys):
    import clover.cli as cli_module

    def _broken_check(method_name):
        raise RuntimeError(f"{method_name} exploded")

    monkeypatch.setattr(cli_module, "_smoke_check_method", _broken_check)

    exit_code = main(["smoke"])
    assert exit_code == 1

    # smoke stops at the first failure -- "coda_prompt" sorts first alphabetically.
    err = capsys.readouterr().err
    assert "coda_prompt FAILED" in err
    assert "exploded" in err
