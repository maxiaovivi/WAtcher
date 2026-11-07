import json
import sys

from safe_wam.cli import main


def test_bad_json_is_an_error_decision_without_loading_torch(tmp_path, monkeypatch, capsys):
    request = tmp_path / "invalid.json"
    request.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["safe-wam", str(request), "--safety-checkpoint", "missing.pt"])
    assert main() == 2
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "INVALID_INPUT"
    assert decision["approved_actions"] is None
    assert decision["robot_commands_sent"] is False


def test_output_never_overwrites_existing_record(tmp_path, monkeypatch, capsys):
    output = tmp_path / "decision.json"
    output.write_text("original experiment", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["safe-wam", str(tmp_path / "absent.json"),
                                    "--safety-checkpoint", "missing.pt", "--output", str(output)])
    assert main() == 2
    assert output.read_text(encoding="utf-8") == "original experiment"
    assert "Cannot create output file" in capsys.readouterr().err
