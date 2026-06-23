from sparrowml.cli import main

def test_cli_smoke(capsys):
    assert main(["validate-contracts"]) == 0
    assert "contracts: valid" in capsys.readouterr().out
