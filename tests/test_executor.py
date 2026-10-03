import json

import pandas as pd
import pytest

from analyst.executor import build_env, check_code, run_code


@pytest.fixture
def tiny_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"month": [1, 2, 3], "sales": [100, 110, 60]}).to_csv(path, index=False)
    return path


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


# ---------------------------------------------------------------- AST check

@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "import socket",
    "import requests",
    "import shutil",
    "import sys",
    "import os.path",
    "from os import path",
    "from subprocess import run",
    "import pandas as pd, os",
])
def test_blocks_forbidden_imports(code, out_dir):
    assert check_code(code, out_dir), f"should have been rejected: {code}"


@pytest.mark.parametrize("code", [
    "eval('1+1')",
    "exec('x = 1')",
    "__import__('os')",
    "().__class__.__bases__[0].__subclasses__()",
])
def test_blocks_forbidden_calls(code, out_dir):
    assert check_code(code, out_dir)


def test_blocks_open_for_writing_outside_output_dir(out_dir):
    assert check_code("open('C:/evil.txt', 'w')", out_dir)
    assert check_code("open('relative.txt', mode='a')", out_dir)
    assert check_code("open(f'{OUTPUT_DIR}/../escape.txt', 'w')", out_dir)
    assert check_code("m = 'w'\nopen(f'{OUTPUT_DIR}/x.txt', m)", out_dir) == []  # f-string ok
    assert check_code("m = 'w'\nopen('C:/x.txt', m)", out_dir)  # unknown mode + bad path


def test_allows_normal_analysis_code(out_dir):
    code = (
        "import json\nimport pandas as pd\nfrom scipy import stats\n"
        "import matplotlib.pyplot as plt\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "with open(CSV_PATH) as f:\n    f.read()\n"            # reading is fine
        "with open(f'{OUTPUT_DIR}/x.json', 'w') as f:\n    f.write('{}')\n"
        "print(json.dumps({'n': len(df)}))\n"
    )
    assert check_code(code, out_dir) == []


def test_rejected_code_is_not_run(tiny_csv, out_dir):
    result = run_code("import os\nprint('ran')", tiny_csv, out_dir)
    assert not result.success
    assert "safety check" in result.error
    assert result.stdout == ""


# ---------------------------------------------------------------- execution

def test_returns_stdout_on_success(tiny_csv, out_dir):
    code = (
        "import json\nimport pandas as pd\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "print(json.dumps({'total_sales': int(df['sales'].sum())}))\n"
    )
    result = run_code(code, tiny_csv, out_dir)
    assert result.success, result.stderr
    assert json.loads(result.stdout) == {"total_sales": 270}


def test_saves_chart_into_output_dir(tiny_csv, out_dir):
    code = (
        "import pandas as pd\nimport matplotlib.pyplot as plt\n"
        "df = pd.read_csv(CSV_PATH)\n"
        "df.plot(x='month', y='sales')\n"
        "plt.savefig(f'{OUTPUT_DIR}/sales.png')\n"
    )
    result = run_code(code, tiny_csv, out_dir)
    assert result.success, result.stderr
    assert [p.endswith("sales.png") for p in result.new_files] == [True]


def test_enforces_timeout(tiny_csv, out_dir):
    result = run_code("while True:\n    pass", tiny_csv, out_dir, timeout_s=2)
    assert not result.success
    assert result.timed_out
    assert result.duration_s < 10


def test_runtime_error_is_reported(tiny_csv, out_dir):
    result = run_code("import pandas as pd\npd.read_csv(CSV_PATH)['missing_col']", tiny_csv, out_dir)
    assert not result.success
    assert "KeyError" in result.error


def test_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "also-secret")
    env = build_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "sk-secret" not in env.values()
