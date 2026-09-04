import py_compile
from pathlib import Path


def test_stage02_screening_script_compiles():
    script = Path(__file__).resolve().parents[1] / "src" / "screen_datasets.py"
    py_compile.compile(str(script), doraise=True)
