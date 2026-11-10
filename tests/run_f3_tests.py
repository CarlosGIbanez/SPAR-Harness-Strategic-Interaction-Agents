"""Run the F3 tests without pytest: python tests/run_f3_tests.py (from the repo root, PYTHONPATH=src)."""
import importlib, sys, traceback
sys.path.insert(0, "tests")
mod = importlib.import_module("test_f3_game")
failed = 0
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    try:
        getattr(mod, name)(); print("ok  ", name)
    except Exception:
        failed += 1; print("FAIL", name); traceback.print_exc()
sys.exit(1 if failed else 0)
