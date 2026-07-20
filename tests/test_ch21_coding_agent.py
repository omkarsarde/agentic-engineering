import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch21"))

from coding_agent import Edit, EditRejected, apply_edit, load_tasks, repository_map, run_suite


class CodingAgentTest(unittest.TestCase):
    def test_fixture_exercises_success_repair_and_rejection(self):
        report = run_suite(load_tasks())
        self.assertEqual(report["tasks"], 6)
        self.assertEqual(report["resolved"], 6)
        self.assertEqual(report["test_runs"], 7)
        self.assertEqual(report["rejected_edits"], 1)
        self.assertEqual(report["proposals"], 8)
        json.dumps(report)

    def test_tight_budget_exposes_unresolved_tasks(self):
        report = run_suite(load_tasks(), max_proposals=1)
        self.assertEqual(report["tasks"], 6)
        self.assertEqual(report["resolved"], 4)

    def test_workspace_escape_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(EditRejected):
                apply_edit(root, Edit("../escape.py", "a", "b", "malformed"))

    def test_tests_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "test_app.py").write_text("value = 1\n", encoding="utf-8")
            with self.assertRaises(EditRejected):
                apply_edit(root, Edit("test_app.py", "1", "2", "game the test"))

    def test_ambiguous_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "app.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
            with self.assertRaises(EditRejected):
                apply_edit(root, Edit("app.py", "x = 1", "x = 2", "ambiguous"))

    def test_repository_map_exposes_structure_not_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
            mapped = repository_map(root)
            self.assertEqual(mapped[0]["symbols"], ["answer"])
            self.assertNotIn("source", mapped[0])


if __name__ == "__main__":
    unittest.main()
