from __future__ import annotations

import unittest
from pathlib import Path


class TrainGptMoETests(unittest.TestCase):
    def test_moe_env_vars_and_class_exist(self) -> None:
        text = Path("train_gpt.py").read_text(encoding="utf-8")
        self.assertIn("MOE_EXPERTS", text)
        self.assertIn("MOE_DEPLOY_EXPERT", text)
        self.assertIn("class MoEMLP", text)


if __name__ == "__main__":
    unittest.main()
