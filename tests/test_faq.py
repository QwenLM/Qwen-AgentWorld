import unittest
from pathlib import Path


class FaqDocumentationTest(unittest.TestCase):
    def test_faq_documents_model_lineage_scope_and_prompt_source(self):
        faq = Path(__file__).resolve().parents[1] / "docs" / "FAQ.md"
        text = faq.read_text(encoding="utf-8")

        self.assertIn("benchmark results", text)
        self.assertIn("not the authoritative source", text)
        self.assertIn("third-party model lineage", text)
        self.assertIn("SimRL or OpenClaw-style runs", text)
        self.assertIn("system_str", text)
        self.assertIn("current_prompt", text)
        self.assertIn("prompt fields shipped with the benchmark/evaluation data", text)


if __name__ == "__main__":
    unittest.main()
