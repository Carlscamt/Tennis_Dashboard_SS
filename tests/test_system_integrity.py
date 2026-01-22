"""
System Integrity Test (Regression Testing)

Verifies that the optimized system components are working correctly together.
Run this after any major refactoring.

Usage:
    python tests/test_system_integrity.py
"""
import sys
import unittest
import subprocess
from pathlib import Path
import importlib

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

class TestSystemIntegrity(unittest.TestCase):
    
    def test_cli_entry_point(self):
        """Test that tennis.py is executable and shows help."""
        cmd = [sys.executable, "tennis.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Tennis Prediction System", result.stdout)
        self.assertIn("scrape", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("predict", result.stdout)

    def test_scraper_module_import(self):
        """Test that src.scraper can be imported and has expected functions."""
        try:
            import src.scraper as scraper
            self.assertTrue(hasattr(scraper, "scrape_historical"))
            self.assertTrue(hasattr(scraper, "scrape_upcoming"))
            self.assertTrue(hasattr(scraper, "scrape_players"))
            self.assertTrue(hasattr(scraper, "CheckpointManager"))
            self.assertTrue(hasattr(scraper, "SessionPool") or hasattr(scraper, "get_session"))
        except ImportError as e:
            self.fail(f"Failed to import src.scraper: {e}")

    def test_pipeline_import(self):
        """Test that TennisPipeline can be imported and instantiated."""
        try:
            from src.pipeline import TennisPipeline
            pipeline = TennisPipeline()
            self.assertTrue(hasattr(pipeline, "predict_upcoming"))
            self.assertTrue(hasattr(pipeline, "daily_update"))
            self.assertTrue(hasattr(pipeline, "_identify_unknown_players")) # verify new method
        except ImportError as e:
            self.fail(f"Failed to import src.pipeline: {e}")

    def test_predict_command_dry_run(self):
        """Test the predict command (dry run / quick check)."""
        # We run with --no-scrape to avoid network calls and --days 0 to be quick
        cmd = [
            sys.executable, "tennis.py", "predict", 
            "--days", "0", 
            "--no-scrape"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        
        # It's okay if it returns 0 (success) or fails gracefully if no data
        # We just want to ensure it doesn't crash with SyntaxError or ImportError
        if result.returncode != 0:
            print(f"\n[INFO] Predict command stdout: {result.stdout}")
            print(f"[INFO] Predict command stderr: {result.stderr}")
        
        # Check for standard error output indicating import/syntax crashes
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertNotIn("SyntaxError", result.stderr)
        self.assertNotIn("IndentationError", result.stderr)

if __name__ == "__main__":
    print(f"Running system integrity tests from: {ROOT}")
    unittest.main()
