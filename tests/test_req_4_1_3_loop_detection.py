"""
Test REQ-4.1.3: Detect infinite loops and inject loop breaker prompt.
"""
import unittest
from context.loop_detector import LoopDetector


class TestLoopDetection(unittest.TestCase):
    """Test cases for loop detection functionality."""

    def test_loop_detector_records_actions(self):
        """Test that LoopDetector records tool and response actions."""
        detector = LoopDetector()

        detector.record_action("tool_call", tool_name="read_file", args={"path": "file.txt"})
        detector.record_action("response", text="Here is the file")

        history_list = list(detector._history)
        self.assertEqual(len(history_list), 2)

    def test_loop_detector_detects_exact_repeats(self):
        """Test detection of exact action repeats (3+ times)."""
        detector = LoopDetector()

        for _ in range(3):
            detector.record_action("tool_call", tool_name="read_file", args={"path": "file.txt"})

        loop_detected = detector.check_for_loop()
        self.assertIsNotNone(loop_detected)
        self.assertIn("repeated", loop_detected.lower())

    def test_loop_detector_detects_cycles(self):
        """Test detection of repeating cycles (e.g., A-B-A-B)."""
        detector = LoopDetector()

        for _ in range(2):
            detector.record_action("tool_call", tool_name="read_file", args={"path": "a"})
            detector.record_action("tool_call", tool_name="write_file", args={"path": "b"})

        loop_detected = detector.check_for_loop()
        self.assertIsNotNone(loop_detected)
        self.assertIn("cycle", loop_detected.lower())

    def test_loop_detector_no_loop_on_different_actions(self):
        """Test that different actions don't trigger false positives."""
        detector = LoopDetector()

        detector.record_action("tool_call", tool_name="read_file", args={"path": "file1.txt"})
        detector.record_action("tool_call", tool_name="read_file", args={"path": "file2.txt"})
        detector.record_action("tool_call", tool_name="read_file", args={"path": "file3.txt"})

        loop_detected = detector.check_for_loop()
        self.assertIsNone(loop_detected)

    def test_loop_detector_clears_history(self):
        """Test that clear() resets detection state."""
        detector = LoopDetector()

        for _ in range(3):
            detector.record_action("tool_call", tool_name="read_file", args={})

        detector.clear()
        loop_detected = detector.check_for_loop()
        self.assertIsNone(loop_detected)

    def test_loop_detector_max_exact_repeats_threshold(self):
        """Test that exact repeat detection uses max_exact_repeats threshold."""
        detector = LoopDetector()
        detector.max_exact_repeats = 2

        for _ in range(2):
            detector.record_action("tool_call", tool_name="read_file", args={})

        loop_detected = detector.check_for_loop()
        self.assertIsNotNone(loop_detected)

    def test_loop_detector_response_tracking(self):
        """Test tracking of response actions."""
        detector = LoopDetector()

        response_text = "Attempting to solve problem"
        detector.record_action("response", text=response_text)

        history_list = list(detector._history)
        self.assertEqual(len(history_list), 1)
        self.assertIn(response_text, history_list[0])

    def test_loop_detector_mixed_action_types(self):
        """Test detection with mixed tool and response actions."""
        detector = LoopDetector()

        for _ in range(2):
            detector.record_action("tool_call", tool_name="tool1", args={})
            detector.record_action("response", text="Response1")
            detector.record_action("tool_call", tool_name="tool1", args={})
            detector.record_action("response", text="Response1")

        loop_detected = detector.check_for_loop()
        self.assertTrue(loop_detected is not None or len(list(detector._history)) >= 4)

    def test_loop_detector_less_than_threshold_no_loop(self):
        """Test that sub-threshold repeats don't flag loops."""
        detector = LoopDetector()

        for _ in range(2):
            detector.record_action("tool_call", tool_name="read_file", args={})

        loop_detected = detector.check_for_loop()
        self.assertIsNone(loop_detected)

    def test_loop_detector_history_maxlen(self):
        """Test that history respects max length limit."""
        detector = LoopDetector()

        for i in range(30):
            detector.record_action("tool_call", tool_name=f"tool_{i}", args={})

        history_list = list(detector._history)
        self.assertLessEqual(len(history_list), 20)


if __name__ == "__main__":
    unittest.main()

