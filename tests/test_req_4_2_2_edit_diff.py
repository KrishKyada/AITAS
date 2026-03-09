"""
Test REQ-4.2.2: edit_file must provide a diff preview before applying changes.
"""
import unittest
from pathlib import Path
import tempfile
from tools.builtin.edit_file import EditTool, EditParams
from tools.base import ToolInvocation, FileDiff
from config.config import Config, Provider


class TestEditDiff(unittest.TestCase):
    """Test cases for edit tool diff functionality."""

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write("Line 1\nLine 2\nLine 3\n")
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Clean up temporary file."""
        Path(self.temp_path).unlink()

    def test_edit_file_get_confirmation_creates_diff_for_existing_file(self):
        """Test that get_confirmation creates FileDiff for existing file."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        params = EditParams(
            path=self.temp_path,
            old_string="Line 2",
            new_string="Modified Line 2",
        )
        invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

        confirmation = tool.get_confirmation(invocation)

        self.assertIsNotNone(confirmation)
        self.assertIsNotNone(confirmation.diff)
        self.assertIsNotNone(confirmation.diff.old_content)
        self.assertIsNotNone(confirmation.diff.new_content)

    def test_edit_file_diff_shows_changes(self):
        """Test that diff correctly shows old and new content."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        old_str = "Line 2"
        new_str = "REPLACED"

        params = EditParams(
            path=self.temp_path,
            old_string=old_str,
            new_string=new_str,
        )
        invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

        confirmation = tool.get_confirmation(invocation)

        self.assertIn(old_str, confirmation.diff.old_content)
        self.assertIn(new_str, confirmation.diff.new_content)

    def test_edit_file_diff_for_new_file(self):
        """Test that diff is created for new file creation."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = Path(tmpdir) / "new_file.txt"
            
            params = EditParams(
                path=str(new_file),
                old_string="",
                new_string="New content",
            )
            invocation = ToolInvocation(params=params.__dict__, cwd=Path(tmpdir))

            confirmation = tool.get_confirmation(invocation)

            self.assertIsNotNone(confirmation)
            self.assertIsNotNone(confirmation.diff)
            self.assertTrue(confirmation.diff.is_new_file)
            self.assertEqual(confirmation.diff.old_content, "")
            self.assertEqual(confirmation.diff.new_content, "New content")

    def test_edit_file_confirmation_has_affected_paths(self):
        """Test that confirmation includes affected paths."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        params = EditParams(
            path=self.temp_path,
            old_string="Line 1",
            new_string="Modified",
        )
        invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

        confirmation = tool.get_confirmation(invocation)

        self.assertIsNotNone(confirmation.affected_paths)
        self.assertGreater(len(confirmation.affected_paths), 0)
        self.assertEqual(confirmation.affected_paths[0], Path(self.temp_path).resolve())

    def test_edit_file_confirmation_description(self):
        """Test that confirmation has descriptive text."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        params = EditParams(
            path=self.temp_path,
            old_string="Line 1",
            new_string="Modified",
        )
        invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

        confirmation = tool.get_confirmation(invocation)

        self.assertIsNotNone(confirmation.description)
        self.assertTrue("Edit" in confirmation.description or "file" in confirmation.description.lower())

    def test_edit_file_diff_handles_multiline_changes(self):
        """Test diff with multi-line replacements."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        params = EditParams(
            path=self.temp_path,
            old_string="Line 1\nLine 2",
            new_string="Replaced block\nWith multiple\nLines",
        )
        invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

        confirmation = tool.get_confirmation(invocation)

        self.assertIn("Line 1\nLine 2", confirmation.diff.old_content)
        self.assertIn("Replaced block", confirmation.diff.new_content)

    def test_edit_file_confirmation_tool_name(self):
        """Test that confirmation includes tool name."""
        config = Config(provider=Provider.API)
        tool = EditTool(config)

        params = EditParams(
            path=self.temp_path,
            old_string="Line 1",
            new_string="Modified",
        )
        invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

        confirmation = tool.get_confirmation(invocation)

        self.assertEqual(confirmation.tool_name, "edit")

    def test_edit_file_diff_to_diff_method(self):
        """Test that FileDiff.to_diff() generates proper diff output."""
        diff = FileDiff(
            path=Path("/test/file.py"),
            old_content="def foo():\n    pass",
            new_content="def foo():\n    return True",
            is_new_file=False,
        )

        diff_output = diff.to_diff()
        self.assertIsInstance(diff_output, dict)
        self.assertIn("path", diff_output)
        self.assertTrue("old_content" in diff_output or "changes" in str(diff_output).lower())


if __name__ == "__main__":
    unittest.main()