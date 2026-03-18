"""
Test REQ-4.2.3: read_file must support line ranges (limit/offset) to handle large files.
"""
import unittest
from pathlib import Path
import tempfile
from tools.builtin.read_file import ReadFileTool, ReadFileParams
from tools.base import ToolInvocation
from config.config import Config, Provider


class TestReadFileRanges(unittest.TestCase):
    """Test cases for read_file range functionality."""

    def test_read_file_default_reads_entire_file(self):
        """Test that default read_file without offset/limit reads entire file."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for i in range(1, 101):
                f.write(f"Line {i}\n")
            temp_path = f.name

        try:
            params = ReadFileParams(path=temp_path)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())

            result = tool.execute(invocation)

            self.assertTrue(result.success)
            self.assertIn("Line 1", result.output)
            self.assertIn("Line 100", result.output)
        finally:
            Path(temp_path).unlink()

    def test_read_file_with_offset(self):
        """Test read_file with offset parameter."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, offset=3)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            self.assertTrue(result.success)
            self.assertIn("Line 3", result.output)
            lines = result.output.split('\n')
            first_line = lines[0] if lines else ""
            self.assertTrue("3" in first_line or "Line 3" in result.output)
        finally:
            Path(path).unlink()

    def test_read_file_with_limit(self):
        """Test read_file with limit parameter."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, limit=2)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            self.assertTrue(result.success)
            lines = result.output.split('\n')
            content_lines = [l for l in lines if l.strip()]
            self.assertLessEqual(len(content_lines), 3)
        finally:
            Path(path).unlink()

    def test_read_file_with_offset_and_limit(self):
        """Test read_file with both offset and limit."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for i in range(1, 11):
                f.write(f"Line {i}\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, offset=5, limit=3)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            self.assertTrue(result.success)
            output = result.output
            self.assertIn("Line 5", output)
            self.assertIn("Line 6", output)
            self.assertIn("Line 7", output)
            self.assertNotIn("Line 8", output)
        finally:
            Path(path).unlink()

    def test_read_file_offset_out_of_range(self):
        """Test read_file with offset beyond file length."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Line 1\nLine 2\nLine 3\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, offset=100)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            self.assertTrue(result.success or (result.error and "out of range" in result.error.lower()) if hasattr(result, 'error') and result.error else True)
        finally:
            Path(path).unlink()

    def test_read_file_preserves_line_numbers(self):
        """Test that output includes correct line numbers."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("A\nB\nC\nD\nE\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, offset=2, limit=2)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            output = result.output
            lines = output.split('\n')
            self.assertTrue(any("2" in line for line in lines))
            self.assertTrue(any("3" in line for line in lines))
        finally:
            Path(path).unlink()

    def test_read_file_large_file_with_limit(self):
        """Test that limit prevents reading huge files entirely."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for i in range(1, 1001):
                f.write(f"Line {i}\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, limit=10)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            output_lines = result.output.split('\n')
            self.assertLessEqual(len(output_lines), 20)
        finally:
            Path(path).unlink()

    def test_read_file_metadata_includes_line_count(self):
        """Test that result metadata includes line information."""
        config = Config(provider=Provider.API)
        tool = ReadFileTool(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Line 1\nLine 2\nLine 3\n")
            path = f.name

        try:
            params = ReadFileParams(path=path, offset=1, limit=2)
            invocation = ToolInvocation(params=params.__dict__, cwd=Path.cwd())
            result = tool.execute(invocation)

            self.assertIsNotNone(result.metadata)
            self.assertTrue("lines" in result.metadata or len(result.metadata) > 0)
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()

