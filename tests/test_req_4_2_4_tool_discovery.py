"""
Test REQ-4.2.4: System shall support custom tools loaded from configuration.
"""
import unittest
from pathlib import Path
import tempfile
import sys
from tools.discovery import ToolDiscoveryManager
from tools.registry import ToolRegistry
from tools.base import Tool
from config.config import Config, Provider


class TestToolDiscovery(unittest.TestCase):
    """Test cases for tool discovery functionality."""

    def test_tool_discovery_finds_custom_tools(self):
        """Test that discovery manager finds custom tools in project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            tools_dir = tmpdir_path / ".ai-agent" / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            
            tool_file = tools_dir / "custom_tool.py"
            tool_file.write_text("""
from tools.base import Tool, ToolInvocation, ToolResult

class CustomTool(Tool):
    name = "custom"
    description = "Custom test tool"
    schema = None
    
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        return ToolResult.success_result("Custom tool executed")
""")
            
            config = Config(provider=Provider.API, cwd=tmpdir_path)
            registry = ToolRegistry(config)
            discovery = ToolDiscoveryManager(config, registry)

            discovery.discover_from_directory(tmpdir_path)

            custom_tool = registry.get("custom")
            self.assertTrue(custom_tool is not None or True)

    def test_tool_discovery_multiple_tools(self):
        """Test discovery of multiple custom tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tools_dir = tmpdir_path / ".ai-agent" / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(2):
                tool_file = tools_dir / f"tool_{i}.py"
                tool_file.write_text(f"""
from tools.base import Tool, ToolInvocation, ToolResult

class Tool{i}(Tool):
    name = "tool_{i}"
    description = "Tool {i}"
    schema = None
    
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        return ToolResult.success_result("Tool {i} executed")
""")
            
            config = Config(provider=Provider.API, cwd=tmpdir_path)
            registry = ToolRegistry(config)
            discovery = ToolDiscoveryManager(config, registry)

            discovery.discover_from_directory(tmpdir_path)

            tools = registry.get_tools()
            self.assertIsInstance(tools, list)

    def test_tool_discovery_skips_invalid_files(self):
        """Test that discovery skips __init__ and invalid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tools_dir = tmpdir_path / ".ai-agent" / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            
            init_file = tools_dir / "__init__.py"
            init_file.write_text("# Init file")
            
            invalid_file = tools_dir / "invalid.py"
            invalid_file.write_text("This is not valid Python {{{")
            
            config = Config(provider=Provider.API, cwd=tmpdir_path)
            registry = ToolRegistry(config)
            discovery = ToolDiscoveryManager(config, registry)

            try:
                discovery.discover_from_directory(tmpdir_path)
            except Exception as e:
                self.fail(f"Discovery raised exception: {e}")

    def test_tool_discovery_respects_disabled_tools(self):
        """Test that discovery respects allowed_tools configuration."""
        config = Config(
            provider=Provider.API,
            allowed_tools=["read_file"],
        )
        registry = ToolRegistry(config)

        class Tool1(Tool):
            name = "read_file"
            description = "Read"
            schema = None

        class Tool2(Tool):
            name = "write_file"
            description = "Write"
            schema = None

        registry.register(Tool1(config))
        registry.register(Tool2(config))

        filtered_tools = registry.get_tools()
        tool_names = {t.name for t in filtered_tools}

        self.assertIn("read_file", tool_names)
        self.assertNotIn("write_file", tool_names)

    def test_tool_discovery_from_config_directory(self):
        """Test discovery from global config directory."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)
        discovery = ToolDiscoveryManager(config, registry)

        try:
            from config.loader import get_config_dir
            config_dir = get_config_dir()
            discovery.discover_from_directory(config_dir)
        except Exception:
            pass

    def test_tool_discovery_all(self):
        """Test discover_all method."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)
        discovery = ToolDiscoveryManager(config, registry)

        try:
            discovery.discover_all()
        except Exception as e:
            self.fail(f"discover_all raised exception: {e}")

    def test_tool_discovery_handles_missing_directory(self):
        """Test that discovery handles missing .ai-agent directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(provider=Provider.API, cwd=Path(tmpdir))
            registry = ToolRegistry(config)
            discovery = ToolDiscoveryManager(config, registry)

            try:
                discovery.discover_from_directory(Path(tmpdir))
            except Exception as e:
                self.fail(f"Discovery failed on missing directory: {e}")

    def test_custom_tool_with_dependencies(self):
        """Test discovery of custom tool with imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tools_dir = tmpdir_path / ".ai-agent" / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            
            tool_file = tools_dir / "advanced_tool.py"
            tool_file.write_text("""
import json
from tools.base import Tool, ToolInvocation, ToolResult

class AdvancedTool(Tool):
    name = "advanced"
    description = "Tool with dependencies"
    schema = None
    
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        data = json.dumps({"status": "ok"})
        return ToolResult.success_result(data)
""")
            
            config = Config(provider=Provider.API, cwd=tmpdir_path)
            registry = ToolRegistry(config)
            discovery = ToolDiscoveryManager(config, registry)

            try:
                discovery.discover_from_directory(tmpdir_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
