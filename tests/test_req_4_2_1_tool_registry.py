"""
Test REQ-4.2.1: System provides built-in tools.
"""
import unittest
from config.config import Config, Provider
from tools.registry import ToolRegistry, create_default_registry
from tools.builtin import get_all_builtin_tools
from tools.base import Tool


class TestToolRegistry(unittest.TestCase):
    """Test cases for tool registry functionality."""

    def test_tool_registry_registers_tool(self):
        """Test that tools can be registered."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)

        class MockTool(Tool):
            name = "mock_tool"
            description = "A mock tool"
            schema = None

        tool = MockTool(config)
        registry.register(tool)

        retrieved = registry.get("mock_tool")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "mock_tool")

    def test_tool_registry_get_all_tools(self):
        """Test retrieving all registered tools."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)

        class Tool1(Tool):
            name = "tool1"
            description = "Tool 1"
            schema = None

        class Tool2(Tool):
            name = "tool2"
            description = "Tool 2"
            schema = None

        registry.register(Tool1(config))
        registry.register(Tool2(config))

        tools = registry.get_tools()
        self.assertEqual(len(tools), 2)
        tool_names = {t.name for t in tools}
        self.assertIn("tool1", tool_names)
        self.assertIn("tool2", tool_names)

    def test_tool_registry_get_schemas(self):
        """Test that tool schemas are generated correctly."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)

        class SchemaedTool(Tool):
            name = "schemaed"
            description = "Tool with schema"
            
            def __init__(self, config):
                super().__init__(config)

        registry.register(SchemaedTool(config))

        schemas = registry.get_schemas()
        self.assertGreaterEqual(len(schemas), 1)
        self.assertTrue(any(s.get("name") == "schemaed" for s in schemas))

    def test_tool_registry_unregister(self):
        """Test that tools can be unregistered."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)

        class TempTool(Tool):
            name = "temp"
            description = "Temporary"
            schema = None

        registry.register(TempTool(config))
        self.assertIsNotNone(registry.get("temp"))

        registry.unregister("temp")
        self.assertIsNone(registry.get("temp"))

    def test_tool_registry_overwrite_warning(self):
        """Test that registering duplicate tool name overwrites."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)

        class Tool1(Tool):
            name = "duplicate"
            description = "First"
            schema = None

        class Tool2(Tool):
            name = "duplicate"
            description = "Second"
            schema = None

        registry.register(Tool1(config))
        registry.register(Tool2(config))

        tool = registry.get("duplicate")
        self.assertEqual(tool.description, "Second")

    def test_builtin_tools_available(self):
        """Test that all built-in tools are available."""
        builtin_tools = get_all_builtin_tools()
        self.assertGreater(len(builtin_tools), 0)

    def test_default_registry_includes_builtin_tools(self):
        """Test that create_default_registry includes built-in tools."""
        config = Config(provider=Provider.API)
        registry = create_default_registry(config)

        tools = registry.get_tools()
        tool_names = {t.name for t in tools}

        self.assertIn("read_file", tool_names)
        self.assertIn("write_file", tool_names)
        self.assertIn("edit", tool_names)
        self.assertIn("shell", tool_names)

    def test_tool_registry_with_allowed_tools_filter(self):
        """Test that allowed_tools config filters available tools."""
        config = Config(provider=Provider.API, allowed_tools=["read_file", "write_file"])
        registry = create_default_registry(config)

        tools = registry.get_tools()
        tool_names = {t.name for t in tools}

        self.assertIn("read_file", tool_names)
        self.assertIn("write_file", tool_names)
        filtered_tools = [t for t in tool_names if t not in ["read_file", "write_file"]]
        self.assertEqual(len(filtered_tools), 0)

    def test_tool_registry_mcp_tools_separate(self):
        """Test that MCP tools are tracked separately."""
        config = Config(provider=Provider.API)
        registry = ToolRegistry(config)

        class RegularTool(Tool):
            name = "regular"
            description = "Regular"
            schema = None

        class MCPToolMock(Tool):
            name = "mcp_tool"
            description = "MCP"
            schema = None

        registry.register(RegularTool(config))
        registry.register_mcp_tool(MCPToolMock(config))

        tools = registry.get_tools()
        tool_names = {t.name for t in tools}
        self.assertIn("regular", tool_names)
        self.assertIn("mcp_tool", tool_names)


if __name__ == "__main__":
    unittest.main()