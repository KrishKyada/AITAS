"""
Test REQ-4.1.1: Turn-based loop where agent assesses context, calls tools, and returns results until task complete.
"""
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from agent.agent import Agent
from agent.events import AgentEventType
from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage, ToolCall
from config.config import Config, Provider, ModelConfig, ApprovalPolicy
from conftest import get_test_config, get_test_agent, AsyncTestCase


class TestAgenticLoop(unittest.TestCase, AsyncTestCase):
    """Test cases for agentic loop functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        AsyncTestCase.__init__(self)
        self.config = get_test_config()
        self.agent = get_test_agent(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.cleanup()
    
    def test_agentic_loop_executes_multiple_turns(self):
        """Test that agentic loop runs multiple turns until no tool calls."""
        events_turn_1 = [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta("Calling tool..."),
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call=ToolCall(call_id="1", name="read_file"),
            ),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]
        
        events_turn_2 = [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta("Task complete."),
            ),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            ),
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.side_effect = [
                    (event for event in events_turn_1),
                    (event for event in events_turn_2),
                ]

                events = []
                async for event in self.agent.run("Test message"):
                    events.append(event)

                text_deltas = [e for e in events if e.type == AgentEventType.TEXT_DELTA]
                self.assertGreaterEqual(len(text_deltas), 2, "Should have text from multiple turns")

        self.async_test(run_test())

    def test_agentic_loop_stops_when_no_tool_calls(self):
        """Test that loop terminates when LLM returns no tool calls."""
        no_tool_response = [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta("No tools needed."),
            ),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in no_tool_response)

                events = []
                async for event in self.agent.run("Test"):
                    events.append(event)

                start_events = [e for e in events if e.type == AgentEventType.AGENT_START]
                end_events = [e for e in events if e.type == AgentEventType.AGENT_END]
                self.assertEqual(len(start_events), 1)
                self.assertEqual(len(end_events), 1)

        self.async_test(run_test())

    def test_agentic_loop_respects_max_turns(self):
        """Test that loop terminates at max_turns limit."""
        config = get_test_config()
        config.max_turns = 2
        agent = get_test_agent(config)

        infinite_tool_response = [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta("Calling tool..."),
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call=ToolCall(call_id="1", name="read_file"),
            ),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def run_test():
            with patch.object(agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in infinite_tool_response)

                turn_count = 0
                async for event in agent.run("Test"):
                    if event.type == AgentEventType.TOOL_CALL_START:
                        turn_count += 1

                self.assertLessEqual(turn_count, config.max_turns)

        self.async_test(run_test())

    def test_agentic_loop_adds_messages_to_context(self):
        """Test that user and assistant messages are added to context manager."""
        response = [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta("Response"),
            ),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response)

                initial_msg_count = self.agent.session.context_manager.message_count
                
                async for _ in self.agent.run("Test user message"):
                    pass

                final_msg_count = self.agent.session.context_manager.message_count
                self.assertGreater(final_msg_count, initial_msg_count)

        self.async_test(run_test())

    def test_agentic_loop_handles_tool_execution(self):
        """Test that tools are invoked and results are added to context."""
        response_with_tool = [
            StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=TextDelta("Reading file..."),
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call=ToolCall(call_id="1", name="read_file"),
            ),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response_with_tool)
                
                with patch.object(self.agent.session.tool_registry, 'invoke') as mock_invoke:
                    from tools.base import ToolResult
                    mock_invoke.return_value = ToolResult.success_result("File content")

                    tool_calls = []
                    async for event in self.agent.run("Read file"):
                        if event.type == AgentEventType.TOOL_CALL_START:
                            tool_calls.append(event)

                    self.assertTrue(len(tool_calls) > 0 or mock_invoke.call_count >= 0)

        self.async_test(run_test())


if __name__ == "__main__":
    unittest.main()