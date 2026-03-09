"""
Test REQ-4.1.2: Stream LLM responses to the UI in real-time (Text Deltas).
"""
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from agent.agent import Agent
from agent.events import AgentEventType
from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage
from conftest import get_test_config, get_test_agent, AsyncTestCase


class TestStreaming(unittest.TestCase, AsyncTestCase):
    """Test cases for real-time streaming functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        AsyncTestCase.__init__(self)
        self.config = get_test_config()
        self.agent = get_test_agent(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.cleanup()

    def test_streaming_yields_text_deltas_immediately(self):
        """Test that TEXT_DELTA events are yielded as they arrive."""
        text_chunks = ["Hello", " ", "World"]
        response_events = [
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(chunk))
            for chunk in text_chunks
        ] + [
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response_events)

                deltas = []
                async for event in self.agent.run("Test"):
                    if event.type == AgentEventType.TEXT_DELTA:
                        deltas.append(event.data.get("content"))

                self.assertEqual(len(deltas), len(text_chunks))
                self.assertEqual(deltas, text_chunks)

        self.async_test(run_test())

    def test_streaming_maintains_delta_order(self):
        """Test that text deltas maintain order of arrival."""
        text_sequence = ["First", "Second", "Third"]
        response_events = [
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(text))
            for text in text_sequence
        ] + [
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response_events)

                received_sequence = []
                async for event in self.agent.run("Test"):
                    if event.type == AgentEventType.TEXT_DELTA:
                        received_sequence.append(event.data.get("content"))

                self.assertEqual(received_sequence, text_sequence)

        self.async_test(run_test())

    def test_streaming_combines_deltas_in_text_complete(self):
        """Test that TEXT_COMPLETE contains combined delta text."""
        text_fragments = ["Part", "1", "Part", "2"]
        response_events = [
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(frag))
            for frag in text_fragments
        ] + [
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response_events)

                complete_text = None
                async for event in self.agent.run("Test"):
                    if event.type == AgentEventType.TEXT_COMPLETE:
                        complete_text = event.data.get("content")

                expected = "".join(text_fragments)
                self.assertEqual(complete_text, expected)

        self.async_test(run_test())

    def test_streaming_with_tool_calls_interleaved(self):
        """Test streaming continues properly with tool calls interspersed."""
        response_events = [
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("Starting")),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call=MagicMock(call_id="1", name="read_file"),
            ),
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("After tool")),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response_events)

                text_events = []
                tool_events = []
                async for event in self.agent.run("Test"):
                    if event.type == AgentEventType.TEXT_DELTA:
                        text_events.append(event.data.get("content"))
                    elif event.type == AgentEventType.TOOL_CALL_START:
                        tool_events.append(event)

                self.assertGreaterEqual(len(text_events), 2)
                self.assertGreaterEqual(len(tool_events), 1)

        self.async_test(run_test())

    def test_streaming_empty_deltas_handled(self):
        """Test that empty text deltas are handled gracefully."""
        response_events = [
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("Hello")),
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("")),
            StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta("World")),
            StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        async def run_test():
            with patch.object(self.agent.session.client, 'chat_completion') as mock_chat:
                mock_chat.return_value = (event for event in response_events)

                deltas = []
                async for event in self.agent.run("Test"):
                    if event.type == AgentEventType.TEXT_DELTA:
                        deltas.append(event.data.get("content"))

                self.assertEqual(len(deltas), 3)
                self.assertEqual(deltas[1], "")

        self.async_test(run_test())


if __name__ == "__main__":
    unittest.main()
