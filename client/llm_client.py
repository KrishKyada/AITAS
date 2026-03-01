import asyncio
from typing import Any, AsyncGenerator
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from client.response import (
    StreamEventType,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)
from config.config import Config


class LLMClient:
    def __init__(self, config: Config) -> None:
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3
        self.config = config
        self.tools_supported: bool = True

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,  
                base_url=self.config.base_url,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str, Any]]):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                },
            }
            for tool in tools
        ]

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()

        # If model doesn't support native tools, don't send them
        if not self.tools_supported:
            tools = None

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate limit exceeded: {e}",
                    )
                    return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Connection error: {e}",
                    )
                    return
            except APIError as e:
                # Detect "does not support tools" error from Ollama
                error_msg = str(e).lower()
                if "does not support tools" in error_msg and tools:
                    self.tools_supported = False
                    # Retry without tools — caller will use prompt-based fallback
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=f"API error: {e}",
                )
                return

    async def _stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                cached = 0
                if (
                    hasattr(chunk.usage, "prompt_tokens_details")
                    and chunk.usage.prompt_tokens_details
                    and hasattr(chunk.usage.prompt_tokens_details, "cached_tokens")
                ):
                    cached = chunk.usage.prompt_tokens_details.cached_tokens or 0
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                    cached_tokens=cached,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content),
                )

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call_delta.id and not tool_calls[idx]["id"]:
                        tool_calls[idx]["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            if not tool_calls[idx]["name"]:
                                tool_calls[idx]["name"] = tool_call_delta.function.name
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    tool_call_delta=ToolCallDelta(
                                        call_id=tool_calls[idx]["id"],
                                        name=tool_call_delta.function.name,
                                    ),
                                )

                        if tool_call_delta.function.arguments:
                            tool_calls[idx]["arguments"] += tool_call_delta.function.arguments

                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_calls[idx]["name"],
                                    arguments_delta=tool_call_delta.function.arguments,
                                ),
                            )

        for idx, tc in tool_calls.items():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=tc["id"],
                    name=tc["name"],
                    arguments=parse_tool_call_arguments(tc["arguments"]),
                ),
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _non_stream_response(
        self,
        client: AsyncOpenAI,
        kwargs: dict[str, Any],
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=parse_tool_call_arguments(tc.function.arguments),
                    )
                )

        usage = None
        if response.usage:
            cached = 0
            if (
                hasattr(response.usage, "prompt_tokens_details")
                and response.usage.prompt_tokens_details
                and hasattr(response.usage.prompt_tokens_details, "cached_tokens")
            ):
                cached = response.usage.prompt_tokens_details.cached_tokens or 0
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
                cached_tokens=cached,
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
