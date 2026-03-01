import asyncio
import os
from pathlib import Path
import sys
import click

from agent.agent import Agent
from agent.events import AgentEventType
from agent.persistence import PersistenceManager, SessionSnapshot
from agent.session import Session
from client.ollama import check_ollama_running, list_ollama_models
from config.config import ApprovalPolicy, Config, Provider
from config.loader import load_config
from rich.rule import Rule
from ui.tui import TUI, get_console

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.agent: Agent | None = None
        self.config = config
        self.tui = TUI(config, console)
        self.current_task = None

    async def run_single(self, message: str) -> str | None:
        async with Agent(self.config) as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self) -> str | None:
        self.tui.print_welcome(
            "AITAS",
            lines=[
                f"provider: {self.config.provider.value}",
                f"model: {self.config.model_name}",
                f"cwd: {self.config.cwd}",
                "commands: /help /config /approval /model /exit",
            ],
        )

        async with Agent(
            self.config,
            confirmation_callback=self.tui.handle_confirmation,
        ) as agent:
            self.agent = agent

            while True:
                try:
                    user_input = console.input("\n[bold bright_cyan]AITAS >[/bold bright_cyan] ").strip()
                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        should_continue = await self._handle_command(user_input)
                        if not should_continue:
                            break
                        continue

                    # Create task for agent execution so we can cancel it
                    self.current_task = asyncio.create_task(self._process_message(user_input))

                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        console.print("\n[yellow]Agent execution cancelled[/yellow]")
                    finally:
                        self.current_task = None

                except KeyboardInterrupt:
                    # Handle Ctrl+C during agent execution
                    if self.current_task and not self.current_task.done():
                        console.print("\n[yellow]Cancelling agent execution...[/yellow]")
                        self.current_task.cancel()
                        try:
                            await self.current_task
                        except asyncio.CancelledError:
                            pass
                        self.current_task = None
                        # Continue the loop to get next input
                        continue
                    else:
                        # Ctrl+C pressed while waiting for input - inform user
                        console.print("\n[dim]Type /exit to quit AITAS[/dim]")
                except EOFError:
                    break

        console.print("\n[dim]AITAS session ended. Goodbye![/dim]")

    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool_kind = None
        tool = self.agent.session.tool_registry.get(tool_name)
        if not tool:
            tool_kind = None

        tool_kind = tool.kind.value

        return tool_kind

    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None

        try:
            async for event in self.agent.run(message):
                if event.type == AgentEventType.TEXT_DELTA:
                    content = event.data.get("content", "")
                    if not assistant_streaming:
                        self.tui.begin_assistant()
                        assistant_streaming = True
                    self.tui.stream_assistant_delta(content)
                elif event.type == AgentEventType.TEXT_COMPLETE:
                    final_response = event.data.get("content")
                    if assistant_streaming:
                        self.tui.end_assistant()
                        assistant_streaming = False
                elif event.type == AgentEventType.AGENT_ERROR:
                    error = event.data.get("error", "Unknown error")
                    console.print(f"\n[error]Error: {error}[/error]")
                elif event.type == AgentEventType.TOOL_CALL_START:
                    tool_name = event.data.get("name", "unknown")
                    tool_kind = self._get_tool_kind(tool_name)
                    self.tui.tool_call_start(
                        event.data.get("call_id", ""),
                        tool_name,
                        tool_kind,
                        event.data.get("arguments", {}),
                    )
                elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                    tool_name = event.data.get("name", "unknown")
                    tool_kind = self._get_tool_kind(tool_name)
                    self.tui.tool_call_complete(
                        event.data.get("call_id", ""),
                        tool_name,
                        tool_kind,
                        event.data.get("success", False),
                        event.data.get("output", ""),
                        event.data.get("error"),
                        event.data.get("metadata"),
                        event.data.get("diff"),
                        event.data.get("truncated", False),
                        event.data.get("exit_code"),
                    )
        except asyncio.CancelledError:
            if assistant_streaming:
                self.tui.end_assistant()
            console.print("\n[yellow]Agent execution cancelled[/yellow]")
            raise

        return final_response

    async def _handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()
        parts = cmd.split(maxsplit=1)
        cmd_name = parts[0]
        cmd_args = parts[1] if len(parts) > 1 else ""
        if cmd_name == "/exit" or cmd_name == "/quit":
            return False
        elif command == "/help":
            self.tui.show_help()
        elif command == "/clear":
            self.agent.session.context_manager.clear()
            self.agent.session.loop_detector.clear()
            console.print("[success]Conversation cleared [/success]")
        elif command == "/config":
            console.print("\n[bold]Current Configuration[/bold]")
            console.print(f"  Provider: {self.config.provider.value}")
            console.print(f"  Model: {self.config.model_name}")
            console.print(f"  Temperature: {self.config.temperature}")
            console.print(f"  Approval: {self.config.approval.value}")
            console.print(f"  Working Dir: {self.config.cwd}")
            console.print(f"  Max Turns: {self.config.max_turns}")
            console.print(f"  Hooks Enabled: {self.config.hooks_enabled}")
            if self.config.provider == Provider.OLLAMA:
                console.print(f"  Ollama URL: {self.config.ollama_base_url}")
        elif cmd_name == "/model":
            if cmd_args:
                self.config.model_name = cmd_args
                console.print(f"[success]Model changed to: {cmd_args} [/success]")
            else:
                console.print(f"Current model: {self.config.model_name}")
        elif cmd_name == "/approval":
            if cmd_args:
                try:
                    approval = ApprovalPolicy(cmd_args)
                    self.config.approval = approval
                    console.print(
                        f"[success]Approval policy changed to: {cmd_args} [/success]"
                    )
                except:
                    console.print(
                        f"[error]Incorrect approval policy: {cmd_args} [/error]"
                    )
                    console.print(
                        f"Valid options: {', '.join(p for p in ApprovalPolicy)}"
                    )
            else:
                console.print(f"Current approval policy: {self.config.approval.value}")
        elif cmd_name == "/stats":
            stats = self.agent.session.get_stats()
            console.print("\n[bold]Session Statistics [/bold]")
            for key, value in stats.items():
                console.print(f"   {key}: {value}")
        elif cmd_name == "/tools":
            tools = self.agent.session.tool_registry.get_tools()
            console.print(f"\n[bold]Available tools ({len(tools)}) [/bold]")
            for tool in tools:
                console.print(f"  • {tool.name}")
        elif cmd_name == "/mcp":
            mcp_servers = self.agent.session.mcp_manager.get_all_servers()
            console.print(f"\n[bold]MCP Servers ({len(mcp_servers)}) [/bold]")
            for server in mcp_servers:
                status = server["status"]
                status_color = "green" if status == "connected" else "red"
                console.print(
                    f"  • {server['name']}: [{status_color}]{status}[/{status_color}] ({server['tools']} tools)"
                )
        elif cmd_name == "/save":
            persistence_manager = PersistenceManager()
            session_snapshot = SessionSnapshot(
                session_id=self.agent.session.session_id,
                created_at=self.agent.session.created_at,
                updated_at=self.agent.session.updated_at,
                turn_count=self.agent.session.turn_count,
                messages=self.agent.session.context_manager.get_messages(),
                total_usage=self.agent.session.context_manager.total_usage,
            )
            persistence_manager.save_session(session_snapshot)
            console.print(
                f"[success]Session saved: {self.agent.session.session_id}[/success]"
            )
        elif cmd_name == "/sessions":
            persistence_manager = PersistenceManager()
            sessions = persistence_manager.list_sessions()
            console.print("\n[bold]Saved Sessions[/bold]")
            for s in sessions:
                console.print(
                    f"  • {s['session_id']} (turns: {s['turn_count']}, updated: {s['updated_at']})"
                )
        elif cmd_name == "/resume":
            if not cmd_args:
                console.print(f"[error]Usage: /resume <session_id> [/error]")
            else:
                persistence_manager = PersistenceManager()
                snapshot = persistence_manager.load_session(cmd_args)
                if not snapshot:
                    console.print(f"[error]Session does not exist [/error]")
                else:
                    session = Session(
                        config=self.config,
                    )
                    await session.initialize()
                    session.session_id = snapshot.session_id
                    session.created_at = snapshot.created_at
                    session.updated_at = snapshot.updated_at
                    session.turn_count = snapshot.turn_count
                    session.context_manager.total_usage = snapshot.total_usage

                    for msg in snapshot.messages:
                        if msg.get("role") == "system":
                            continue
                        elif msg["role"] == "user":
                            session.context_manager.add_user_message(
                                msg.get("content", "")
                            )
                        elif msg["role"] == "assistant":
                            session.context_manager.add_assistant_message(
                                msg.get("content", ""), msg.get("tool_calls")
                            )
                        elif msg["role"] == "tool":
                            session.context_manager.add_tool_result(
                                msg.get("tool_call_id", ""), msg.get("content", "")
                            )

                    await self.agent.session.client.close()
                    await self.agent.session.mcp_manager.shutdown()

                    self.agent.session = session
                    console.print(
                        f"[success]Resumed session: {session.session_id}[/success]"
                    )
        elif cmd_name == "/checkpoint":
            persistence_manager = PersistenceManager()
            session_snapshot = SessionSnapshot(
                session_id=self.agent.session.session_id,
                created_at=self.agent.session.created_at,
                updated_at=self.agent.session.updated_at,
                turn_count=self.agent.session.turn_count,
                messages=self.agent.session.context_manager.get_messages(),
                total_usage=self.agent.session.context_manager.total_usage,
            )
            checkpoint_id = persistence_manager.save_checkpoint(session_snapshot)
            console.print(f"[success]Checkpoint created: {checkpoint_id}[/success]")
        elif cmd_name == "/restore":
            if not cmd_args:
                console.print(f"[error]Usage: /restire <checkpoint_id> [/error]")
            else:
                persistence_manager = PersistenceManager()
                snapshot = persistence_manager.load_checkpoint(cmd_args)
                if not snapshot:
                    console.print(f"[error]Checkpoint does not exist [/error]")
                else:
                    session = Session(
                        config=self.config,
                    )
                    await session.initialize()
                    session.session_id = snapshot.session_id
                    session.created_at = snapshot.created_at
                    session.updated_at = snapshot.updated_at
                    session.turn_count = snapshot.turn_count
                    session.context_manager.total_usage = snapshot.total_usage

                    for msg in snapshot.messages:
                        if msg.get("role") == "system":
                            continue
                        elif msg["role"] == "user":
                            session.context_manager.add_user_message(
                                msg.get("content", "")
                            )
                        elif msg["role"] == "assistant":
                            session.context_manager.add_assistant_message(
                                msg.get("content", ""), msg.get("tool_calls")
                            )
                        elif msg["role"] == "tool":
                            session.context_manager.add_tool_result(
                                msg.get("tool_call_id", ""), msg.get("content", "")
                            )

                    await self.agent.session.client.close()
                    await self.agent.session.mcp_manager.shutdown()

                    self.agent.session = session
                    console.print(
                        f"[success]Resumed session: {session.session_id}, checkpoint: {checkpoint_id}[/success]"
                    )
        else:
            console.print(f"[error]Unknown command: {cmd_name}[/error]")

        return True


async def select_provider(config: Config) -> Config:
    """Prompt the user to choose between API or Ollama provider."""
    console.print()
    console.print("[bold bright_cyan]  Select Provider[/bold bright_cyan]")
    console.print(Rule(style="grey35"))
    console.print("  [bright_cyan]1[/bright_cyan]  API [dim](OpenAI / OpenRouter / compatible)[/dim]")
    console.print("  [bright_cyan]2[/bright_cyan]  Ollama [dim](local models)[/dim]")

    while True:
        choice = console.input("\n[bold bright_cyan]  >[/bold bright_cyan] ").strip()
        if choice in ("1", "2"):
            break
        console.print("[error]  Invalid choice. Enter 1 or 2.[/error]")

    if choice == "1":
        config.provider = Provider.API

        # --- API configuration ---
        console.print()
        console.print("[bold bright_cyan]  API Configuration[/bold bright_cyan]")
        console.print(Rule(style="grey35"))

        # API key
        existing_key = os.environ.get("API_KEY", "")
        if existing_key:
            masked = existing_key[:4] + "..." + existing_key[-4:] if len(existing_key) > 8 else "****"
            console.print(f"  [dim]API_KEY already set ({masked})[/dim]")
            change_key = console.input("  [muted]Change it? (y/N):[/muted] ").strip().lower()
            if change_key in ("y", "yes"):
                existing_key = ""

        if not existing_key:
            while True:
                api_key = console.input("  [bright_cyan]API Key:[/bright_cyan] ").strip()
                if api_key:
                    os.environ["API_KEY"] = api_key
                    break
                console.print("  [error]API key cannot be empty.[/error]")

        # Base URL
        existing_url = os.environ.get("BASE_URL", "")
        if existing_url:
            console.print(f"  [dim]BASE_URL already set ({existing_url})[/dim]")
            change_url = console.input("  [muted]Change it? (y/N):[/muted] ").strip().lower()
            if change_url in ("y", "yes"):
                existing_url = ""

        if not existing_url:
            while True:
                base_url = console.input("  [bright_cyan]Base URL:[/bright_cyan] ").strip()
                if base_url:
                    os.environ["BASE_URL"] = base_url
                    break
                console.print("  [error]Base URL cannot be empty.[/error]")

        # Model name
        current_model = config.model.name
        console.print(f"  [dim]Current model: {current_model}[/dim]")
        model_input = console.input("  [bright_cyan]Model name[/bright_cyan] [dim](Enter to keep):[/dim] ").strip()
        if model_input:
            config.model.name = model_input

        console.print(f"\n  [success]API configured: model={config.model.name}[/success]")
        return config

    # --- Ollama path ---
    config.provider = Provider.OLLAMA
    console.print("\n[dim]Checking Ollama server...[/dim]")

    if not await check_ollama_running(config.ollama_base_url):
        console.print(
            f"[error]Ollama server is not reachable at {config.ollama_base_url}[/error]"
        )
        console.print(
            "[dim]Make sure Ollama is running (ollama serve) and try again.[/dim]"
        )
        sys.exit(1)

    models = await list_ollama_models(config.ollama_base_url)
    if not models:
        console.print("[error]No models found in Ollama.[/error]")
        console.print("[dim]Pull a model first: ollama pull <model-name>[/dim]")
        sys.exit(1)

    console.print(f"\n[bold bright_cyan]  Available Models ({len(models)})[/bold bright_cyan]")
    console.print(Rule(style="grey35"))
    for i, model_name in enumerate(models, 1):
        console.print(f"  [bright_cyan]{i}[/bright_cyan]  {model_name}")

    while True:
        model_choice = console.input(
            "\n[bold bright_cyan]  >[/bold bright_cyan] "
        ).strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
            break
        console.print(
            f"[error]Invalid choice. Enter a number between 1 and {len(models)}.[/error]"
        )

    selected_model = models[int(model_choice) - 1]
    config.model.name = selected_model
    console.print(f"[success]Using Ollama model: {selected_model}[/success]")
    return config


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Current working directory",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["api", "ollama"], case_sensitive=False),
    default=None,
    help="LLM provider: api or ollama",
)
def main(
    prompt: str | None,
    cwd: Path | None,
    provider: str | None,
):
    try:
        config = load_config(cwd=cwd)
    except Exception as e:
        console.print(f"[error]Configuration Error: {e}[/error]")
        sys.exit(1)

    # If provider passed via CLI flag, set it directly
    if provider:
        config.provider = Provider(provider)

    # If no provider flag, show interactive selection
    if not provider:
        config = asyncio.run(select_provider(config))

    errors = config.validate()

    if errors:
        for error in errors:
            console.print(f"[error]{error}[/error]")

        sys.exit(1)

    cli = CLI(config)

    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


main()
