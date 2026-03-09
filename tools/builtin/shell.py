import asyncio
import os
from pathlib import Path
import signal
import sys
from tools.base import Tool, ToolConfirmation, ToolInvocation, ToolKind, ToolResult
from pydantic import BaseModel, Field
import fnmatch

# Unix-specific imports (only available on Unix-like systems)
if sys.platform != "win32":
    import pty
    import select
    import termios
    import tty
    import fcntl
    import struct

DEV_SERVER_PATTERNS = [
    "npm run dev",
    "npm start",
    "yarn dev",
    "yarn start",
    "pnpm dev",
    "pnpm start",
    "bun dev",
    "bun start",
    "next dev",
    "vite",
    "webpack-dev-server",
    "rails server",
    "python manage.py runserver",
    "flask run",
    "streamlit run",
    "uvicorn",
    "gunicorn",
]

BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "mkfs",
    "fdisk",
    "parted",
    ":(){ :|:& };:",  # Fork bomb
    "chmod 777 /",
    "chmod -R 777",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
}


class ShellParams(BaseModel):
    command: str = Field(..., description="The shell command to execute")
    timeout: int = Field(
        120, ge=1, le=600, description="Timeout in seconds (default: 120)"
    )
    cwd: str | None = Field(None, description="Working directory for the command")
    interactive: bool = Field(
        True,
        description="Enable interactive mode with PTY for commands that require user input. Automatically detects and responds to common prompts. Set to False for simple non-interactive commands if needed.",
    )
    run_in_background: bool = Field(
        False,
        description="Run command in background without waiting for completion. Useful for dev servers (npm run dev, etc). Returns immediately with PID.",
    )


class ShellTool(Tool):
    name = "shell"
    kind = ToolKind.SHELL
    description = """Execute a shell command. Use this for running system commands, scripts and CLI tools.

IMPORTANT: For long-running dev servers (npm run dev, vite, next dev, etc), you MUST set run_in_background=true,
otherwise the command will timeout. Background mode starts the process and returns immediately with the PID."""

    schema = ShellParams

    async def get_confirmation(
        self, invocation: ToolInvocation
    ) -> ToolConfirmation | None:
        params = ShellParams(**invocation.params)

        for blocked in BLOCKED_COMMANDS:
            if blocked in params.command:
                return ToolConfirmation(
                    tool_name=self.name,
                    params=invocation.params,
                    description=f"Execute (BLOCKED): {params.command}",
                    command=params.command,
                    is_dangerous=True,
                )

        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute: {params.command}",
            command=params.command,
            is_dangerous=False,
        )

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ShellParams(**invocation.params)

        command = params.command.lower().strip()
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                return ToolResult.error_result(
                    f"Command blocked for safety: {params.command}",
                    metadata={"blocked": True},
                )

        # Check if this looks like a dev server command without background mode
        if not params.run_in_background:
            for pattern in DEV_SERVER_PATTERNS:
                if pattern.lower() in command:
                    warning = f"⚠️  WARNING: This looks like a dev server command that will run indefinitely.\n"
                    warning += f"Consider using run_in_background=true to avoid timeout.\n"
                    warning += f"Proceeding with timeout of {params.timeout}s...\n\n"
                    # We'll continue but add warning to output later
                    break

        if params.cwd:
            cwd = Path(params.cwd)
            if not cwd.is_absolute():
                cwd = invocation.cwd / cwd
        else:
            cwd = invocation.cwd

        if not cwd.exists():
            return ToolResult.error_result(f"Working directory doesn't exist: {cwd}")

        env = self._build_environment()

        # Handle background execution
        if params.run_in_background:
            return await self._execute_background(params, cwd, env)

        # Route to interactive execution if requested
        if params.interactive:
            if sys.platform == "win32":
                return await self._execute_interactive_windows(
                    params, cwd, env
                )
            else:
                return await self._execute_interactive_unix(
                    params, cwd, env
                )

        # Standard non-interactive execution (original behavior)
        if sys.platform == "win32":
            shell_cmd = ["cmd.exe", "/c", params.command]
        else:
            shell_cmd = ["/bin/bash", "-c", params.command]

        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )

        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=params.timeout,
            )
        except asyncio.TimeoutError:
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            await process.wait()
            return ToolResult.error_result(f"Command timed out after {params.timeout}s")

        stdout = stdout_data.decode("utf-8", errors="replace")
        stderr = stderr_data.decode("utf-8", errors="replace")
        exit_code = process.returncode

        output = ""
        if stdout.strip():
            output += stdout.rstrip()

        if stderr.strip():
            output += "\n--- stderr ---\n"
            output += stderr.rstrip()

        if exit_code != 0:
            output += f"\nExit code: {exit_code}"

        if len(output) > 100 * 1024:
            output = output[: 100 * 1024] + "\n... [output truncated]"

        return ToolResult(
            success=exit_code == 0,
            output=output,
            error=stderr if exit_code != 0 else None,
            exit_code=exit_code,
        )

    def _build_environment(self) -> dict[str, str]:
        env = os.environ.copy()

        shell_environment = self.config.shell_environment

        if not shell_environment.ignore_default_excludes:
            for pattern in shell_environment.exclude_patterns:
                keys_to_remove = [
                    k for k in env.keys() if fnmatch.fnmatch(k.upper(), pattern.upper())
                ]

                for k in keys_to_remove:
                    del env[k]

        if shell_environment.set_vars:
            env.update(shell_environment.set_vars)

        return env

    async def _execute_background(
        self, params: ShellParams, cwd: Path, env: dict[str, str]
    ) -> ToolResult:
        """Execute command in background without waiting for completion."""
        shell_cmd = ["/bin/bash", "-c", params.command] if sys.platform != "win32" else ["cmd.exe", "/c", params.command]

        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )

        # Wait a moment to check if process starts successfully
        await asyncio.sleep(0.5)

        # Check if process is still running
        if process.returncode is not None:
            # Process already exited, read output
            stdout_data, stderr_data = await process.communicate()
            stdout = stdout_data.decode("utf-8", errors="replace")
            stderr = stderr_data.decode("utf-8", errors="replace")

            output = f"Process started but exited immediately (exit code: {process.returncode})\n"
            if stdout.strip():
                output += stdout
            if stderr.strip():
                output += "\n--- stderr ---\n" + stderr

            return ToolResult(
                success=False,
                output=output,
                error=stderr if process.returncode != 0 else None,
                exit_code=process.returncode,
            )

        # Process is running in background
        output = f"Process started in background with PID {process.pid}\n"
        output += f"Command: {params.command}\n"
        output += f"Working directory: {cwd}\n"
        output += "\nNote: Process is running in background. Use system tools (ps, kill, etc) to manage it."

        return ToolResult(
            success=True,
            output=output,
            metadata={"pid": process.pid, "background": True},
        )

    async def _execute_interactive_unix(
        self, params: ShellParams, cwd: Path, env: dict[str, str]
    ) -> ToolResult:
        """Execute command with PTY support on Unix systems for interactive input."""

        def run_with_pty():
            """Run command in a PTY (pseudo-terminal) synchronously."""
            # Create a pseudo-terminal
            master_fd, slave_fd = pty.openpty()

            # Set terminal size (80 columns x 24 rows)
            try:
                winsize = struct.pack("HHHH", 24, 80, 0, 0)
                fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
            except Exception:
                pass  # Ignore if terminal size setting fails

            # Fork the process
            pid = os.fork()

            if pid == 0:  # Child process
                try:
                    # Close master fd in child
                    os.close(master_fd)

                    # Make slave the controlling terminal
                    os.setsid()

                    # Redirect stdin, stdout, stderr to slave
                    os.dup2(slave_fd, 0)  # stdin
                    os.dup2(slave_fd, 1)  # stdout
                    os.dup2(slave_fd, 2)  # stderr

                    if slave_fd > 2:
                        os.close(slave_fd)

                    # Change to working directory
                    os.chdir(str(cwd))

                    # Execute the command
                    os.execve(
                        "/bin/bash",
                        ["/bin/bash", "-c", params.command],
                        env,
                    )
                except Exception as e:
                    print(f"Child process error: {e}", file=sys.stderr)
                    os._exit(1)

            # Parent process
            os.close(slave_fd)

            # Set master to non-blocking
            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            output_data = []
            start_time = asyncio.get_event_loop().time()
            timeout = params.timeout

            # Auto-responder for common interactive prompts
            pending_input = b""

            try:
                while True:
                    # Check timeout
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                        raise TimeoutError(f"Command timed out after {timeout}s")

                    # Wait for data with timeout
                    remaining = timeout - elapsed
                    ready, _, _ = select.select([master_fd], [], [], min(remaining, 0.1))

                    if ready:
                        try:
                            data = os.read(master_fd, 4096)
                            if not data:
                                break  # EOF

                            output_data.append(data)

                            # Detect common prompts and auto-respond
                            pending_input += data
                            response = self._detect_and_respond_to_prompt(pending_input)

                            if response:
                                # Send the response
                                os.write(master_fd, response)
                                pending_input = b""  # Clear buffer after response

                        except OSError as e:
                            if e.errno == 5:  # EIO - process terminated
                                break
                            raise

                    # Check if child process has exited
                    pid_result, status = os.waitpid(pid, os.WNOHANG)
                    if pid_result != 0:
                        # Process has exited, read remaining data
                        try:
                            while True:
                                data = os.read(master_fd, 4096)
                                if not data:
                                    break
                                output_data.append(data)
                        except OSError:
                            pass
                        break

                # Get exit code
                if pid_result == 0:
                    _, status = os.waitpid(pid, 0)

                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1

                return b"".join(output_data), exit_code

            finally:
                os.close(master_fd)
                # Ensure child is cleaned up
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except ProcessLookupError:
                    pass  # Already dead

        # Run the PTY execution in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            output_bytes, exit_code = await loop.run_in_executor(None, run_with_pty)
            output = output_bytes.decode("utf-8", errors="replace")

            if len(output) > 100 * 1024:
                output = output[:100 * 1024] + "\n... [output truncated]"

            return ToolResult(
                success=exit_code == 0,
                output=output.rstrip(),
                error=None if exit_code == 0 else f"Exit code: {exit_code}",
                exit_code=exit_code,
                metadata={"interactive": True},
            )
        except TimeoutError as e:
            return ToolResult.error_result(str(e))
        except Exception as e:
            return ToolResult.error_result(f"Interactive execution failed: {e}")

    def _detect_and_respond_to_prompt(self, buffer: bytes) -> bytes | None:
        """Detect interactive prompts and provide automatic responses."""
        try:
            text = buffer.decode("utf-8", errors="ignore").lower()
        except Exception:
            return None

        # Get the last 800 characters for more context
        check_text = text[-800:] if len(text) > 800 else text

        # Get the last line for more precise matching
        lines = check_text.strip().split('\n')
        last_line = lines[-1] if lines else ""

        # Common prompt patterns and their responses
        prompts = [
            # npm create vite specific
            ("project name:", b"\n"),
            ("project name", b"\n"),
            ("select a framework:", b"\n"),
            ("select a framework", b"\n"),
            ("select a variant:", b"\n"),
            ("select a variant", b"\n"),
            ("package name:", b"\n"),
            ("package name", b"\n"),

            # Framework choices - just press enter for default
            ("vanilla", b"\n"),
            ("vue", b"\n"),
            ("react", b"\n"),
            ("preact", b"\n"),
            ("lit", b"\n"),
            ("svelte", b"\n"),
            ("solid", b"\n"),
            ("qwik", b"\n"),

            # npm/pnpm/yarn prompts
            ("which package manager", b"\n"),
            ("install dependencies", b"y\n"),
            ("initialize git", b"y\n"),

            # General y/n prompts
            ("continue?", b"y\n"),
            ("proceed?", b"y\n"),
            ("ok to proceed?", b"y\n"),
            ("is this ok?", b"y\n"),
            ("overwrite", b"n\n"),
            ("(y/n)", b"y\n"),
            ("[y/n]", b"y\n"),
            ("y/n", b"y\n"),

            # Generic prompts
            ("press enter", b"\n"),
            ("press return", b"\n"),
            ("press any key", b"\n"),

            # Arrow selection indicators (modern CLI tools)
            ("›", b"\n"),
            ("❯", b"\n"),
            (">", b"\n"),
        ]

        for pattern, response in prompts:
            if pattern in check_text:
                # Check if last line looks like a prompt
                prompt_endings = ["?", ":", "›", "❯", ">", "]", ")"]

                if any(last_line.rstrip().endswith(ending) for ending in prompt_endings):
                    return response

                # Also respond if we see common prompt indicators
                if any(indicator in last_line for indicator in [":", "?", "›", "❯", ">"]):
                    return response

        return None

    async def _execute_interactive_windows(
        self, params: ShellParams, cwd: Path, env: dict[str, str]
    ) -> ToolResult:
        """Execute command with stdin piping on Windows (PTY not available)."""

        import time

        def run_interactive_windows():
            """Run interactive command on Windows with stdin piping."""
            try:
                # Create process with stdin pipe
                process = asyncio.subprocess.create_subprocess_exec(
                    "cmd.exe", "/c", params.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )

                # Run in new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                proc = loop.run_until_complete(process)

                output_data = []
                pending_buffer = b""
                start_time = time.time()

                async def read_and_respond():
                    nonlocal pending_buffer

                    while True:
                        # Check timeout
                        if time.time() - start_time > params.timeout:
                            proc.kill()
                            raise TimeoutError(f"Command timed out after {params.timeout}s")

                        try:
                            # Try to read with small timeout
                            chunk = await asyncio.wait_for(
                                proc.stdout.read(1024),
                                timeout=0.1
                            )

                            if not chunk:
                                break  # EOF

                            output_data.append(chunk)
                            pending_buffer += chunk

                            # Detect and respond to prompts
                            response = self._detect_and_respond_to_prompt(pending_buffer)
                            if response:
                                proc.stdin.write(response)
                                await proc.stdin.drain()
                                pending_buffer = b""

                        except asyncio.TimeoutError:
                            # No data available, check if process ended
                            if proc.returncode is not None:
                                break
                            continue

                # Read and auto-respond
                loop.run_until_complete(read_and_respond())

                # Wait for process to finish
                loop.run_until_complete(proc.wait())

                exit_code = proc.returncode or 0
                output = b"".join(output_data)

                loop.close()
                return output, exit_code

            except Exception as e:
                raise RuntimeError(f"Windows interactive execution error: {e}")

        # Run in thread pool
        loop = asyncio.get_event_loop()
        try:
            output_bytes, exit_code = await loop.run_in_executor(None, run_interactive_windows)
            output = output_bytes.decode("utf-8", errors="replace")

            if len(output) > 100 * 1024:
                output = output[:100 * 1024] + "\n... [output truncated]"

            return ToolResult(
                success=exit_code == 0,
                output=output.rstrip(),
                error=None if exit_code == 0 else f"Exit code: {exit_code}",
                exit_code=exit_code,
                metadata={"interactive": True, "platform": "windows"},
            )
        except TimeoutError as e:
            return ToolResult.error_result(str(e))
        except Exception as e:
            return ToolResult.error_result(f"Interactive execution failed on Windows: {e}")
