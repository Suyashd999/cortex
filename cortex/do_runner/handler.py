"""Main DoHandler class for the --do functionality."""

import datetime
import os
import shutil
import subprocess
import time
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from .database import DoRunDatabase
from .diagnosis import AutoFixer, ErrorDiagnoser, LoginHandler
from .managers import CortexUserManager, ProtectedPathsManager
from .models import (
    CommandLog,
    CommandStatus,
    DoRun,
    RunMode,
    TaskNode,
    TaskTree,
)
from .terminal import TerminalMonitor
from .verification import (
    ConflictDetector,
    FileUsefulnessAnalyzer,
    VerificationRunner,
)

console = Console()


class DoHandler:
    """Main handler for the --do functionality."""
    
    def __init__(self, llm_callback: Callable[[str], dict] | None = None):
        self.db = DoRunDatabase()
        self.paths_manager = ProtectedPathsManager()
        self.user_manager = CortexUserManager
        self.current_run: DoRun | None = None
        self._granted_privileges: list[str] = []
        self.llm_callback = llm_callback
        
        self._task_tree: TaskTree | None = None
        self._permission_requests_count = 0
        
        self._terminal_monitor: TerminalMonitor | None = None
        
        # Session tracking
        self.current_session_id: str | None = None
        
        # Initialize helper classes
        self._diagnoser = ErrorDiagnoser()
        self._auto_fixer = AutoFixer(llm_callback=llm_callback)
        self._login_handler = LoginHandler()
        self._conflict_detector = ConflictDetector()
        self._verification_runner = VerificationRunner()
        self._file_analyzer = FileUsefulnessAnalyzer()
        
        # Initialize notification manager
        try:
            from cortex.notification_manager import NotificationManager
            self.notifier = NotificationManager()
        except ImportError:
            self.notifier = None
    
    def _send_notification(self, title: str, message: str, level: str = "normal"):
        """Send a desktop notification."""
        if self.notifier:
            self.notifier.send(title, message, level=level)
        else:
            console.print(f"[bold yellow]🔔 {title}:[/bold yellow] {message}")
    
    def setup_cortex_user(self) -> bool:
        """Ensure the cortex user exists."""
        if not self.user_manager.user_exists():
            console.print("[yellow]Setting up cortex user...[/yellow]")
            success, message = self.user_manager.create_user()
            if success:
                console.print(f"[green]✓ {message}[/green]")
            else:
                console.print(f"[red]✗ {message}[/red]")
            return success
        return True
    
    def analyze_commands_for_protected_paths(
        self, 
        commands: list[tuple[str, str]]
    ) -> list[tuple[str, str, list[str]]]:
        """Analyze commands and identify protected paths they access."""
        results = []
        
        for command, purpose in commands:
            protected = []
            parts = command.split()
            for part in parts:
                if part.startswith("/") or part.startswith("~"):
                    path = os.path.expanduser(part)
                    if self.paths_manager.is_protected(path):
                        protected.append(path)
            
            results.append((command, purpose, protected))
        
        return results
    
    def request_user_confirmation(
        self,
        commands: list[tuple[str, str, list[str]]],
    ) -> bool:
        """Show commands to user and request confirmation."""
        console.print()
        console.print(Panel(
            "[bold yellow]⚠️  Cortex wants to execute the following commands[/bold yellow]",
            expand=False,
        ))
        console.print()
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Command", style="green")
        table.add_column("Purpose", style="white")
        table.add_column("Protected Paths", style="yellow")
        
        for i, (cmd, purpose, protected) in enumerate(commands, 1):
            protected_str = ", ".join(protected) if protected else "-"
            table.add_row(str(i), cmd, purpose, protected_str)
        
        console.print(table)
        console.print()
        
        all_protected = []
        for _, _, protected in commands:
            all_protected.extend(protected)
        
        if all_protected:
            console.print("[bold red]⚠️  These commands will access protected system paths![/bold red]")
            console.print(f"[dim]Protected paths: {', '.join(set(all_protected))}[/dim]")
            console.print()
        
        return Confirm.ask("[bold]Do you want to proceed?[/bold]", default=False)
    
    def _needs_sudo(self, cmd: str, protected_paths: list[str]) -> bool:
        """Determine if a command needs sudo to execute."""
        sudo_commands = [
            "systemctl", "service", "apt", "apt-get", "dpkg",
            "mount", "umount", "fdisk", "mkfs", "chown", "chmod",
            "useradd", "userdel", "usermod", "groupadd", "groupdel",
        ]
        
        cmd_parts = cmd.split()
        if not cmd_parts:
            return False
        
        base_cmd = cmd_parts[0]
        
        if base_cmd in sudo_commands:
            return True
        
        if protected_paths:
            return True
        
        if any(p in cmd for p in ["/etc/", "/var/lib/", "/usr/", "/opt/", "/root/"]):
            return True
        
        return False
    
    # Commands that benefit from streaming output (long-running with progress)
    STREAMING_COMMANDS = [
        "docker pull", "docker push", "docker build",
        "apt install", "apt-get install", "apt update", "apt-get update", "apt upgrade", "apt-get upgrade",
        "pip install", "pip3 install", "pip download", "pip3 download",
        "npm install", "npm ci", "yarn install", "yarn add",
        "cargo build", "cargo install",
        "go build", "go install", "go get",
        "gem install", "bundle install",
        "wget", "curl -o", "curl -O",
        "git clone", "git pull", "git fetch",
        "make", "cmake", "ninja",
        "rsync", "scp",
    ]
    
    # Interactive commands that need a TTY - cannot be run in background/automated
    INTERACTIVE_COMMANDS = [
        "docker exec -it", "docker exec -ti", "docker run -it", "docker run -ti",
        "docker attach",
        "ollama run", "ollama chat",
        "ssh ", 
        "bash -i", "sh -i", "zsh -i",
        "vi ", "vim ", "nano ", "emacs ",
        "python -i", "python3 -i", "ipython", "node -i",
        "mysql -u", "psql -U", "mongo ", "redis-cli",
        "htop", "top -i", "less ", "more ",
    ]
    
    def _should_stream_output(self, cmd: str) -> bool:
        """Check if command should use streaming output."""
        cmd_lower = cmd.lower()
        return any(streaming_cmd in cmd_lower for streaming_cmd in self.STREAMING_COMMANDS)
    
    def _is_interactive_command(self, cmd: str) -> bool:
        """Check if command requires interactive TTY and cannot be automated."""
        cmd_lower = cmd.lower()
        # Check explicit patterns
        if any(interactive in cmd_lower for interactive in self.INTERACTIVE_COMMANDS):
            return True
        # Check for -it or -ti flags in docker commands
        if "docker" in cmd_lower and (" -it " in cmd_lower or " -ti " in cmd_lower or 
                                       cmd_lower.endswith(" -it") or cmd_lower.endswith(" -ti")):
            return True
        return False
    
    # Timeout settings by command type (in seconds)
    COMMAND_TIMEOUTS = {
        "docker pull": 1800,      # 30 minutes for large images
        "docker push": 1800,      # 30 minutes for large images
        "docker build": 3600,     # 1 hour for complex builds
        "apt install": 900,       # 15 minutes
        "apt-get install": 900,
        "apt update": 300,        # 5 minutes
        "apt-get update": 300,
        "apt upgrade": 1800,      # 30 minutes
        "apt-get upgrade": 1800,
        "pip install": 600,       # 10 minutes
        "pip3 install": 600,
        "npm install": 900,       # 15 minutes
        "yarn install": 900,
        "git clone": 600,         # 10 minutes
        "make": 1800,             # 30 minutes
        "cargo build": 1800,
    }
    
    def _get_command_timeout(self, cmd: str) -> int:
        """Get appropriate timeout for a command."""
        cmd_lower = cmd.lower()
        for cmd_pattern, timeout in self.COMMAND_TIMEOUTS.items():
            if cmd_pattern in cmd_lower:
                return timeout
        return 600  # Default 10 minutes for streaming commands
    
    def _execute_with_streaming(
        self,
        cmd: str,
        needs_sudo: bool,
        timeout: int | None = None,  # None = auto-detect
    ) -> tuple[bool, str, str]:
        """Execute a command with real-time output streaming."""
        import select
        import sys
        
        # Auto-detect timeout if not specified
        if timeout is None:
            timeout = self._get_command_timeout(cmd)
        
        # Show timeout info for long operations
        if timeout > 300:
            console.print(f"[dim]      ⏱️  Timeout: {timeout // 60} minutes (large operation)[/dim]")
        
        stdout_lines = []
        stderr_lines = []
        
        try:
            if needs_sudo:
                process = subprocess.Popen(
                    ["sudo", "bash", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            
            # Use select for non-blocking reads on both stdout and stderr
            import time
            start_time = time.time()
            
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    return False, "\n".join(stdout_lines), f"Command timed out after {timeout} seconds"
                
                # Check if process has finished
                if process.poll() is not None:
                    # Read any remaining output
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        for line in remaining_stdout.splitlines():
                            stdout_lines.append(line)
                            self._print_progress_line(line, is_stderr=False)
                    if remaining_stderr:
                        for line in remaining_stderr.splitlines():
                            stderr_lines.append(line)
                            self._print_progress_line(line, is_stderr=True)
                    break
                
                # Try to read from stdout/stderr without blocking
                try:
                    readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                    
                    for stream in readable:
                        line = stream.readline()
                        if line:
                            line = line.rstrip()
                            if stream == process.stdout:
                                stdout_lines.append(line)
                                self._print_progress_line(line, is_stderr=False)
                            else:
                                stderr_lines.append(line)
                                self._print_progress_line(line, is_stderr=True)
                except (ValueError, OSError):
                    # Stream closed
                    break
            
            return (
                process.returncode == 0,
                "\n".join(stdout_lines).strip(),
                "\n".join(stderr_lines).strip(),
            )
            
        except Exception as e:
            return False, "\n".join(stdout_lines), str(e)
    
    def _print_progress_line(self, line: str, is_stderr: bool = False) -> None:
        """Print a progress line with appropriate formatting."""
        if not line.strip():
            return
        
        line = line.strip()
        
        # Docker pull progress patterns
        if any(p in line for p in ["Pulling from", "Digest:", "Status:", "Pull complete", "Downloading", "Extracting"]):
            console.print(f"[dim]      📦 {line}[/dim]")
        # Docker build progress
        elif line.startswith("Step ") or line.startswith("---> "):
            console.print(f"[dim]      🔨 {line}[/dim]")
        # apt progress patterns
        elif any(p in line for p in ["Get:", "Hit:", "Fetched", "Reading", "Building", "Setting up", "Processing", "Unpacking"]):
            console.print(f"[dim]      📦 {line}[/dim]")
        # pip progress patterns
        elif any(p in line for p in ["Collecting", "Downloading", "Installing", "Successfully"]):
            console.print(f"[dim]      📦 {line}[/dim]")
        # npm progress patterns
        elif any(p in line for p in ["npm", "added", "packages", "audited"]):
            console.print(f"[dim]      📦 {line}[/dim]")
        # git progress patterns
        elif any(p in line for p in ["Cloning", "remote:", "Receiving", "Resolving", "Checking out"]):
            console.print(f"[dim]      📦 {line}[/dim]")
        # wget/curl progress
        elif "%" in line and any(c.isdigit() for c in line):
            # Progress percentage - update in place
            console.print(f"[dim]      ⬇️  {line[:80]}[/dim]", end="\r")
        # Error lines
        elif is_stderr and any(p in line.lower() for p in ["error", "fail", "denied", "cannot", "unable"]):
            console.print(f"[yellow]      ⚠ {line}[/yellow]")
        # Truncate very long lines
        elif len(line) > 100:
            console.print(f"[dim]      {line[:100]}...[/dim]")
    
    def _execute_single_command(
        self, 
        cmd: str, 
        needs_sudo: bool,
        timeout: int = 120
    ) -> tuple[bool, str, str]:
        """Execute a single command with proper privilege handling."""
        # Check for interactive commands that need a TTY
        if self._is_interactive_command(cmd):
            return self._handle_interactive_command(cmd, needs_sudo)
        
        # Use streaming for long-running commands
        if self._should_stream_output(cmd):
            return self._execute_with_streaming(cmd, needs_sudo, timeout=300)
        
        try:
            if needs_sudo:
                result = subprocess.run(
                    ["sudo", "bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            return (
                result.returncode == 0,
                result.stdout.strip(),
                result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def _handle_interactive_command(
        self, 
        cmd: str, 
        needs_sudo: bool
    ) -> tuple[bool, str, str]:
        """Handle interactive commands that need a TTY.
        
        These commands cannot be run in the background - they need user interaction.
        We'll either:
        1. Try to open in a new terminal window
        2. Or inform the user to run it manually
        """
        console.print()
        console.print(f"[yellow]⚡ Interactive command detected[/yellow]")
        console.print(f"[dim]   This command requires a terminal for interaction.[/dim]")
        console.print()
        
        full_cmd = f"sudo {cmd}" if needs_sudo else cmd
        
        # Try to detect if we can open a new terminal
        terminal_cmds = [
            ("gnome-terminal", f'gnome-terminal -- bash -c "{full_cmd}; echo; echo Press Enter to close...; read"'),
            ("konsole", f'konsole -e bash -c "{full_cmd}; echo; echo Press Enter to close...; read"'),
            ("xterm", f'xterm -e bash -c "{full_cmd}; echo; echo Press Enter to close...; read"'),
            ("x-terminal-emulator", f'x-terminal-emulator -e bash -c "{full_cmd}; echo; echo Press Enter to close...; read"'),
        ]
        
        # Check which terminal is available
        for term_name, term_cmd in terminal_cmds:
            if shutil.which(term_name):
                console.print(f"[cyan]🖥️  Opening in new terminal window ({term_name})...[/cyan]")
                console.print(f"[dim]   Command: {full_cmd}[/dim]")
                console.print()
                
                try:
                    # Start the terminal in background
                    subprocess.Popen(
                        term_cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True, f"Command opened in new {term_name} window", ""
                except Exception as e:
                    console.print(f"[yellow]   ⚠ Could not open terminal: {e}[/yellow]")
                    break
        
        # Fallback: ask user to run manually
        console.print(f"[bold cyan]📋 Please run this command manually in another terminal:[/bold cyan]")
        console.print()
        console.print(f"   [green]{full_cmd}[/green]")
        console.print()
        console.print(f"[dim]   This command needs interactive input (TTY).[/dim]")
        console.print(f"[dim]   Cortex cannot capture its output automatically.[/dim]")
        console.print()
        
        # Return special status indicating manual run needed
        return True, "INTERACTIVE_COMMAND_MANUAL", f"Interactive command - run manually: {full_cmd}"
    
    def execute_commands_as_cortex(
        self,
        commands: list[tuple[str, str, list[str]]],
        user_query: str,
    ) -> DoRun:
        """Execute commands with granular error handling and auto-recovery."""
        run = DoRun(
            run_id=self.db._generate_run_id(),
            summary="",
            mode=RunMode.CORTEX_EXEC,
            user_query=user_query,
            started_at=datetime.datetime.now().isoformat(),
            session_id=self.current_session_id or "",
        )
        self.current_run = run
        
        console.print()
        console.print("[bold cyan]🚀 Executing commands with conflict detection...[/bold cyan]")
        console.print()
        
        # Phase 1: Conflict Detection
        console.print("[dim]Checking for conflicts...[/dim]")
        
        cleanup_commands = []
        for cmd, purpose, protected in commands:
            conflict = self._conflict_detector.check_for_conflicts(cmd, purpose)
            if conflict["has_conflict"]:
                console.print(f"[yellow]   ⚠ {conflict['conflict_type']}: {conflict['suggestion']}[/yellow]")
                if conflict["cleanup_commands"]:
                    cleanup_commands.extend(conflict["cleanup_commands"])
        
        if cleanup_commands:
            console.print("[dim]Running cleanup commands...[/dim]")
            for cleanup_cmd in cleanup_commands:
                self._execute_single_command(cleanup_cmd, needs_sudo=True)
        
        console.print()
        
        all_protected = set()
        for _, _, protected in commands:
            all_protected.update(protected)
        
        if all_protected:
            console.print(f"[dim]📁 Protected paths involved: {', '.join(all_protected)}[/dim]")
            console.print()
        
        # Phase 2: Execute Commands
        for i, (cmd, purpose, protected) in enumerate(commands, 1):
            console.print(f"[bold][{i}/{len(commands)}][/bold] {cmd}")
            console.print(f"[dim]   └─ {purpose}[/dim]")
            
            file_check = self._file_analyzer.check_file_exists_and_usefulness(cmd, purpose, user_query)
            
            if file_check["recommendations"]:
                self._file_analyzer.apply_file_recommendations(file_check["recommendations"])
            
            cmd_log = CommandLog(
                command=cmd,
                purpose=purpose,
                timestamp=datetime.datetime.now().isoformat(),
                status=CommandStatus.RUNNING,
            )
            
            start_time = time.time()
            needs_sudo = self._needs_sudo(cmd, protected)
            
            success, stdout, stderr = self._execute_single_command(cmd, needs_sudo)
            
            if not success:
                diagnosis = self._diagnoser.diagnose_error(cmd, stderr)
                
                console.print(f"[yellow]   ⚠ Error: {diagnosis['description']}[/yellow]")
                console.print(f"[dim]   Error type: {diagnosis['error_type']} | Category: {diagnosis.get('category', 'unknown')}[/dim]")
                
                # Check if this is a login/credential required error
                if diagnosis.get("category") == "login_required":
                    console.print(f"[cyan]   🔐 Authentication required for this command[/cyan]")
                    
                    login_success, login_msg = self._login_handler.handle_login(cmd, stderr)
                    
                    if login_success:
                        console.print(f"[green]   ✓ {login_msg}[/green]")
                        console.print(f"[cyan]   Retrying command...[/cyan]")
                        
                        # Retry the command after successful login
                        success, stdout, stderr = self._execute_single_command(cmd, needs_sudo)
                        
                        if success:
                            console.print(f"[green]   ✓ Command succeeded after authentication![/green]")
                        else:
                            console.print(f"[yellow]   Command still failed after login: {stderr[:100]}[/yellow]")
                    else:
                        console.print(f"[yellow]   {login_msg}[/yellow]")
                else:
                    # Not a login error, proceed with regular error handling
                    if diagnosis.get("extracted_path"):
                        console.print(f"[dim]   Problematic path: {diagnosis['extracted_path']}[/dim]")
                    if diagnosis.get("extracted_info"):
                        for key, value in diagnosis["extracted_info"].items():
                            if value:
                                console.print(f"[dim]   {key}: {value}[/dim]")
                    
                    fixed, fix_message, fix_commands = self._auto_fixer.auto_fix_error(
                        cmd, stderr, diagnosis, max_attempts=3
                    )
                    
                    if fixed:
                        success = True
                        console.print(f"[green]   ✓ Auto-fixed: {fix_message}[/green]")
                        _, stdout, stderr = self._execute_single_command(cmd, needs_sudo=True)
                    else:
                        if fix_commands:
                            console.print(f"[yellow]   Attempted fixes: {len(fix_commands)} command(s)[/yellow]")
                        console.print(f"[yellow]   Result: {fix_message}[/yellow]")
            
            cmd_log.duration_seconds = time.time() - start_time
            cmd_log.output = stdout
            cmd_log.error = stderr
            cmd_log.status = CommandStatus.SUCCESS if success else CommandStatus.FAILED
            
            run.commands.append(cmd_log)
            run.files_accessed.extend(protected)
            
            if success:
                console.print(f"[green]   ✓ Success ({cmd_log.duration_seconds:.2f}s)[/green]")
                if stdout:
                    output_lines = stdout.split('\n')
                    if len(output_lines) > 3:
                        display_output = '\n'.join(output_lines[:3]) + f"\n   ... ({len(output_lines) - 3} more lines)"
                    else:
                        display_output = stdout[:200] + ('...' if len(stdout) > 200 else '')
                    console.print(f"[dim]   {display_output}[/dim]")
            else:
                console.print(f"[red]   ✗ Failed: {stderr[:200]}[/red]")
                
                final_diagnosis = self._diagnoser.diagnose_error(cmd, stderr)
                if final_diagnosis["fix_commands"] and not final_diagnosis["can_auto_fix"]:
                    console.print(f"[yellow]   💡 Manual intervention required:[/yellow]")
                    console.print(f"[yellow]      Issue: {final_diagnosis['description']}[/yellow]")
                    
                    console.print(f"[yellow]   Suggested commands to try:[/yellow]")
                    for fix_cmd in final_diagnosis["fix_commands"]:
                        if not fix_cmd.startswith("#"):
                            console.print(f"[cyan]      $ {fix_cmd}[/cyan]")
                        else:
                            console.print(f"[dim]      {fix_cmd}[/dim]")
            
            console.print()
        
        self._granted_privileges = []
        
        # Phase 3: Verification Tests
        console.print("[bold]Running verification tests...[/bold]")
        all_tests_passed, test_results = self._verification_runner.run_verification_tests(run.commands, user_query)
        
        # Phase 4: Auto-repair if tests failed
        if not all_tests_passed:
            console.print()
            console.print("[bold yellow]🔧 Auto-repairing test failures...[/bold yellow]")
            
            repair_success = self._handle_test_failures(test_results, run)
            
            if repair_success:
                console.print("[dim]Re-running verification tests...[/dim]")
                all_tests_passed, test_results = self._verification_runner.run_verification_tests(run.commands, user_query)
        
        run.completed_at = datetime.datetime.now().isoformat()
        run.summary = self._generate_summary(run)
        
        if test_results:
            passed = sum(1 for t in test_results if t["passed"])
            run.summary += f" | Tests: {passed}/{len(test_results)} passed"
        
        self.db.save_run(run)
        
        console.print()
        if all_tests_passed:
            console.print(f"[bold green]✓ Run completed successfully[/bold green] (ID: {run.run_id})")
        else:
            console.print(f"[bold yellow]⚠ Run completed with issues[/bold yellow] (ID: {run.run_id})")
        console.print(f"[dim]Summary: {run.summary}[/dim]")
        
        return run
    
    def _handle_resource_conflict(
        self,
        idx: int,
        cmd: str,
        conflict: dict,
        commands_to_skip: set,
        cleanup_commands: list,
    ) -> bool:
        """Handle any resource conflict with user options.
        
        This is a GENERAL handler for all resource types:
        - Docker containers
        - Services
        - Files/directories  
        - Packages
        - Ports
        - Users/groups
        - Virtual environments
        - Databases
        - Cron jobs
        """
        resource_type = conflict.get("resource_type", "resource")
        resource_name = conflict.get("resource_name", "unknown")
        conflict_type = conflict.get("conflict_type", "unknown")
        suggestion = conflict.get("suggestion", "")
        is_active = conflict.get("is_active", True)
        alternatives = conflict.get("alternative_actions", [])
        
        # Resource type icons
        icons = {
            "container": "🐳",
            "compose": "🐳",
            "service": "⚙️",
            "file": "📄",
            "directory": "📁",
            "package": "📦",
            "pip_package": "🐍",
            "npm_package": "📦",
            "port": "🔌",
            "user": "👤",
            "group": "👥",
            "venv": "🐍",
            "mysql_database": "🗄️",
            "postgres_database": "🗄️",
            "cron_job": "⏰",
        }
        icon = icons.get(resource_type, "📌")
        
        # Display the conflict
        console.print()
        if is_active:
            console.print(f"[cyan]   {icon} {resource_type.replace('_', ' ').title()} '{resource_name}' already exists![/cyan]")
        else:
            console.print(f"[yellow]   {icon} {resource_type.replace('_', ' ').title()} '{resource_name}' exists (inactive)[/yellow]")
        console.print(f"[dim]   {suggestion}[/dim]")
        
        # If there are alternatives, show them
        if alternatives:
            console.print()
            console.print("   [bold]What would you like to do?[/bold]")
            for j, alt in enumerate(alternatives, 1):
                console.print(f"   {j}. {alt['description']}")
            console.print()
            
            from rich.prompt import Prompt
            choice = Prompt.ask(
                "   Choose an option",
                choices=[str(k) for k in range(1, len(alternatives) + 1)],
                default="1"
            )
            
            selected = alternatives[int(choice) - 1]
            action = selected["action"]
            action_commands = selected.get("commands", [])
            
            # Handle different actions
            if action in ["use_existing", "use_different"]:
                console.print(f"[green]   ✓ Using existing {resource_type} '{resource_name}'[/green]")
                commands_to_skip.add(idx)
                return True
                
            elif action == "start_existing":
                console.print(f"[cyan]   Starting existing {resource_type}...[/cyan]")
                for start_cmd in action_commands:
                    needs_sudo = start_cmd.startswith("sudo")
                    success, _, stderr = self._execute_single_command(start_cmd, needs_sudo=needs_sudo)
                    if success:
                        console.print(f"[green]   ✓ {start_cmd}[/green]")
                    else:
                        console.print(f"[red]   ✗ {start_cmd}: {stderr[:50]}[/red]")
                commands_to_skip.add(idx)
                return True
                
            elif action in ["restart", "upgrade", "reinstall"]:
                console.print(f"[cyan]   {action.title()}ing {resource_type}...[/cyan]")
                for action_cmd in action_commands:
                    needs_sudo = action_cmd.startswith("sudo")
                    success, _, stderr = self._execute_single_command(action_cmd, needs_sudo=needs_sudo)
                    if success:
                        console.print(f"[green]   ✓ {action_cmd}[/green]")
                    else:
                        console.print(f"[red]   ✗ {action_cmd}: {stderr[:50]}[/red]")
                commands_to_skip.add(idx)
                return True
                
            elif action in ["recreate", "backup", "replace", "stop_existing"]:
                console.print(f"[cyan]   Preparing to {action.replace('_', ' ')}...[/cyan]")
                for action_cmd in action_commands:
                    needs_sudo = action_cmd.startswith("sudo")
                    success, _, stderr = self._execute_single_command(action_cmd, needs_sudo=needs_sudo)
                    if success:
                        console.print(f"[green]   ✓ {action_cmd}[/green]")
                    else:
                        console.print(f"[red]   ✗ {action_cmd}: {stderr[:50]}[/red]")
                # Don't skip - let the original command run after cleanup
                return True
                
            elif action == "modify":
                console.print(f"[cyan]   Will modify existing {resource_type}[/cyan]")
                # Don't skip - let the original command run to modify
                return True
            
            elif action == "install_first":
                # Install a missing tool/dependency first
                console.print(f"[cyan]   Installing required dependency '{resource_name}'...[/cyan]")
                all_success = True
                for action_cmd in action_commands:
                    needs_sudo = action_cmd.startswith("sudo")
                    success, stdout, stderr = self._execute_single_command(action_cmd, needs_sudo=needs_sudo)
                    if success:
                        console.print(f"[green]   ✓ {action_cmd}[/green]")
                    else:
                        console.print(f"[red]   ✗ {action_cmd}: {stderr[:50]}[/red]")
                        all_success = False
                
                if all_success:
                    console.print(f"[green]   ✓ '{resource_name}' installed. Continuing with original command...[/green]")
                    # Don't skip - run the original command now that the tool is installed
                    return True
                else:
                    console.print(f"[red]   ✗ Failed to install '{resource_name}'[/red]")
                    commands_to_skip.add(idx)
                    return True
            
            elif action == "use_apt":
                # User chose to use apt instead of snap
                console.print(f"[cyan]   Skipping snap command - use apt instead[/cyan]")
                commands_to_skip.add(idx)
                return True
            
            elif action == "refresh":
                # Refresh snap package
                console.print(f"[cyan]   Refreshing snap package...[/cyan]")
                for action_cmd in action_commands:
                    needs_sudo = action_cmd.startswith("sudo")
                    success, _, stderr = self._execute_single_command(action_cmd, needs_sudo=needs_sudo)
                    if success:
                        console.print(f"[green]   ✓ {action_cmd}[/green]")
                    else:
                        console.print(f"[red]   ✗ {action_cmd}: {stderr[:50]}[/red]")
                commands_to_skip.add(idx)
                return True
        
        # No alternatives - use default behavior (add to cleanup if available)
        if conflict.get("cleanup_commands"):
            cleanup_commands.extend(conflict["cleanup_commands"])
        
        return False
    
    def _handle_test_failures(
        self,
        test_results: list[dict[str, Any]],
        run: DoRun,
    ) -> bool:
        """Handle failed verification tests by attempting auto-repair."""
        failed_tests = [t for t in test_results if not t["passed"]]
        
        if not failed_tests:
            return True
        
        console.print()
        console.print("[bold yellow]🔧 Attempting to fix test failures...[/bold yellow]")
        
        all_fixed = True
        
        for test in failed_tests:
            test_name = test["test"]
            output = test["output"]
            
            console.print(f"[dim]   Fixing: {test_name}[/dim]")
            
            if "nginx -t" in test_name:
                diagnosis = self._diagnoser.diagnose_error("nginx -t", output)
                fixed, msg, _ = self._auto_fixer.auto_fix_error("nginx -t", output, diagnosis, max_attempts=3)
                if fixed:
                    console.print(f"[green]   ✓ Fixed: {msg}[/green]")
                else:
                    console.print(f"[red]   ✗ Could not fix: {msg}[/red]")
                    all_fixed = False
            
            elif "apache2ctl" in test_name:
                diagnosis = self._diagnoser.diagnose_error("apache2ctl configtest", output)
                fixed, msg, _ = self._auto_fixer.auto_fix_error("apache2ctl configtest", output, diagnosis, max_attempts=3)
                if fixed:
                    console.print(f"[green]   ✓ Fixed: {msg}[/green]")
                else:
                    all_fixed = False
            
            elif "systemctl is-active" in test_name:
                import re
                svc_match = re.search(r'is-active\s+(\S+)', test_name)
                if svc_match:
                    service = svc_match.group(1)
                    success, _, err = self._execute_single_command(
                        f"sudo systemctl start {service}", needs_sudo=True
                    )
                    if success:
                        console.print(f"[green]   ✓ Started service {service}[/green]")
                    else:
                        console.print(f"[yellow]   ⚠ Could not start {service}: {err[:50]}[/yellow]")
            
            elif "file exists" in test_name:
                import re
                path_match = re.search(r'file exists: (.+)', test_name)
                if path_match:
                    path = path_match.group(1)
                    parent = os.path.dirname(path)
                    if parent and not os.path.exists(parent):
                        self._execute_single_command(f"sudo mkdir -p {parent}", needs_sudo=True)
                        console.print(f"[green]   ✓ Created directory {parent}[/green]")
        
        return all_fixed
    
    def execute_with_task_tree(
        self,
        commands: list[tuple[str, str, list[str]]],
        user_query: str,
    ) -> DoRun:
        """Execute commands using the task tree system with advanced auto-repair."""
        run = DoRun(
            run_id=self.db._generate_run_id(),
            summary="",
            mode=RunMode.CORTEX_EXEC,
            user_query=user_query,
            started_at=datetime.datetime.now().isoformat(),
            session_id=self.current_session_id or "",
        )
        self.current_run = run
        self._permission_requests_count = 0
        
        self._task_tree = TaskTree()
        for cmd, purpose, protected in commands:
            task = self._task_tree.add_root_task(cmd, purpose)
            task.reasoning = f"Protected paths: {', '.join(protected)}" if protected else ""
        
        console.print()
        console.print(Panel(
            "[bold cyan]🌳 Task Tree Execution Mode[/bold cyan]\n"
            "[dim]Commands will be executed with auto-repair capabilities.[/dim]\n"
            "[dim]Conflict detection and verification tests enabled.[/dim]",
            expand=False,
        ))
        console.print()
        
        # Phase 1: Conflict Detection
        console.print("[bold]📋 Phase 1: Checking for conflicts...[/bold]")
        
        conflicts_found = []
        cleanup_commands = []
        commands_to_skip = set()  # Track commands that should be skipped (use existing)
        commands_to_replace = {}  # Track commands that should be replaced
        
        for i, (cmd, purpose, protected) in enumerate(commands):
            conflict = self._conflict_detector.check_for_conflicts(cmd, purpose)
            if conflict["has_conflict"]:
                conflicts_found.append((i, cmd, conflict))
        
        if conflicts_found:
            console.print(f"[yellow]   Found {len(conflicts_found)} potential conflict(s)[/yellow]")
            
            for idx, cmd, conflict in conflicts_found:
                # Handle any resource conflict with alternatives
                handled = self._handle_resource_conflict(idx, cmd, conflict, commands_to_skip, cleanup_commands)
            
            # Run cleanup commands for non-Docker conflicts
            if cleanup_commands:
                console.print("[dim]   Running cleanup commands...[/dim]")
                for cleanup_cmd in cleanup_commands:
                    self._execute_single_command(cleanup_cmd, needs_sudo=True)
                    console.print(f"[dim]   ✓ {cleanup_cmd}[/dim]")
            
            # Filter out skipped commands
            if commands_to_skip:
                filtered_commands = [
                    (cmd, purpose, protected) 
                    for i, (cmd, purpose, protected) in enumerate(commands) 
                    if i not in commands_to_skip
                ]
                # Update task tree to skip these tasks
                for task in self._task_tree.root_tasks:
                    task_idx = next(
                        (i for i, (c, p, pr) in enumerate(commands) if c == task.command),
                        None
                    )
                    if task_idx in commands_to_skip:
                        task.status = CommandStatus.SKIPPED
                        task.output = "Using existing resource"
                commands = filtered_commands
        else:
            console.print("[green]   ✓ No conflicts detected[/green]")
        
        console.print()
        
        all_protected = set()
        for _, _, protected in commands:
            all_protected.update(protected)
        
        if all_protected:
            console.print(f"[dim]📁 Protected paths: {', '.join(all_protected)}[/dim]")
            console.print()
        
        # Phase 2: Execute Commands
        console.print("[bold]🚀 Phase 2: Executing commands...[/bold]")
        console.print()
        
        for root_task in self._task_tree.root_tasks:
            self._execute_task_node(root_task, run, commands)
        
        # Phase 3: Verification Tests
        console.print()
        console.print("[bold]🧪 Phase 3: Running verification tests...[/bold]")
        
        all_tests_passed, test_results = self._verification_runner.run_verification_tests(run.commands, user_query)
        
        # Phase 4: Auto-repair if tests failed
        if not all_tests_passed:
            console.print()
            console.print("[bold]🔧 Phase 4: Auto-repairing test failures...[/bold]")
            
            repair_success = self._handle_test_failures(test_results, run)
            
            if repair_success:
                console.print()
                console.print("[dim]   Re-running verification tests...[/dim]")
                all_tests_passed, test_results = self._verification_runner.run_verification_tests(run.commands, user_query)
        
        run.completed_at = datetime.datetime.now().isoformat()
        run.summary = self._generate_tree_summary(run)
        
        if test_results:
            passed = sum(1 for t in test_results if t["passed"])
            run.summary += f" | Tests: {passed}/{len(test_results)} passed"
        
        self.db.save_run(run)
        
        console.print()
        console.print("[bold]Task Execution Tree:[/bold]")
        self._task_tree.print_tree()
        
        console.print()
        if all_tests_passed:
            console.print(f"[bold green]✓ Run completed successfully[/bold green] (ID: {run.run_id})")
        else:
            console.print(f"[bold yellow]⚠ Run completed with issues[/bold yellow] (ID: {run.run_id})")
        console.print(f"[dim]Summary: {run.summary}[/dim]")
        
        if self._permission_requests_count > 1:
            console.print(f"[dim]Permission requests made: {self._permission_requests_count}[/dim]")
        
        # Interactive session - suggest next steps
        self._interactive_session(run, commands, user_query)
        
        return run
    
    def _interactive_session(
        self,
        run: DoRun,
        commands: list[tuple[str, str, list[str]]],
        user_query: str,
    ) -> None:
        """Interactive session after task completion - suggest next steps."""
        from rich.prompt import Prompt
        
        # Generate context-aware suggestions based on what was done
        suggestions = self._generate_suggestions(run, commands, user_query)
        
        # Track context for natural language processing
        context = {
            "original_query": user_query,
            "executed_commands": [cmd for cmd, _, _ in commands],
            "session_actions": [],
        }
        
        console.print()
        console.print("[bold cyan]━━━ What would you like to do next? ━━━[/bold cyan]")
        console.print()
        
        # Display suggestions
        self._display_suggestions(suggestions)
        
        console.print()
        console.print("[dim]💡 Tip: You can type any request in natural language, like:[/dim]")
        console.print("[dim]   • 'run the container on port 3000'[/dim]")
        console.print("[dim]   • 'show me how to connect to it'[/dim]")
        console.print("[dim]   • 'install another package called redis'[/dim]")
        console.print()
        
        while True:
            try:
                response = Prompt.ask(
                    "[bold]What would you like to do?[/bold]",
                    default="exit"
                )
                
                response_stripped = response.strip()
                response_lower = response_stripped.lower()
                
                # Check for exit keywords
                if response_lower in ["exit", "quit", "done", "no", "n", "bye", "thanks", "nothing", ""]:
                    console.print("[dim]👋 Session ended. Run 'cortex do history' to see past runs.[/dim]")
                    break
                
                # Try to parse as number (for suggestion selection)
                try:
                    choice = int(response_stripped)
                    if suggestions and 1 <= choice <= len(suggestions):
                        suggestion = suggestions[choice - 1]
                        self._execute_suggestion(suggestion, run, user_query)
                        context["session_actions"].append(suggestion.get("label", ""))
                        
                        # Continue the session
                        console.print()
                        suggestions = self._generate_suggestions(run, commands, user_query)
                        self._display_suggestions(suggestions)
                        console.print()
                        continue
                    elif suggestions and choice == len(suggestions) + 1:
                        console.print("[dim]👋 Session ended.[/dim]")
                        break
                except ValueError:
                    pass
                
                # Handle natural language request
                handled = self._handle_natural_language_request(
                    response_stripped, 
                    suggestions, 
                    context, 
                    run,
                    commands
                )
                
                if handled:
                    context["session_actions"].append(response_stripped)
                    # Update context with the new query for better suggestions
                    context["last_query"] = response_stripped
                    
                    # Refresh suggestions based on new context
                    console.print()
                    # Use the new query for generating suggestions
                    combined_query = f"{user_query}. Then: {response_stripped}"
                    suggestions = self._generate_suggestions(run, commands, combined_query)
                    self._display_suggestions(suggestions)
                    console.print()
                
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]👋 Session ended.[/dim]")
                break
    
    def _display_suggestions(self, suggestions: list[dict]) -> None:
        """Display numbered suggestions."""
        if not suggestions:
            console.print("[dim]No specific suggestions available.[/dim]")
            return
        
        for i, suggestion in enumerate(suggestions, 1):
            icon = suggestion.get("icon", "💡")
            label = suggestion.get("label", "")
            desc = suggestion.get("description", "")
            console.print(f"  [cyan]{i}.[/cyan] {icon} {label}")
            if desc:
                console.print(f"      [dim]{desc}[/dim]")
        
        console.print(f"  [cyan]{len(suggestions) + 1}.[/cyan] 🚪 Exit session")
    
    def _handle_natural_language_request(
        self,
        request: str,
        suggestions: list[dict],
        context: dict,
        run: DoRun,
        commands: list[tuple[str, str, list[str]]],
    ) -> bool:
        """Handle a natural language request from the user.
        
        Uses LLM if available for full understanding, falls back to pattern matching.
        Returns True if the request was handled, False otherwise.
        """
        request_lower = request.lower()
        
        # Quick keyword matching for common actions (fast path)
        keyword_handlers = [
            (["start", "run", "begin", "launch", "execute"], "start"),
            (["setup", "configure", "config", "set up"], "setup"),
            (["demo", "example", "sample", "code"], "demo"),
            (["test", "verify", "check", "validate"], "test"),
        ]
        
        # Check if request is a simple match to existing suggestions
        for keywords, action_type in keyword_handlers:
            if any(kw in request_lower for kw in keywords):
                # Only use quick match if it's a very simple request
                if len(request.split()) <= 4:
                    for suggestion in suggestions:
                        if suggestion.get("type") == action_type:
                            self._execute_suggestion(suggestion, run, context["original_query"])
                            return True
        
        # Use LLM for full understanding if available
        console.print()
        console.print(f"[cyan]🤔 Understanding your request...[/cyan]")
        
        if self.llm_callback:
            return self._handle_request_with_llm(request, context, run, commands)
        else:
            # Fall back to pattern matching
            return self._handle_request_with_patterns(request, context, run)
    
    def _handle_request_with_llm(
        self,
        request: str,
        context: dict,
        run: DoRun,
        commands: list[tuple[str, str, list[str]]],
    ) -> bool:
        """Handle request using LLM for full understanding."""
        try:
            # Call LLM to understand the request
            llm_response = self.llm_callback(request, context)
            
            if not llm_response or llm_response.get("response_type") == "error":
                console.print(f"[yellow]⚠ Could not process request: {llm_response.get('error', 'Unknown error')}[/yellow]")
                return False
            
            response_type = llm_response.get("response_type")
            
            # Handle do_commands - execute with confirmation
            if response_type == "do_commands" and llm_response.get("do_commands"):
                do_commands = llm_response["do_commands"]
                reasoning = llm_response.get("reasoning", "")
                
                console.print()
                console.print(f"[cyan]🤖 {reasoning}[/cyan]")
                console.print()
                
                # Show commands and ask for confirmation
                console.print("[bold]📋 Commands to execute:[/bold]")
                for i, cmd_info in enumerate(do_commands, 1):
                    cmd = cmd_info.get("command", "")
                    purpose = cmd_info.get("purpose", "")
                    sudo = "🔐 " if cmd_info.get("requires_sudo") else ""
                    console.print(f"  {i}. {sudo}[green]{cmd}[/green]")
                    if purpose:
                        console.print(f"     [dim]{purpose}[/dim]")
                console.print()
                
                if not Confirm.ask("Execute these commands?", default=True):
                    console.print("[dim]Skipped.[/dim]")
                    return False
                
                # Execute the commands
                console.print()
                executed_in_session = []
                for cmd_info in do_commands:
                    cmd = cmd_info.get("command", "")
                    purpose = cmd_info.get("purpose", "Execute command")
                    needs_sudo = cmd_info.get("requires_sudo", False) or self._needs_sudo(cmd, [])
                    
                    console.print(f"[COMMAND] {cmd}")
                    console.print(f"   └─ {purpose}")
                    
                    success, stdout, stderr = self._execute_single_command(cmd, needs_sudo)
                    
                    if success:
                        console.print(f"   [green]✓ Success[/green]")
                        if stdout:
                            output_preview = stdout[:300] + ('...' if len(stdout) > 300 else '')
                            console.print(f"   [dim]{output_preview}[/dim]")
                        executed_in_session.append(cmd)
                    else:
                        console.print(f"   [red]✗ Failed: {stderr[:150]}[/red]")
                        
                        # Offer to diagnose and fix
                        if Confirm.ask("   Try to auto-fix?", default=True):
                            diagnosis = self._diagnoser.diagnose_error(cmd, stderr)
                            fixed, msg, _ = self._auto_fixer.auto_fix_error(cmd, stderr, diagnosis)
                            if fixed:
                                console.print(f"   [green]✓ Fixed: {msg}[/green]")
                                executed_in_session.append(cmd)
                
                # Track executed commands in context for suggestion generation
                if "executed_commands" not in context:
                    context["executed_commands"] = []
                context["executed_commands"].extend(executed_in_session)
                
                return True
            
            # Handle single command - execute directly
            elif response_type == "command" and llm_response.get("command"):
                cmd = llm_response["command"]
                reasoning = llm_response.get("reasoning", "")
                
                console.print()
                console.print(f"[cyan]📋 Running:[/cyan] [green]{cmd}[/green]")
                if reasoning:
                    console.print(f"   [dim]{reasoning}[/dim]")
                
                needs_sudo = self._needs_sudo(cmd, [])
                success, stdout, stderr = self._execute_single_command(cmd, needs_sudo)
                
                if success:
                    console.print(f"[green]✓ Success[/green]")
                    if stdout:
                        console.print(f"[dim]{stdout[:500]}{'...' if len(stdout) > 500 else ''}[/dim]")
                else:
                    console.print(f"[red]✗ Failed: {stderr[:200]}[/red]")
                
                return True
            
            # Handle answer - just display it
            elif response_type == "answer" and llm_response.get("answer"):
                console.print()
                console.print(llm_response["answer"])
                return True
            
            else:
                console.print(f"[yellow]I didn't understand that. Could you rephrase?[/yellow]")
                return False
                
        except Exception as e:
            console.print(f"[yellow]⚠ Error processing request: {e}[/yellow]")
            # Fall back to pattern matching
            return self._handle_request_with_patterns(request, context, run)
    
    def _handle_request_with_patterns(
        self,
        request: str,
        context: dict,
        run: DoRun,
    ) -> bool:
        """Handle request using pattern matching (fallback when LLM not available)."""
        # Try to generate a command from the natural language request
        generated = self._generate_command_from_request(request, context)
        
        if generated:
            cmd = generated.get("command")
            purpose = generated.get("purpose", "Execute user request")
            needs_confirm = generated.get("needs_confirmation", True)
            
            console.print()
            console.print(f"[cyan]📋 I'll run this command:[/cyan]")
            console.print(f"   [green]{cmd}[/green]")
            console.print(f"   [dim]{purpose}[/dim]")
            console.print()
            
            if needs_confirm:
                if not Confirm.ask("Proceed?", default=True):
                    console.print("[dim]Skipped.[/dim]")
                    return False
            
            # Execute the command
            needs_sudo = self._needs_sudo(cmd, [])
            success, stdout, stderr = self._execute_single_command(cmd, needs_sudo)
            
            if success:
                console.print(f"[green]✓ Success[/green]")
                if stdout:
                    output_preview = stdout[:500] + ('...' if len(stdout) > 500 else '')
                    console.print(f"[dim]{output_preview}[/dim]")
            else:
                console.print(f"[red]✗ Failed: {stderr[:200]}[/red]")
                
                # Offer to diagnose the error
                if Confirm.ask("Would you like me to try to fix this?", default=True):
                    diagnosis = self._diagnoser.diagnose_error(cmd, stderr)
                    fixed, msg, _ = self._auto_fixer.auto_fix_error(cmd, stderr, diagnosis)
                    if fixed:
                        console.print(f"[green]✓ Fixed: {msg}[/green]")
            
            return True
        
        # Couldn't understand the request
        console.print(f"[yellow]I'm not sure how to do that. Could you be more specific?[/yellow]")
        console.print(f"[dim]Try something like: 'run the container', 'show me the config', or select a number.[/dim]")
        return False
    
    def _generate_command_from_request(
        self,
        request: str,
        context: dict,
    ) -> dict | None:
        """Generate a command from a natural language request."""
        request_lower = request.lower()
        executed_cmds = context.get("executed_commands", [])
        cmd_context = " ".join(executed_cmds).lower()
        
        # Pattern matching for common requests
        patterns = [
            # Docker patterns
            (r"run.*(?:container|image|docker)(?:.*port\s*(\d+))?", self._gen_docker_run),
            (r"stop.*(?:container|docker)", self._gen_docker_stop),
            (r"remove.*(?:container|docker)", self._gen_docker_remove),
            (r"(?:show|list).*(?:containers?|images?)", self._gen_docker_list),
            (r"logs?(?:\s+of)?(?:\s+the)?(?:\s+container)?", self._gen_docker_logs),
            (r"exec.*(?:container|docker)|shell.*(?:container|docker)", self._gen_docker_exec),
            
            # Service patterns
            (r"(?:start|restart).*(?:service|nginx|apache|postgres|mysql|redis)", self._gen_service_start),
            (r"stop.*(?:service|nginx|apache|postgres|mysql|redis)", self._gen_service_stop),
            (r"status.*(?:service|nginx|apache|postgres|mysql|redis)", self._gen_service_status),
            
            # Package patterns
            (r"install\s+(.+)", self._gen_install_package),
            (r"update\s+(?:packages?|system)", self._gen_update_packages),
            
            # File patterns
            (r"(?:show|cat|view|read).*(?:config|file|log)(?:.*?([/\w\.\-]+))?", self._gen_show_file),
            (r"edit.*(?:config|file)(?:.*?([/\w\.\-]+))?", self._gen_edit_file),
            
            # Info patterns
            (r"(?:check|show|what).*(?:version|status)", self._gen_check_version),
            (r"(?:how|where).*(?:connect|access|use)", self._gen_show_connection_info),
        ]
        
        import re
        for pattern, handler in patterns:
            match = re.search(pattern, request_lower)
            if match:
                return handler(request, match, context)
        
        # Use LLM if available to generate command
        if self.llm_callback:
            return self._llm_generate_command(request, context)
        
        return None
    
    # Command generators
    def _gen_docker_run(self, request: str, match, context: dict) -> dict:
        # Find the image from context
        executed = context.get("executed_commands", [])
        image = "your-image"
        for cmd in executed:
            if "docker pull" in cmd:
                image = cmd.split("docker pull")[-1].strip()
                break
        
        # Check for port in request
        port = match.group(1) if match.lastindex and match.group(1) else "8080"
        container_name = image.split("/")[-1].split(":")[0]
        
        return {
            "command": f"docker run -d --name {container_name} -p {port}:{port} {image}",
            "purpose": f"Run {image} container on port {port}",
            "needs_confirmation": True,
        }
    
    def _gen_docker_stop(self, request: str, match, context: dict) -> dict:
        return {
            "command": "docker ps -q | xargs -r docker stop",
            "purpose": "Stop all running containers",
            "needs_confirmation": True,
        }
    
    def _gen_docker_remove(self, request: str, match, context: dict) -> dict:
        return {
            "command": "docker ps -aq | xargs -r docker rm",
            "purpose": "Remove all containers",
            "needs_confirmation": True,
        }
    
    def _gen_docker_list(self, request: str, match, context: dict) -> dict:
        if "image" in request.lower():
            return {"command": "docker images", "purpose": "List Docker images", "needs_confirmation": False}
        return {"command": "docker ps -a", "purpose": "List all containers", "needs_confirmation": False}
    
    def _gen_docker_logs(self, request: str, match, context: dict) -> dict:
        return {
            "command": "docker logs $(docker ps -lq) --tail 50",
            "purpose": "Show logs of the most recent container",
            "needs_confirmation": False,
        }
    
    def _gen_docker_exec(self, request: str, match, context: dict) -> dict:
        return {
            "command": "docker exec -it $(docker ps -lq) /bin/sh",
            "purpose": "Open shell in the most recent container",
            "needs_confirmation": True,
        }
    
    def _gen_service_start(self, request: str, match, context: dict) -> dict:
        # Extract service name
        services = ["nginx", "apache2", "postgresql", "mysql", "redis", "docker"]
        service = "nginx"  # default
        for svc in services:
            if svc in request.lower():
                service = svc
                break
        
        if "restart" in request.lower():
            return {"command": f"sudo systemctl restart {service}", "purpose": f"Restart {service}", "needs_confirmation": True}
        return {"command": f"sudo systemctl start {service}", "purpose": f"Start {service}", "needs_confirmation": True}
    
    def _gen_service_stop(self, request: str, match, context: dict) -> dict:
        services = ["nginx", "apache2", "postgresql", "mysql", "redis", "docker"]
        service = "nginx"
        for svc in services:
            if svc in request.lower():
                service = svc
                break
        return {"command": f"sudo systemctl stop {service}", "purpose": f"Stop {service}", "needs_confirmation": True}
    
    def _gen_service_status(self, request: str, match, context: dict) -> dict:
        services = ["nginx", "apache2", "postgresql", "mysql", "redis", "docker"]
        service = "nginx"
        for svc in services:
            if svc in request.lower():
                service = svc
                break
        return {"command": f"systemctl status {service}", "purpose": f"Check {service} status", "needs_confirmation": False}
    
    def _gen_install_package(self, request: str, match, context: dict) -> dict:
        package = match.group(1).strip() if match.group(1) else "package-name"
        # Clean up common words
        package = package.replace("please", "").replace("the", "").replace("package", "").strip()
        return {
            "command": f"sudo apt install -y {package}",
            "purpose": f"Install {package}",
            "needs_confirmation": True,
        }
    
    def _gen_update_packages(self, request: str, match, context: dict) -> dict:
        return {
            "command": "sudo apt update && sudo apt upgrade -y",
            "purpose": "Update all packages",
            "needs_confirmation": True,
        }
    
    def _gen_show_file(self, request: str, match, context: dict) -> dict:
        # Try to extract file path or use common config locations
        file_path = match.group(1) if match.lastindex and match.group(1) else None
        
        if not file_path:
            if "nginx" in request.lower():
                file_path = "/etc/nginx/nginx.conf"
            elif "apache" in request.lower():
                file_path = "/etc/apache2/apache2.conf"
            elif "postgres" in request.lower():
                file_path = "/etc/postgresql/*/main/postgresql.conf"
            else:
                file_path = "/etc/hosts"
        
        return {"command": f"cat {file_path}", "purpose": f"Show {file_path}", "needs_confirmation": False}
    
    def _gen_edit_file(self, request: str, match, context: dict) -> dict:
        file_path = match.group(1) if match.lastindex and match.group(1) else "/etc/hosts"
        return {
            "command": f"sudo nano {file_path}",
            "purpose": f"Edit {file_path}",
            "needs_confirmation": True,
        }
    
    def _gen_check_version(self, request: str, match, context: dict) -> dict:
        # Try to determine what to check version of
        tools = {
            "docker": "docker --version",
            "node": "node --version && npm --version",
            "python": "python3 --version && pip3 --version",
            "nginx": "nginx -v",
            "postgres": "psql --version",
        }
        
        for tool, cmd in tools.items():
            if tool in request.lower():
                return {"command": cmd, "purpose": f"Check {tool} version", "needs_confirmation": False}
        
        # Default: show multiple versions
        return {
            "command": "docker --version; node --version 2>/dev/null; python3 --version",
            "purpose": "Check installed tool versions",
            "needs_confirmation": False,
        }
    
    def _gen_show_connection_info(self, request: str, match, context: dict) -> dict:
        executed = context.get("executed_commands", [])
        
        # Check what was installed to provide relevant connection info
        if any("ollama" in cmd for cmd in executed):
            return {
                "command": "echo 'Ollama API: http://localhost:11434' && curl -s http://localhost:11434/api/tags 2>/dev/null | head -5",
                "purpose": "Show Ollama connection info",
                "needs_confirmation": False,
            }
        elif any("postgres" in cmd for cmd in executed):
            return {
                "command": "echo 'PostgreSQL: psql -U postgres -h localhost' && sudo -u postgres psql -c '\\conninfo'",
                "purpose": "Show PostgreSQL connection info",
                "needs_confirmation": False,
            }
        elif any("nginx" in cmd for cmd in executed):
            return {
                "command": "echo 'Nginx: http://localhost:80' && curl -I http://localhost 2>/dev/null | head -3",
                "purpose": "Show Nginx connection info",
                "needs_confirmation": False,
            }
        
        return {
            "command": "ss -tlnp | head -20",
            "purpose": "Show listening ports and services",
            "needs_confirmation": False,
        }
    
    def _llm_generate_command(self, request: str, context: dict) -> dict | None:
        """Use LLM to generate a command from the request."""
        if not self.llm_callback:
            return None
        
        try:
            prompt = f"""Given this context:
- User originally asked: {context.get('original_query', 'N/A')}
- Commands executed: {', '.join(context.get('executed_commands', [])[:5])}
- Previous session actions: {', '.join(context.get('session_actions', [])[:3])}

The user now asks: "{request}"

Generate a single Linux command to fulfill this request.
Respond with JSON: {{"command": "...", "purpose": "..."}}
If you cannot generate a safe command, respond with: {{"error": "reason"}}"""

            result = self.llm_callback(prompt)
            if result and isinstance(result, dict):
                if "command" in result:
                    return {
                        "command": result["command"],
                        "purpose": result.get("purpose", "Execute user request"),
                        "needs_confirmation": True,
                    }
        except Exception:
            pass
        
        return None
    
    def _generate_suggestions(
        self,
        run: DoRun,
        commands: list[tuple[str, str, list[str]]],
        user_query: str,
    ) -> list[dict]:
        """Generate context-aware suggestions based on what was installed/configured."""
        suggestions = []
        
        # Analyze what was done
        executed_cmds = [cmd for cmd, _, _ in commands]
        cmd_str = " ".join(executed_cmds).lower()
        query_lower = user_query.lower()
        
        # Docker-related suggestions
        if "docker" in cmd_str or "docker" in query_lower:
            if "pull" in cmd_str:
                # Suggest running the container
                for cmd, _, _ in commands:
                    if "docker pull" in cmd:
                        image = cmd.split("docker pull")[-1].strip()
                        suggestions.append({
                            "type": "start",
                            "icon": "🚀",
                            "label": f"Start the container",
                            "description": f"Run {image} in a container",
                            "command": f"docker run -d --name {image.split('/')[-1].split(':')[0]} {image}",
                            "purpose": f"Start {image} container",
                        })
                        suggestions.append({
                            "type": "demo",
                            "icon": "📝",
                            "label": "Show demo usage",
                            "description": f"Example docker-compose and run commands",
                            "demo_type": "docker",
                            "image": image,
                        })
                        break
        
        # Ollama/Model runner suggestions
        if "ollama" in cmd_str or "ollama" in query_lower or "model" in query_lower:
            suggestions.append({
                "type": "start",
                "icon": "🚀",
                "label": "Start Ollama server",
                "description": "Run Ollama in the background",
                "command": "docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama",
                "purpose": "Start Ollama server container",
            })
            suggestions.append({
                "type": "setup",
                "icon": "⚙️",
                "label": "Pull a model",
                "description": "Download a model like llama2, mistral, or codellama",
                "command": "docker exec ollama ollama pull llama2",
                "purpose": "Download llama2 model",
            })
            suggestions.append({
                "type": "demo",
                "icon": "📝",
                "label": "Show API demo",
                "description": "Example curl commands and Python code",
                "demo_type": "ollama",
            })
            suggestions.append({
                "type": "test",
                "icon": "🧪",
                "label": "Test the installation",
                "description": "Verify Ollama is running correctly",
                "command": "curl http://localhost:11434/api/tags",
                "purpose": "Check Ollama API",
            })
        
        # Nginx suggestions
        if "nginx" in cmd_str or "nginx" in query_lower:
            suggestions.append({
                "type": "start",
                "icon": "🚀",
                "label": "Start Nginx",
                "description": "Start the Nginx web server",
                "command": "sudo systemctl start nginx",
                "purpose": "Start Nginx service",
            })
            suggestions.append({
                "type": "setup",
                "icon": "⚙️",
                "label": "Configure a site",
                "description": "Set up a new virtual host",
                "demo_type": "nginx_config",
            })
            suggestions.append({
                "type": "test",
                "icon": "🧪",
                "label": "Test configuration",
                "description": "Verify Nginx config is valid",
                "command": "sudo nginx -t",
                "purpose": "Test Nginx configuration",
            })
        
        # PostgreSQL suggestions
        if "postgres" in cmd_str or "postgresql" in query_lower:
            suggestions.append({
                "type": "start",
                "icon": "🚀",
                "label": "Start PostgreSQL",
                "description": "Start the database server",
                "command": "sudo systemctl start postgresql",
                "purpose": "Start PostgreSQL service",
            })
            suggestions.append({
                "type": "setup",
                "icon": "⚙️",
                "label": "Create a database",
                "description": "Create a new database and user",
                "demo_type": "postgres_setup",
            })
            suggestions.append({
                "type": "test",
                "icon": "🧪",
                "label": "Test connection",
                "description": "Verify PostgreSQL is accessible",
                "command": "sudo -u postgres psql -c '\\l'",
                "purpose": "List PostgreSQL databases",
            })
        
        # Node.js/npm suggestions
        if "node" in cmd_str or "npm" in cmd_str or "nodejs" in query_lower:
            suggestions.append({
                "type": "demo",
                "icon": "📝",
                "label": "Show starter code",
                "description": "Example Express.js server",
                "demo_type": "nodejs",
            })
            suggestions.append({
                "type": "test",
                "icon": "🧪",
                "label": "Verify installation",
                "description": "Check Node.js and npm versions",
                "command": "node --version && npm --version",
                "purpose": "Check Node.js installation",
            })
        
        # Python/pip suggestions
        if "python" in cmd_str or "pip" in cmd_str:
            suggestions.append({
                "type": "demo",
                "icon": "📝",
                "label": "Show example code",
                "description": "Example Python usage",
                "demo_type": "python",
            })
            suggestions.append({
                "type": "test",
                "icon": "🧪",
                "label": "Test import",
                "description": "Verify packages are importable",
                "demo_type": "python_test",
            })
        
        # Generic suggestions if nothing specific matched
        if not suggestions:
            # Add a generic test suggestion
            suggestions.append({
                "type": "test",
                "icon": "🧪",
                "label": "Run a quick test",
                "description": "Verify the installation works",
                "demo_type": "generic_test",
            })
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _execute_suggestion(
        self,
        suggestion: dict,
        run: DoRun,
        user_query: str,
    ) -> None:
        """Execute a suggestion."""
        suggestion_type = suggestion.get("type")
        
        if suggestion_type == "demo":
            self._show_demo(suggestion.get("demo_type", "generic"), suggestion)
        elif "command" in suggestion:
            console.print()
            console.print(f"[cyan]Executing:[/cyan] {suggestion['command']}")
            console.print()
            
            needs_sudo = "sudo" in suggestion["command"]
            success, stdout, stderr = self._execute_single_command(
                suggestion["command"], 
                needs_sudo=needs_sudo
            )
            
            if success:
                console.print(f"[green]✓ Success[/green]")
                if stdout:
                    console.print(f"[dim]{stdout[:500]}{'...' if len(stdout) > 500 else ''}[/dim]")
            else:
                console.print(f"[red]✗ Failed: {stderr[:200]}[/red]")
        else:
            console.print("[yellow]This suggestion requires manual action.[/yellow]")
    
    def _show_demo(self, demo_type: str, suggestion: dict) -> None:
        """Show demo code/commands for a specific type."""
        console.print()
        
        if demo_type == "docker":
            image = suggestion.get("image", "your-image")
            console.print("[bold cyan]📝 Docker Usage Examples[/bold cyan]")
            console.print()
            console.print("[dim]# Run container in foreground:[/dim]")
            console.print(f"[green]docker run -it {image}[/green]")
            console.print()
            console.print("[dim]# Run container in background:[/dim]")
            console.print(f"[green]docker run -d --name myapp {image}[/green]")
            console.print()
            console.print("[dim]# Run with port mapping:[/dim]")
            console.print(f"[green]docker run -d -p 8080:8080 {image}[/green]")
            console.print()
            console.print("[dim]# Run with volume mount:[/dim]")
            console.print(f"[green]docker run -d -v /host/path:/container/path {image}[/green]")
        
        elif demo_type == "ollama":
            console.print("[bold cyan]📝 Ollama API Examples[/bold cyan]")
            console.print()
            console.print("[dim]# List available models:[/dim]")
            console.print("[green]curl http://localhost:11434/api/tags[/green]")
            console.print()
            console.print("[dim]# Generate text:[/dim]")
            console.print('''[green]curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello, how are you?"
}'[/green]''')
            console.print()
            console.print("[dim]# Python example:[/dim]")
            console.print('''[green]import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={
        'model': 'llama2',
        'prompt': 'Explain quantum computing in simple terms',
        'stream': False
    })
print(response.json()['response'])[/green]''')
        
        elif demo_type == "nginx_config":
            console.print("[bold cyan]📝 Nginx Configuration Example[/bold cyan]")
            console.print()
            console.print("[dim]# Create a new site config:[/dim]")
            console.print("[green]sudo nano /etc/nginx/sites-available/mysite[/green]")
            console.print()
            console.print("[dim]# Example config:[/dim]")
            console.print('''[green]server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
    }
}[/green]''')
            console.print()
            console.print("[dim]# Enable the site:[/dim]")
            console.print("[green]sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/[/green]")
            console.print("[green]sudo nginx -t && sudo systemctl reload nginx[/green]")
        
        elif demo_type == "postgres_setup":
            console.print("[bold cyan]📝 PostgreSQL Setup Example[/bold cyan]")
            console.print()
            console.print("[dim]# Create a new user and database:[/dim]")
            console.print("[green]sudo -u postgres createuser --interactive myuser[/green]")
            console.print("[green]sudo -u postgres createdb mydb -O myuser[/green]")
            console.print()
            console.print("[dim]# Connect to the database:[/dim]")
            console.print("[green]psql -U myuser -d mydb[/green]")
            console.print()
            console.print("[dim]# Python connection example:[/dim]")
            console.print('''[green]import psycopg2

conn = psycopg2.connect(
    dbname="mydb",
    user="myuser",
    password="mypassword",
    host="localhost"
)
cursor = conn.cursor()
cursor.execute("SELECT version();")
print(cursor.fetchone())[/green]''')
        
        elif demo_type == "nodejs":
            console.print("[bold cyan]📝 Node.js Example[/bold cyan]")
            console.print()
            console.print("[dim]# Create a simple Express server:[/dim]")
            console.print('''[green]// server.js
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.json({ message: 'Hello from Node.js!' });
});

app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});[/green]''')
            console.print()
            console.print("[dim]# Run it:[/dim]")
            console.print("[green]npm init -y && npm install express && node server.js[/green]")
        
        elif demo_type == "python":
            console.print("[bold cyan]📝 Python Example[/bold cyan]")
            console.print()
            console.print("[dim]# Simple HTTP server:[/dim]")
            console.print("[green]python3 -m http.server 8000[/green]")
            console.print()
            console.print("[dim]# Flask web app:[/dim]")
            console.print('''[green]from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return {'message': 'Hello from Python!'}

if __name__ == '__main__':
    app.run(debug=True)[/green]''')
        
        else:
            console.print("[dim]No specific demo available. Check the documentation for usage examples.[/dim]")
        
        console.print()
    
    def _execute_task_node(
        self,
        task: TaskNode,
        run: DoRun,
        original_commands: list[tuple[str, str, list[str]]],
        depth: int = 0,
    ):
        """Execute a single task node with auto-repair capabilities."""
        indent = "  " * depth
        task_num = f"[{task.task_type.value.upper()}]"
        
        # Check if task was marked as skipped (e.g., using existing resource)
        if task.status == CommandStatus.SKIPPED:
            console.print(f"{indent}[cyan][SKIPPED] {task.command[:60]}{'...' if len(task.command) > 60 else ''}[/cyan]")
            console.print(f"{indent}[green]   └─ ✓ {task.output or 'Using existing resource'}[/green]")
            
            # Log the skipped command
            cmd_log = CommandLog(
                command=task.command,
                purpose=task.purpose,
                timestamp=datetime.datetime.now().isoformat(),
                status=CommandStatus.SKIPPED,
                output=task.output or "Using existing resource",
            )
            run.commands.append(cmd_log)
            return
        
        console.print(f"{indent}[bold]{task_num}[/bold] {task.command[:60]}{'...' if len(task.command) > 60 else ''}")
        console.print(f"{indent}[dim]   └─ {task.purpose}[/dim]")
        
        protected_paths = []
        user_query = run.user_query if run else ""
        for cmd, _, protected in original_commands:
            if cmd == task.command:
                protected_paths = protected
                break
        
        file_check = self._file_analyzer.check_file_exists_and_usefulness(task.command, task.purpose, user_query)
        
        if file_check["recommendations"]:
            self._file_analyzer.apply_file_recommendations(file_check["recommendations"])
        
        task.status = CommandStatus.RUNNING
        start_time = time.time()
        
        needs_sudo = self._needs_sudo(task.command, protected_paths)
        success, stdout, stderr = self._execute_single_command(task.command, needs_sudo)
        
        task.output = stdout
        task.error = stderr
        task.duration_seconds = time.time() - start_time
        
        cmd_log = CommandLog(
            command=task.command,
            purpose=task.purpose,
            timestamp=datetime.datetime.now().isoformat(),
            status=CommandStatus.SUCCESS if success else CommandStatus.FAILED,
            output=stdout,
            error=stderr,
            duration_seconds=task.duration_seconds,
        )
        
        if success:
            task.status = CommandStatus.SUCCESS
            console.print(f"{indent}[green]   ✓ Success ({task.duration_seconds:.2f}s)[/green]")
            if stdout:
                output_preview = stdout[:100] + ('...' if len(stdout) > 100 else '')
                console.print(f"{indent}[dim]   {output_preview}[/dim]")
            run.commands.append(cmd_log)
            return
        
        task.status = CommandStatus.NEEDS_REPAIR
        diagnosis = self._diagnoser.diagnose_error(task.command, stderr)
        task.failure_reason = diagnosis.get("description", "Unknown error")
        
        console.print(f"{indent}[yellow]   ⚠ Error: {diagnosis['error_type']}[/yellow]")
        console.print(f"{indent}[dim]   {diagnosis['description']}[/dim]")
        
        # Check if this is a login/credential required error
        if diagnosis.get("category") == "login_required":
            console.print(f"{indent}[cyan]   🔐 Authentication required[/cyan]")
            
            login_success, login_msg = self._login_handler.handle_login(task.command, stderr)
            
            if login_success:
                console.print(f"{indent}[green]   ✓ {login_msg}[/green]")
                console.print(f"{indent}[cyan]   Retrying command...[/cyan]")
                
                # Retry the command
                needs_sudo = self._needs_sudo(task.command, [])
                success, new_stdout, new_stderr = self._execute_single_command(task.command, needs_sudo)
                
                if success:
                    task.status = CommandStatus.SUCCESS
                    task.reasoning = "Succeeded after authentication"
                    cmd_log.status = CommandStatus.SUCCESS
                    cmd_log.stdout = new_stdout[:500] if new_stdout else ""
                    console.print(f"{indent}[green]   ✓ Command succeeded after authentication![/green]")
                    run.commands.append(cmd_log)
                    return
                else:
                    # Still failed after login
                    stderr = new_stderr
                    diagnosis = self._diagnoser.diagnose_error(task.command, stderr)
                    console.print(f"{indent}[yellow]   Command still failed: {stderr[:100]}[/yellow]")
            else:
                console.print(f"{indent}[yellow]   {login_msg}[/yellow]")
        
        if diagnosis.get("extracted_path"):
            console.print(f"{indent}[dim]   Path: {diagnosis['extracted_path']}[/dim]")
        
        # Handle timeout errors specially - don't blindly retry
        if diagnosis.get("category") == "timeout" or "timed out" in stderr.lower():
            console.print(f"{indent}[yellow]   ⏱️  This operation timed out[/yellow]")
            
            # Check if it's a docker pull - those might still be running
            if "docker pull" in task.command.lower():
                console.print(f"{indent}[cyan]   ℹ️  Docker pull may still be downloading in background[/cyan]")
                console.print(f"{indent}[dim]   Check with: docker images | grep <image-name>[/dim]")
                console.print(f"{indent}[dim]   Or retry with: docker pull --timeout=0 <image>[/dim]")
            elif "apt" in task.command.lower():
                console.print(f"{indent}[cyan]   ℹ️  Package installation timed out[/cyan]")
                console.print(f"{indent}[dim]   Check apt status: sudo dpkg --configure -a[/dim]")
                console.print(f"{indent}[dim]   Then retry the command[/dim]")
            else:
                console.print(f"{indent}[cyan]   ℹ️  You can retry this command manually[/cyan]")
            
            # Mark as needing manual intervention, not auto-fix
            task.status = CommandStatus.NEEDS_REPAIR
            task.failure_reason = "Operation timed out - may need manual retry"
            cmd_log.status = CommandStatus.FAILED
            cmd_log.error = stderr
            run.commands.append(cmd_log)
            return
        
        if task.repair_attempts < task.max_repair_attempts:
            task.repair_attempts += 1
            console.print(f"{indent}[cyan]   🔧 Auto-fix attempt {task.repair_attempts}/{task.max_repair_attempts}[/cyan]")
            
            fixed, fix_message, fix_commands = self._auto_fixer.auto_fix_error(
                task.command, stderr, diagnosis, max_attempts=3
            )
            
            for fix_cmd in fix_commands:
                repair_task = self._task_tree.add_repair_task(
                    parent=task,
                    command=fix_cmd,
                    purpose=f"Auto-fix: {diagnosis['error_type']}",
                    reasoning=fix_message,
                )
                repair_task.status = CommandStatus.SUCCESS
            
            if fixed:
                task.status = CommandStatus.SUCCESS
                task.reasoning = f"Auto-fixed: {fix_message}"
                console.print(f"{indent}[green]   ✓ {fix_message}[/green]")
                cmd_log.status = CommandStatus.SUCCESS
                run.commands.append(cmd_log)
                return
            else:
                console.print(f"{indent}[yellow]   Auto-fix incomplete: {fix_message}[/yellow]")
        
        task.status = CommandStatus.FAILED
        task.reasoning = self._generate_task_failure_reasoning(task, diagnosis)
        console.print(f"{indent}[red]   ✗ Failed: {diagnosis['description'][:100]}[/red]")
        console.print(f"{indent}[dim]   Reasoning: {task.reasoning}[/dim]")
        
        if diagnosis.get("fix_commands") or stderr:
            console.print(f"\n{indent}[yellow]💡 Manual intervention available[/yellow]")
            
            suggested_cmds = diagnosis.get("fix_commands", [f"sudo {task.command}"])
            console.print(f"{indent}[dim]   Suggested commands:[/dim]")
            for cmd in suggested_cmds[:3]:
                console.print(f"{indent}[cyan]   $ {cmd}[/cyan]")
            
            if Confirm.ask(f"{indent}Run manually while Cortex monitors?", default=False):
                manual_success = self._supervise_manual_intervention_for_task(
                    task, suggested_cmds, run
                )
                if manual_success:
                    task.status = CommandStatus.SUCCESS
                    task.reasoning = "Completed via monitored manual intervention"
                    cmd_log.status = CommandStatus.SUCCESS
        
        cmd_log.status = task.status
        run.commands.append(cmd_log)
    
    def _supervise_manual_intervention_for_task(
        self,
        task: TaskNode,
        suggested_commands: list[str],
        run: DoRun,
    ) -> bool:
        """Supervise manual intervention for a specific task with terminal monitoring."""
        console.print("\n[bold cyan]═══ Manual Intervention Mode ═══[/bold cyan]")
        console.print("\n[yellow]Run these commands in another terminal:[/yellow]")
        
        for i, cmd in enumerate(suggested_commands, 1):
            console.print(f"[bold]{i}. {cmd}[/bold]")
        
        self._terminal_monitor = TerminalMonitor(
            notification_callback=lambda title, msg: self._send_notification(title, msg)
        )
        self._terminal_monitor.start()
        
        console.print("\n[dim]🔍 Cortex is now monitoring your terminal for issues...[/dim]")
        
        try:
            while True:
                console.print()
                action = Confirm.ask("Have you completed the manual step?", default=True)
                
                if action:
                    success = Confirm.ask("Was it successful?", default=True)
                    
                    if success:
                        console.print("[green]✓ Manual step completed successfully[/green]")
                        
                        verify_task = self._task_tree.add_verify_task(
                            parent=task,
                            command="# Manual verification",
                            purpose="User confirmed manual intervention success",
                        )
                        verify_task.status = CommandStatus.SUCCESS
                        
                        return True
                    else:
                        console.print("\n[yellow]What happened?[/yellow]")
                        console.print("  1. Still permission denied")
                        console.print("  2. File/path not found")
                        console.print("  3. Service error")
                        console.print("  4. Other error")
                        
                        try:
                            choice = input("Enter choice (1-4): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            return False
                        
                        if choice == "1":
                            console.print("\n[cyan]Try: sudo su - then run the command[/cyan]")
                        elif choice == "2":
                            console.print("\n[cyan]Verify the path exists: ls -la <path>[/cyan]")
                        elif choice == "3":
                            console.print("\n[cyan]Check service status: systemctl status <service>[/cyan]")
                        else:
                            console.print("\n[cyan]Please describe the error and try again[/cyan]")
                        
                        retry = Confirm.ask("Continue trying?", default=True)
                        if not retry:
                            return False
                else:
                    console.print("[dim]Take your time. Cortex is still monitoring...[/dim]")
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Manual intervention cancelled[/yellow]")
            return False
        finally:
            if self._terminal_monitor:
                self._terminal_monitor.stop()
                self._terminal_monitor = None
    
    def _generate_task_failure_reasoning(
        self,
        task: TaskNode,
        diagnosis: dict,
    ) -> str:
        """Generate detailed reasoning for why a task failed."""
        parts = []
        
        parts.append(f"Error: {diagnosis.get('error_type', 'unknown')}")
        
        if task.repair_attempts > 0:
            parts.append(f"Repair attempts: {task.repair_attempts} (all failed)")
        
        if diagnosis.get("extracted_path"):
            parts.append(f"Problem path: {diagnosis['extracted_path']}")
        
        error_type = diagnosis.get("error_type", "")
        if "permission" in error_type.lower():
            parts.append("Root cause: Insufficient file system permissions")
        elif "not_found" in error_type.lower():
            parts.append("Root cause: Required file or directory does not exist")
        elif "service" in error_type.lower():
            parts.append("Root cause: System service issue")
        
        if diagnosis.get("fix_commands"):
            parts.append(f"Suggested fix: {diagnosis['fix_commands'][0][:50]}...")
        
        return " | ".join(parts)
    
    def _generate_tree_summary(self, run: DoRun) -> str:
        """Generate a summary from the task tree execution."""
        if not self._task_tree:
            return self._generate_summary(run)
        
        summary = self._task_tree.get_summary()
        
        total = sum(summary.values())
        success = summary.get("success", 0)
        failed = summary.get("failed", 0)
        repaired = summary.get("needs_repair", 0)
        
        parts = [
            f"Total tasks: {total}",
            f"Successful: {success}",
            f"Failed: {failed}",
        ]
        
        if repaired > 0:
            parts.append(f"Repair attempted: {repaired}")
        
        if self._permission_requests_count > 1:
            parts.append(f"Permission requests: {self._permission_requests_count}")
        
        return " | ".join(parts)
    
    def provide_manual_instructions(
        self,
        commands: list[tuple[str, str, list[str]]],
        user_query: str,
    ) -> DoRun:
        """Provide instructions for manual execution and monitor progress."""
        run = DoRun(
            run_id=self.db._generate_run_id(),
            summary="",
            mode=RunMode.USER_MANUAL,
            user_query=user_query,
            started_at=datetime.datetime.now().isoformat(),
            session_id=self.current_session_id or "",
        )
        self.current_run = run
        
        console.print()
        console.print(Panel(
            "[bold cyan]📋 Manual Execution Instructions[/bold cyan]",
            expand=False,
        ))
        console.print()
        
        cwd = os.getcwd()
        console.print(f"[bold]1. Open a new terminal and navigate to:[/bold]")
        console.print(f"   [cyan]cd {cwd}[/cyan]")
        console.print()
        
        console.print(f"[bold]2. Execute the following commands in order:[/bold]")
        console.print()
        
        for i, (cmd, purpose, protected) in enumerate(commands, 1):
            console.print(f"   [bold yellow]Step {i}:[/bold yellow] {purpose}")
            needs_sudo = self._needs_sudo(cmd, protected)
            
            if protected:
                console.print(f"   [red]⚠️  Accesses protected paths: {', '.join(protected)}[/red]")
            
            if needs_sudo and not cmd.strip().startswith("sudo"):
                console.print(f"   [cyan]sudo {cmd}[/cyan]")
            else:
                console.print(f"   [cyan]{cmd}[/cyan]")
            console.print()
            
            run.commands.append(CommandLog(
                command=cmd,
                purpose=purpose,
                timestamp=datetime.datetime.now().isoformat(),
                status=CommandStatus.PENDING,
            ))
        
        console.print("[bold]3. Once done, return to this terminal and press Enter.[/bold]")
        console.print()
        
        monitor = TerminalMonitor(
            notification_callback=lambda title, msg: self._send_notification(title, msg, "normal")
        )
        
        expected_commands = [cmd for cmd, _, _ in commands]
        monitor.start_monitoring(expected_commands)
        
        console.print("[dim]🔍 Monitoring terminal activity... (press Enter when done)[/dim]")
        
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        
        observed = monitor.stop_monitoring()
        
        console.print()
        console.print("[bold]📊 Execution Summary:[/bold]")
        
        if observed:
            console.print(f"[green]Observed {len(observed)} commands in terminal[/green]")
            for obs in observed[-10:]:
                console.print(f"[dim]  • {obs['command']}[/dim]")
        else:
            console.print("[yellow]No commands observed (this is normal if bash history sync is disabled)[/yellow]")
        
        for obs in observed:
            run.commands.append(CommandLog(
                command=obs["command"],
                purpose="User-executed command",
                timestamp=obs["timestamp"],
                status=CommandStatus.SUCCESS,
            ))
        
        run.completed_at = datetime.datetime.now().isoformat()
        run.summary = self._generate_summary(run)
        
        self.db.save_run(run)
        
        console.print()
        console.print(f"[bold green]✓ Session recorded[/bold green] (ID: {run.run_id})")
        console.print(f"[dim]Summary: {run.summary}[/dim]")
        
        return run
    
    def _generate_summary(self, run: DoRun) -> str:
        """Generate a summary of what was done in the run."""
        successful = sum(1 for c in run.commands if c.status == CommandStatus.SUCCESS)
        failed = sum(1 for c in run.commands if c.status == CommandStatus.FAILED)
        
        mode_str = "automated" if run.mode == RunMode.CORTEX_EXEC else "manual"
        
        if failed == 0:
            return f"Successfully executed {successful} commands ({mode_str}) for: {run.user_query[:50]}"
        else:
            return f"Executed {successful} commands with {failed} failures ({mode_str}) for: {run.user_query[:50]}"
    
    def get_run_history(self, limit: int = 20) -> list[DoRun]:
        """Get recent do run history."""
        return self.db.get_recent_runs(limit)
    
    def get_run(self, run_id: str) -> DoRun | None:
        """Get a specific run by ID."""
        return self.db.get_run(run_id)
    
    # Expose diagnosis and auto-fix methods for external use
    def _diagnose_error(self, cmd: str, stderr: str) -> dict[str, Any]:
        """Diagnose a command failure."""
        return self._diagnoser.diagnose_error(cmd, stderr)
    
    def _auto_fix_error(
        self,
        cmd: str,
        stderr: str,
        diagnosis: dict[str, Any],
        max_attempts: int = 5,
    ) -> tuple[bool, str, list[str]]:
        """Auto-fix an error."""
        return self._auto_fixer.auto_fix_error(cmd, stderr, diagnosis, max_attempts)
    
    def _check_for_conflicts(self, cmd: str, purpose: str) -> dict[str, Any]:
        """Check for conflicts."""
        return self._conflict_detector.check_for_conflicts(cmd, purpose)
    
    def _run_verification_tests(
        self,
        commands_executed: list[CommandLog],
        user_query: str,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Run verification tests."""
        return self._verification_runner.run_verification_tests(commands_executed, user_query)
    
    def _check_file_exists_and_usefulness(
        self,
        cmd: str,
        purpose: str,
        user_query: str,
    ) -> dict[str, Any]:
        """Check file existence and usefulness."""
        return self._file_analyzer.check_file_exists_and_usefulness(cmd, purpose, user_query)
    
    def _analyze_file_usefulness(
        self,
        content: str,
        purpose: str,
        user_query: str,
    ) -> dict[str, Any]:
        """Analyze file usefulness."""
        return self._file_analyzer.analyze_file_usefulness(content, purpose, user_query)


def setup_cortex_user() -> bool:
    """Setup the cortex user if it doesn't exist."""
    handler = DoHandler()
    return handler.setup_cortex_user()


def get_do_handler() -> DoHandler:
    """Get a DoHandler instance."""
    return DoHandler()

