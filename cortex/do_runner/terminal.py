"""Terminal monitoring for the manual execution flow."""

import datetime
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

console = Console()


class TerminalMonitor:
    """
    Monitors terminal commands for the manual execution flow.
    
    Monitors multiple sources:
    - Bash history file
    - Cursor terminal files (if available)
    - External terminal output
    """
    
    def __init__(self, notification_callback: Callable[[str, str], None] | None = None):
        self.notification_callback = notification_callback
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._commands_observed: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._cursor_terminals_dir: Path | None = None
        self._expected_commands: list[str] = []
        
        # Try to find Cursor terminals directory
        cursor_base = Path.home() / ".cursor" / "projects"
        if cursor_base.exists():
            for project_dir in cursor_base.iterdir():
                terminals_path = project_dir / "terminals"
                if terminals_path.exists():
                    self._cursor_terminals_dir = terminals_path
                    break
    
    def start(self):
        """Start monitoring terminal for commands."""
        self.start_monitoring()
    
    def stop(self):
        """Stop monitoring terminal."""
        self.stop_monitoring()
    
    def start_monitoring(self, expected_commands: list[str] | None = None):
        """Start monitoring terminal for commands."""
        self._monitoring = True
        self._expected_commands = expected_commands or []
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> list[dict[str, Any]]:
        """Stop monitoring and return observed commands."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        
        with self._lock:
            return list(self._commands_observed)
    
    def get_observed_commands(self) -> list[dict[str, Any]]:
        """Get all observed commands so far."""
        with self._lock:
            return list(self._commands_observed)
    
    def _monitor_loop(self):
        """Monitor loop that watches for terminal activity from multiple sources."""
        history_file = Path.home() / ".bash_history"
        
        file_positions: dict[str, int] = {}
        
        if history_file.exists():
            try:
                file_positions[str(history_file)] = history_file.stat().st_size
            except OSError:
                pass
        
        if self._cursor_terminals_dir and self._cursor_terminals_dir.exists():
            for term_file in self._cursor_terminals_dir.glob("*.txt"):
                try:
                    file_positions[str(term_file)] = term_file.stat().st_size
                except OSError:
                    pass
        
        while self._monitoring:
            time.sleep(0.5)
            
            if history_file.exists():
                self._check_file_for_new_commands(
                    history_file, file_positions, source="bash_history"
                )
            
            if self._cursor_terminals_dir and self._cursor_terminals_dir.exists():
                for term_file in self._cursor_terminals_dir.glob("*.txt"):
                    self._check_file_for_new_commands(
                        term_file, file_positions, source=f"cursor_terminal:{term_file.stem}"
                    )
    
    def _check_file_for_new_commands(
        self,
        file_path: Path,
        positions: dict[str, int],
        source: str,
    ):
        """Check a file for new commands and process them."""
        try:
            current_size = file_path.stat().st_size
            key = str(file_path)
            
            if key not in positions:
                positions[key] = current_size
                return
            
            if current_size > positions[key]:
                with open(file_path) as f:
                    f.seek(positions[key])
                    new_content = f.read()
                    
                    new_commands = self._extract_commands_from_content(new_content, source)
                    
                    for cmd in new_commands:
                        self._process_observed_command(cmd, source)
                
                positions[key] = current_size
                
        except OSError:
            pass
    
    def _extract_commands_from_content(self, content: str, source: str) -> list[str]:
        """Extract commands from terminal content."""
        commands = []
        
        if source == "bash_history":
            commands = [c.strip() for c in content.strip().split("\n") if c.strip()]
        else:
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                prompt_patterns = [
                    r"^\$ (.+)$",
                    r"^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+:.+\$ (.+)$",
                    r"^>>> (.+)$",
                ]
                
                for pattern in prompt_patterns:
                    match = re.match(pattern, line)
                    if match:
                        commands.append(match.group(1))
                        break
        
        return commands
    
    def _process_observed_command(self, command: str, source: str = "unknown"):
        """Process an observed command."""
        with self._lock:
            self._commands_observed.append({
                "command": command,
                "timestamp": datetime.datetime.now().isoformat(),
                "source": source,
            })
        
        issues = self._check_command_issues(command)
        if issues and self.notification_callback:
            self.notification_callback(f"Cortex: Issue in {source}", issues)
    
    def _check_command_issues(self, command: str) -> str | None:
        """Check if a command has potential issues and return a warning."""
        issues = []
        
        if any(p in command for p in ["/etc/", "/var/", "/usr/"]):
            if not command.startswith("sudo") and not command.startswith("cat"):
                issues.append("May need sudo for system files")
        
        if "rm -rf /" in command:
            issues.append("DANGER: Destructive command detected!")
        
        typo_checks = {
            "sudp": "sudo",
            "suod": "sudo",
            "cta": "cat",
            "mdir": "mkdir",
            "mkidr": "mkdir",
        }
        for typo, correct in typo_checks.items():
            if command.startswith(typo + " "):
                issues.append(f"Typo? Did you mean '{correct}'?")
        
        return "; ".join(issues) if issues else None

