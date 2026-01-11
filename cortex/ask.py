"""Natural language query interface for Cortex.

Handles user questions about installed packages, configurations,
and system state using an agentic LLM loop with command execution.
"""

import json
import os
import re
import shlex
import sqlite3
import subprocess
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class LLMResponseType(str, Enum):
    """Type of response from the LLM."""
    COMMAND = "command"
    ANSWER = "answer"


class SystemCommand(BaseModel):
    """Pydantic model for a system command to be executed.
    
    The LLM must return either a command to execute for data gathering,
    or a final answer to the user's question.
    """
    response_type: LLMResponseType = Field(
        description="Whether this is a command to execute or a final answer"
    )
    command: str | None = Field(
        default=None,
        description="The shell command to execute (only for response_type='command')"
    )
    answer: str | None = Field(
        default=None,
        description="The final answer to the user (only for response_type='answer')"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this command/answer was chosen"
    )

    @field_validator("command")
    @classmethod
    def validate_command_not_empty(cls, v: str | None, info) -> str | None:
        if info.data.get("response_type") == LLMResponseType.COMMAND:
            if not v or not v.strip():
                raise ValueError("Command cannot be empty when response_type is 'command'")
        return v

    @field_validator("answer")
    @classmethod
    def validate_answer_not_empty(cls, v: str | None, info) -> str | None:
        if info.data.get("response_type") == LLMResponseType.ANSWER:
            if not v or not v.strip():
                raise ValueError("Answer cannot be empty when response_type is 'answer'")
        return v


class CommandValidator:
    """Validates and filters commands to ensure they are read-only.
    
    Only allows commands that fetch data, blocks any that modify the system.
    """
    
    # Commands that are purely read-only and safe
    ALLOWED_COMMANDS: set[str] = {
        # System info
        "uname", "hostname", "uptime", "whoami", "id", "groups", "w", "who", "last",
        "date", "cal", "timedatectl",
        # File/directory listing (read-only)
        "ls", "pwd", "tree", "file", "stat", "readlink", "realpath", "dirname", "basename",
        "find", "locate", "which", "whereis", "type", "command",
        # Text viewing (read-only)
        "cat", "head", "tail", "less", "more", "wc", "nl", "strings",
        # Text processing (non-modifying)
        "grep", "egrep", "fgrep", "awk", "sed", "cut", "sort", "uniq", "tr", "column",
        "diff", "comm", "join", "paste", "expand", "unexpand", "fold", "fmt",
        # Package queries (read-only)
        "dpkg-query", "dpkg", "apt-cache", "apt-mark", "apt-config", "aptitude",
        "pip3", "pip", "python3", "python", "gem", "npm", "cargo", "go",
        # System info commands
        "lsb_release", "hostnamectl", "lscpu", "lsmem", "lsblk", "lspci", "lsusb",
        "lshw", "dmidecode", "hwinfo", "inxi",
        # Process/resource info
        "ps", "top", "htop", "pgrep", "pidof", "pstree", "free", "vmstat", "iostat",
        "mpstat", "sar", "nproc", "getconf",
        # Disk/filesystem info
        "df", "du", "mount", "findmnt", "blkid", "lsof", "fuser", "fdisk",
        # Network info (read-only)
        "ip", "ifconfig", "netstat", "ss", "route", "arp", "ping", "traceroute",
        "tracepath", "nslookup", "dig", "host", "getent", "hostname",
        # GPU info
        "nvidia-smi", "nvcc", "rocm-smi", "clinfo",
        # Environment
        "env", "printenv", "echo", "printf",
        # Systemd info (read-only)
        "systemctl", "journalctl", "loginctl", "timedatectl", "localectl",
        # Kernel/modules
        "uname", "lsmod", "modinfo", "sysctl",
        # Misc info
        "getconf", "locale", "xdpyinfo", "xrandr",
    }
    
    # Subcommands that are blocked for otherwise allowed commands
    BLOCKED_SUBCOMMANDS: dict[str, set[str]] = {
        "dpkg": {"--configure", "-i", "--install", "--remove", "-r", "--purge", "-P", 
                 "--unpack", "--clear-avail", "--forget-old-unavail", "--update-avail",
                 "--merge-avail", "--set-selections", "--clear-selections"},
        "apt-mark": {"auto", "manual", "hold", "unhold", "showauto", "showmanual"},  # only show* are safe
        "pip3": {"install", "uninstall", "download", "wheel", "cache"},
        "pip": {"install", "uninstall", "download", "wheel", "cache"},
        "python3": {"-c"},  # Block arbitrary code execution
        "python": {"-c"},
        "npm": {"install", "uninstall", "update", "ci", "run", "exec", "init", "publish"},
        "gem": {"install", "uninstall", "update", "cleanup", "pristine"},
        "cargo": {"install", "uninstall", "build", "run", "clean", "publish"},
        "go": {"install", "get", "build", "run", "clean", "mod"},
        "systemctl": {"start", "stop", "restart", "reload", "enable", "disable", 
                      "mask", "unmask", "edit", "set-property", "reset-failed",
                      "daemon-reload", "daemon-reexec", "kill", "isolate",
                      "set-default", "set-environment", "unset-environment"},
        "mount": {"--bind", "-o", "--move"},  # Block actual mounting
        "fdisk": {"-l"},  # Only allow listing (-l), block everything else (inverted logic handled below)
        "sysctl": {"-w", "--write", "-p", "--load"},  # Block writes
    }
    
    # Commands that are completely blocked (never allowed)
    BLOCKED_COMMANDS: set[str] = {
        # Dangerous/destructive
        "rm", "rmdir", "unlink", "shred",
        "mv", "cp", "install", "mkdir", "touch",
        # Editors (sed is allowed for text processing, redirections are blocked separately)
        "nano", "vim", "vi", "emacs", "ed",
        # Package modification
        "apt", "apt-get", "aptitude", "dpkg-reconfigure", "update-alternatives",
        "snap", "flatpak",
        # System modification
        "shutdown", "reboot", "poweroff", "halt", "init", "telinit",
        "useradd", "userdel", "usermod", "groupadd", "groupdel", "groupmod",
        "passwd", "chpasswd", "chage",
        "chmod", "chown", "chgrp", "chattr", "setfacl",
        "ln", "mkfifo", "mknod",
        # Dangerous utilities
        "dd", "mkfs", "fsck", "parted", "gdisk", "cfdisk", "sfdisk",
        "kill", "killall", "pkill",
        "nohup", "disown", "bg", "fg",
        "crontab", "at", "batch",
        "su", "sudo", "doas", "pkexec",
        # Network modification
        "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
        "ifup", "ifdown", "dhclient",
        # Shell/code execution
        "bash", "sh", "zsh", "fish", "dash", "csh", "tcsh", "ksh",
        "eval", "exec", "source",
        "xargs",  # Can be used to execute arbitrary commands
        "tee",  # Writes to files
    }
    
    # Patterns that indicate dangerous operations
    DANGEROUS_PATTERNS: list[str] = [
        r">\s*[^|]",           # Output redirection (except pipes)
        r">>\s*",              # Append redirection
        r"<\s*",               # Input redirection  
        r"\$\(",               # Command substitution
        r"`[^`]+`",            # Backtick command substitution
        r";\s*",               # Command chaining
        r"&&\s*",              # AND chaining
        r"\|\|\s*",            # OR chaining
        r"\|.*(?:sh|bash|zsh|exec|eval|xargs)",  # Piping to shell
    ]
    
    @classmethod
    def validate_command(cls, command: str) -> tuple[bool, str]:
        """Validate a command for safety.
        
        Args:
            command: The shell command to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command or not command.strip():
            return False, "Empty command"
        
        command = command.strip()
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Command contains blocked pattern (redirections, chaining, or subshells)"
        
        # Parse the command
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"
        
        if not parts:
            return False, "Empty command"
        
        # Get base command (handle sudo prefix)
        base_cmd = parts[0]
        cmd_args = parts[1:]
        
        if base_cmd == "sudo":
            return False, "sudo is not allowed - only read-only commands permitted"
        
        # Check if command is completely blocked
        if base_cmd in cls.BLOCKED_COMMANDS:
            return False, f"Command '{base_cmd}' is not allowed - it can modify the system"
        
        # Check if command is in allowed list
        if base_cmd not in cls.ALLOWED_COMMANDS:
            return False, f"Command '{base_cmd}' is not in the allowed list of read-only commands"
        
        # Check for blocked subcommands
        if base_cmd in cls.BLOCKED_SUBCOMMANDS:
            blocked = cls.BLOCKED_SUBCOMMANDS[base_cmd]
            for arg in cmd_args:
                # Handle fdisk specially - only -l is allowed
                if base_cmd == "fdisk":
                    if arg not in ["-l", "--list"]:
                        return False, f"fdisk only allows -l/--list for listing partitions"
                elif arg in blocked:
                    return False, f"Subcommand '{arg}' is not allowed for '{base_cmd}' - it can modify the system"
        
        # Special handling for pip/pip3 - only allow show, list, freeze, check, config
        if base_cmd in ["pip", "pip3"]:
            if cmd_args:
                allowed_pip_cmds = {"show", "list", "freeze", "check", "config", "--version", "-V", "help", "--help"}
                if cmd_args[0] not in allowed_pip_cmds:
                    return False, f"pip command '{cmd_args[0]}' is not allowed - only read-only commands like 'show', 'list', 'freeze' are permitted"
        
        # Special handling for apt-mark - only showhold, showauto, showmanual
        if base_cmd == "apt-mark":
            if cmd_args:
                allowed_apt_mark = {"showhold", "showauto", "showmanual"}
                if cmd_args[0] not in allowed_apt_mark:
                    return False, f"apt-mark command '{cmd_args[0]}' is not allowed - only showhold, showauto, showmanual are permitted"
        
        return True, ""
    
    @classmethod
    def execute_command(cls, command: str, timeout: int = 10) -> tuple[bool, str, str]:
        """Execute a validated command and return the result.
        
        Args:
            command: The shell command to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Validate first
        is_valid, error = cls.validate_command(command)
        if not is_valid:
            return False, "", f"Command blocked: {error}"
        
        try:
            result = subprocess.run(
                command,
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
            return False, "", f"Command execution failed: {e}"


class AskHandler:
    """Handles natural language questions about the system using an agentic loop.
    
    The handler uses an iterative approach:
    1. LLM generates a read-only command to gather information
    2. Command is validated and executed
    3. Output is sent back to LLM
    4. LLM either generates another command or provides final answer
    5. Max 5 iterations before giving up
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        api_key: str,
        provider: str = "claude",
        model: str | None = None,
        debug: bool = False,
    ):
        """Initialize the ask handler.

        Args:
            api_key: API key for the LLM provider
            provider: Provider name ("openai", "claude", or "ollama")
            model: Optional model name override
            debug: Enable debug output to shell
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model or self._default_model()
        self.debug = debug
        
        # Import rich console for debug output
        if self.debug:
            from rich.console import Console
            from rich.panel import Panel
            self._console = Console()
        else:
            self._console = None

        # Initialize cache
        try:
            from cortex.semantic_cache import SemanticCache

            self.cache: SemanticCache | None = SemanticCache()
        except (ImportError, OSError):
            self.cache = None

        self._initialize_client()

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-4o"  # Use gpt-4o for 128K context
        elif self.provider == "claude":
            return "claude-sonnet-4-20250514"
        elif self.provider == "ollama":
            return "llama3.2"
        elif self.provider == "fake":
            return "fake"
        return "gpt-4o"
    
    def _debug_print(self, title: str, content: str, style: str = "dim") -> None:
        """Print debug output if debug mode is enabled."""
        if self.debug and self._console:
            from rich.panel import Panel
            self._console.print(Panel(content, title=f"[bold]{title}[/bold]", style=style))

    def _initialize_client(self):
        if self.provider == "openai":
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        elif self.provider == "claude":
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        elif self.provider == "ollama":
            self.ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            self.client = None
        elif self.provider == "fake":
            self.client = None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_system_prompt(self) -> str:
        return """You are a Linux system assistant that answers questions by executing read-only shell commands.

Your task is to help answer the user's question about their system by:
1. Generating shell commands to gather the needed information
2. Analyzing the command output
3. Either generating another command if more info is needed, or providing the final answer

IMPORTANT RULES:
- You can ONLY use READ-ONLY commands that fetch data (no modifications allowed)
- Allowed commands include: cat, ls, grep, find, dpkg-query, apt-cache, pip3 show/list, ps, df, free, uname, lscpu, etc.
- NEVER use commands that modify the system (rm, mv, cp, apt install, pip install, etc.)
- NEVER use sudo
- NEVER use output redirection (>, >>), command chaining (;, &&, ||), or command substitution ($(), ``)

You must respond with a JSON object in this exact format:
{
    "response_type": "command" | "answer",
    "command": "<shell command to execute>" (only if response_type is "command"),
    "answer": "<your answer to the user>" (only if response_type is "answer"),
    "reasoning": "<brief explanation of your choice>"
}

Examples of ALLOWED commands:
- cat /etc/os-release
- dpkg-query -W -f='${Version}' python3
- pip3 show numpy
- pip3 list
- ls -la /usr/bin/python*
- uname -a
- lscpu
- free -h
- df -h
- ps aux | grep python
- apt-cache show nginx
- systemctl status nginx (read-only status check)

Examples of BLOCKED commands (NEVER use these):
- sudo anything
- apt install/remove
- pip install/uninstall
- rm, mv, cp, mkdir, touch
- echo "text" > file
- command1 && command2"""

    # Maximum characters of command output to include in history
    MAX_OUTPUT_CHARS = 2000
    
    def _truncate_output(self, output: str) -> str:
        """Truncate command output to avoid context length issues."""
        if len(output) <= self.MAX_OUTPUT_CHARS:
            return output
        # Keep first and last portions
        half = self.MAX_OUTPUT_CHARS // 2
        return f"{output[:half]}\n\n... [truncated {len(output) - self.MAX_OUTPUT_CHARS} chars] ...\n\n{output[-half:]}"
    
    def _build_iteration_prompt(
        self, 
        question: str, 
        history: list[dict[str, str]]
    ) -> str:
        """Build the prompt for the current iteration."""
        prompt = f"User Question: {question}\n\n"
        
        if history:
            prompt += "Previous commands and results:\n"
            for i, entry in enumerate(history, 1):
                prompt += f"\n--- Attempt {i} ---\n"
                prompt += f"Command: {entry['command']}\n"
                if entry['success']:
                    truncated_output = self._truncate_output(entry['output'])
                    prompt += f"Output:\n{truncated_output}\n"
                else:
                    prompt += f"Error: {entry['error']}\n"
            prompt += "\n"
            prompt += "Based on the above results, either provide another command to gather more information, or provide the final answer.\n"
        else:
            prompt += "Generate a shell command to gather the information needed to answer this question.\n"
        
        prompt += "\nRespond with a JSON object as specified in the system prompt."
        return prompt

    def _parse_llm_response(self, response_text: str) -> SystemCommand:
        """Parse the LLM response into a SystemCommand object."""
        # Try to extract JSON from the response
        response_text = response_text.strip()
        
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(response_text)
            return SystemCommand(**data)
        except (json.JSONDecodeError, Exception) as e:
            # If parsing fails, treat it as a direct answer
            return SystemCommand(
                response_type=LLMResponseType.ANSWER,
                answer=response_text,
                reasoning="Could not parse structured response, treating as direct answer"
            )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the response text."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            try:
                content = response.choices[0].message.content or ""
            except (IndexError, AttributeError):
                content = ""
            return content.strip()
            
        elif self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            try:
                text = getattr(response.content[0], "text", None) or ""
            except (IndexError, AttributeError):
                text = ""
            return text.strip()
            
        elif self.provider == "ollama":
            import urllib.request

            url = f"{self.ollama_url}/api/generate"
            prompt = f"{system_prompt}\n\n{user_prompt}"

            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            }).encode("utf-8")

            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
                
        elif self.provider == "fake":
            # For testing - return a simple answer
            fake_response = os.environ.get("CORTEX_FAKE_RESPONSE", "")
            if fake_response:
                return fake_response
            return json.dumps({
                "response_type": "answer",
                "answer": "Test mode response",
                "reasoning": "Fake provider for testing"
            })
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def ask(self, question: str) -> str:
        """Ask a natural language question about the system.

        Uses an agentic loop to execute read-only commands and gather information
        to answer the user's question.

        Args:
            question: Natural language question

        Returns:
            Human-readable answer string

        Raises:
            ValueError: If question is empty
            RuntimeError: If LLM API call fails
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        question = question.strip()
        system_prompt = self._get_system_prompt()
        
        # Cache lookup
        cache_key = f"ask:v2:{question}"
        if self.cache is not None:
            cached = self.cache.get_commands(
                prompt=cache_key,
                provider=self.provider,
                model=self.model,
                system_prompt=system_prompt,
            )
            if cached is not None and len(cached) > 0:
                return cached[0]

        # Agentic loop
        history: list[dict[str, str]] = []
        tried_commands: list[str] = []
        
        if self.debug:
            self._debug_print("Ask Query", f"Question: {question}", style="cyan")
        
        for iteration in range(self.MAX_ITERATIONS):
            if self.debug:
                self._debug_print(
                    f"Iteration {iteration + 1}/{self.MAX_ITERATIONS}",
                    f"Calling LLM ({self.provider}/{self.model})...",
                    style="blue"
                )
            
            # Build prompt with history
            user_prompt = self._build_iteration_prompt(question, history)
            
            # Call LLM
            try:
                response_text = self._call_llm(system_prompt, user_prompt)
            except Exception as e:
                raise RuntimeError(f"LLM API call failed: {str(e)}")
            
            if self.debug:
                self._debug_print("LLM Raw Response", response_text[:500] + ("..." if len(response_text) > 500 else ""), style="dim")
            
            # Parse response
            parsed = self._parse_llm_response(response_text)
            
            if self.debug:
                self._debug_print(
                    "LLM Parsed Response",
                    f"Type: {parsed.response_type.value}\n"
                    f"Reasoning: {parsed.reasoning}\n"
                    f"Command: {parsed.command or 'N/A'}\n"
                    f"Answer: {(parsed.answer[:100] + '...') if parsed.answer and len(parsed.answer) > 100 else parsed.answer or 'N/A'}",
                    style="yellow"
                )
            
            # If LLM provides a final answer, return it
            if parsed.response_type == LLMResponseType.ANSWER:
                answer = parsed.answer or "Unable to determine an answer."
                
                if self.debug:
                    self._debug_print("Final Answer", answer, style="green")
                
                # Cache the response
                if self.cache is not None and answer:
                    try:
                        self.cache.put_commands(
                            prompt=cache_key,
                            provider=self.provider,
                            model=self.model,
                            system_prompt=system_prompt,
                            commands=[answer],
                        )
                    except (OSError, sqlite3.Error):
                        pass
                
                return answer
            
            # LLM wants to execute a command
            if parsed.command:
                command = parsed.command
                tried_commands.append(command)
                
                if self.debug:
                    self._debug_print("Executing Command", f"$ {command}", style="magenta")
                
                # Validate and execute the command
                success, stdout, stderr = CommandValidator.execute_command(command)
                
                if self.debug:
                    if success:
                        output_preview = stdout[:1000] + ("..." if len(stdout) > 1000 else "") if stdout else "(empty output)"
                        self._debug_print("Command Output (SUCCESS)", output_preview, style="green")
                    else:
                        self._debug_print("Command Output (FAILED)", f"Error: {stderr}", style="red")
                
                history.append({
                    "command": command,
                    "success": success,
                    "output": stdout if success else "",
                    "error": stderr if not success else "",
                })
        
        # Max iterations reached
        commands_list = "\n".join(f"  - {cmd}" for cmd in tried_commands)
        result = f"Could not find an answer after {self.MAX_ITERATIONS} attempts.\n\nTried commands:\n{commands_list}"
        
        if self.debug:
            self._debug_print("Max Iterations Reached", result, style="red")
        
        return result
