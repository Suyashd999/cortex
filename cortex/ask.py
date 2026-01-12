"""Natural language query interface for Cortex.

Handles user questions about installed packages, configurations,
and system state using an agentic LLM loop with command execution.

The --do mode enables write and execute capabilities with user confirmation
and privilege management.
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
    DO_COMMANDS = "do_commands"  # For --do mode: commands that modify the system


class DoCommand(BaseModel):
    """A single command for --do mode with explanation."""
    command: str = Field(description="The shell command to execute")
    purpose: str = Field(description="Brief explanation of what this command does")
    requires_sudo: bool = Field(default=False, description="Whether this command requires sudo")


class SystemCommand(BaseModel):
    """Pydantic model for a system command to be executed.
    
    The LLM must return either a command to execute for data gathering,
    or a final answer to the user's question.
    In --do mode, it can also return a list of commands to execute.
    """
    response_type: LLMResponseType = Field(
        description="Whether this is a command to execute, a final answer, or do commands"
    )
    command: str | None = Field(
        default=None,
        description="The shell command to execute (only for response_type='command')"
    )
    answer: str | None = Field(
        default=None,
        description="The final answer to the user (only for response_type='answer')"
    )
    do_commands: list[DoCommand] | None = Field(
        default=None,
        description="List of commands to execute (only for response_type='do_commands')"
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
    
    @field_validator("do_commands")
    @classmethod
    def validate_do_commands_not_empty(cls, v: list[DoCommand] | None, info) -> list[DoCommand] | None:
        if info.data.get("response_type") == LLMResponseType.DO_COMMANDS:
            if not v or len(v) == 0:
                raise ValueError("do_commands cannot be empty when response_type is 'do_commands'")
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
        "dpkg-query", "dpkg", "apt-cache", "apt-mark", "apt-config", "aptitude", "apt",
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
        # Container/virtualization info
        "docker", "podman", "kubectl", "crictl", "nerdctl",
        "lxc-ls", "virsh", "vboxmanage",
        # Development tools (version checks)
        "git", "node", "nodejs", "deno", "bun", "ruby", "perl", "php", "java", "javac",
        "rustc", "gcc", "g++", "clang", "clang++", "make", "cmake", "ninja", "meson",
        "dotnet", "mono", "swift", "kotlin", "scala", "groovy", "gradle", "mvn", "ant",
        # Database clients (info/version)
        "mysql", "psql", "sqlite3", "mongosh", "redis-cli",
        # Web/network tools
        "curl", "wget", "httpie", "openssl", "ssh", "scp", "rsync",
        # Cloud CLIs
        "aws", "gcloud", "az", "doctl", "linode-cli", "vultr-cli",
        "terraform", "ansible", "vagrant", "packer",
        # Other common tools
        "jq", "yq", "xmllint", "ffmpeg", "ffprobe", "imagemagick", "convert",
        "gh", "hub", "lab",  # GitHub/GitLab CLIs
        "snap", "flatpak",  # For version/list only
        "systemd-analyze", "bootctl",
    }
    
    # Version check flags - these make ANY command safe (read-only)
    VERSION_FLAGS: set[str] = {
        "--version", "-v", "-V", "--help", "-h", "-help",
        "version", "help", "--info", "-version",
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
        # Container tools - block modifying commands
        "docker": {"run", "exec", "build", "push", "pull", "rm", "rmi", "kill", "stop", "start",
                   "restart", "pause", "unpause", "create", "commit", "tag", "load", "save",
                   "import", "export", "login", "logout", "network", "volume", "system", "prune"},
        "podman": {"run", "exec", "build", "push", "pull", "rm", "rmi", "kill", "stop", "start",
                   "restart", "pause", "unpause", "create", "commit", "tag", "load", "save",
                   "import", "export", "login", "logout", "network", "volume", "system", "prune"},
        "kubectl": {"apply", "create", "delete", "edit", "patch", "replace", "scale", "exec",
                    "run", "expose", "set", "rollout", "drain", "cordon", "uncordon", "taint"},
        # Git - block modifying commands
        "git": {"push", "commit", "add", "rm", "mv", "reset", "revert", "merge", "rebase",
                "checkout", "switch", "restore", "stash", "clean", "init", "clone", "pull",
                "fetch", "cherry-pick", "am", "apply"},
        # Cloud CLIs - block modifying commands
        "aws": {"s3", "ec2", "iam", "lambda", "rds", "ecs", "eks"},  # Block service commands (allow sts, configure list)
        "gcloud": {"compute", "container", "functions", "run", "sql", "storage"},
        # Snap/Flatpak - block modifying commands  
        "snap": {"install", "remove", "refresh", "revert", "enable", "disable", "set", "unset"},
        "flatpak": {"install", "uninstall", "update", "repair"},
    }
    
    # Commands that are completely blocked (never allowed, even with --version)
    BLOCKED_COMMANDS: set[str] = {
        # Dangerous/destructive
        "rm", "rmdir", "unlink", "shred",
        "mv", "cp", "install", "mkdir", "touch",
        # Editors (sed is allowed for text processing, redirections are blocked separately)
        "nano", "vim", "vi", "emacs", "ed",
        # Package modification (apt-get is dangerous, apt is allowed with restrictions)
        "apt-get", "dpkg-reconfigure", "update-alternatives",
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
        
        # Check if this is a version/help check - these are always safe
        # Allow ANY command if it only has version/help flags
        if cmd_args and all(arg in cls.VERSION_FLAGS for arg in cmd_args):
            return True, ""  # Safe: just checking version/help
        
        # Also allow if first arg is a version flag (e.g., "docker --version" or "git version")
        if cmd_args and cmd_args[0] in cls.VERSION_FLAGS:
            return True, ""  # Safe: version/help check
        
        # Check if command is completely blocked (unless it's a version check)
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
        
        # Special handling for docker/podman - allow info and list commands
        if base_cmd in ["docker", "podman"]:
            if cmd_args:
                allowed_docker_cmds = {
                    "ps", "images", "info", "version", "inspect", "logs", "top", "stats",
                    "port", "diff", "history", "search", "events", "container", "image",
                    "--version", "-v", "help", "--help",
                }
                # Also allow "container ls", "image ls", etc.
                if cmd_args[0] not in allowed_docker_cmds:
                    return False, f"docker command '{cmd_args[0]}' is not allowed - only read-only commands like 'ps', 'images', 'info', 'inspect', 'logs' are permitted"
                # Check container/image subcommands
                if cmd_args[0] in ["container", "image"] and len(cmd_args) > 1:
                    allowed_sub = {"ls", "list", "inspect", "history", "prune"}  # prune for info only
                    if cmd_args[1] not in allowed_sub and cmd_args[1] not in cls.VERSION_FLAGS:
                        return False, f"docker {cmd_args[0]} '{cmd_args[1]}' is not allowed - only ls, list, inspect are permitted"
        
        # Special handling for kubectl - allow get, describe, logs
        if base_cmd == "kubectl":
            if cmd_args:
                allowed_kubectl_cmds = {
                    "get", "describe", "logs", "top", "cluster-info", "config", "version",
                    "api-resources", "api-versions", "explain", "auth",
                    "--version", "-v", "help", "--help",
                }
                if cmd_args[0] not in allowed_kubectl_cmds:
                    return False, f"kubectl command '{cmd_args[0]}' is not allowed - only read-only commands like 'get', 'describe', 'logs' are permitted"
        
        # Special handling for git - allow status, log, show, diff, branch, remote, config (get)
        if base_cmd == "git":
            if cmd_args:
                allowed_git_cmds = {
                    "status", "log", "show", "diff", "branch", "remote", "tag", "describe",
                    "ls-files", "ls-tree", "ls-remote", "rev-parse", "rev-list", "cat-file",
                    "config", "shortlog", "blame", "annotate", "grep", "reflog",
                    "version", "--version", "-v", "help", "--help",
                }
                if cmd_args[0] not in allowed_git_cmds:
                    return False, f"git command '{cmd_args[0]}' is not allowed - only read-only commands like 'status', 'log', 'diff', 'branch' are permitted"
                # Block git config --set/--add
                if cmd_args[0] == "config" and any(a in cmd_args for a in ["--add", "--unset", "--remove-section", "--rename-section"]):
                    return False, "git config modifications are not allowed"
        
        # Special handling for snap/flatpak - allow list and info commands
        if base_cmd == "snap":
            if cmd_args:
                allowed_snap = {"list", "info", "find", "version", "connections", "services", "logs", "--version", "help", "--help"}
                if cmd_args[0] not in allowed_snap:
                    return False, f"snap command '{cmd_args[0]}' is not allowed - only list, info, find are permitted"
        
        if base_cmd == "flatpak":
            if cmd_args:
                allowed_flatpak = {"list", "info", "search", "remote-ls", "remotes", "history", "--version", "help", "--help"}
                if cmd_args[0] not in allowed_flatpak:
                    return False, f"flatpak command '{cmd_args[0]}' is not allowed - only list, info, search are permitted"
        
        # Special handling for AWS CLI - allow read-only commands
        if base_cmd == "aws":
            if cmd_args:
                allowed_aws = {"--version", "help", "--help", "sts", "configure"}
                # sts get-caller-identity is safe, configure list is safe
                if cmd_args[0] not in allowed_aws:
                    return False, f"aws command '{cmd_args[0]}' is not allowed - use 'sts get-caller-identity' or 'configure list' for read-only queries"
        
        # Special handling for apt - only allow list, show, search, policy, depends
        if base_cmd == "apt":
            if cmd_args:
                allowed_apt = {"list", "show", "search", "policy", "depends", "rdepends", "madison", "--version", "help", "--help"}
                if cmd_args[0] not in allowed_apt:
                    return False, f"apt command '{cmd_args[0]}' is not allowed - only list, show, search, policy are permitted for read-only queries"
            else:
                return False, "apt requires a subcommand like 'list', 'show', or 'search'"
        
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
    
    In --do mode, the handler can execute write and modify commands with
    user confirmation and privilege management.
    """

    MAX_ITERATIONS = 5
    MAX_DO_ITERATIONS = 15  # More iterations for --do mode since it's solving problems

    def __init__(
        self,
        api_key: str,
        provider: str = "claude",
        model: str | None = None,
        debug: bool = False,
        do_mode: bool = False,
    ):
        """Initialize the ask handler.

        Args:
            api_key: API key for the LLM provider
            provider: Provider name ("openai", "claude", or "ollama")
            model: Optional model name override
            debug: Enable debug output to shell
            do_mode: Enable write/execute mode with user confirmation
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model or self._default_model()
        self.debug = debug
        self.do_mode = do_mode
        
        # Import rich console for debug output
        if self.debug:
            from rich.console import Console
            from rich.panel import Panel
            self._console = Console()
        else:
            self._console = None
        
        # Initialize DoHandler for --do mode
        self._do_handler = None
        if self.do_mode:
            try:
                from cortex.do_runner import DoHandler
                # Pass LLM callback so DoHandler can make LLM calls for interactive session
                self._do_handler = DoHandler(llm_callback=self._call_llm_for_do)
            except (ImportError, OSError, Exception) as e:
                # Log error but don't fail - do mode just won't work
                if self.debug and self._console:
                    self._console.print(f"[yellow]Warning: Could not initialize DoHandler: {e}[/yellow]")
                pass

        # Initialize cache
        try:
            from cortex.semantic_cache import SemanticCache

            self.cache: SemanticCache | None = SemanticCache()
        except (ImportError, OSError, sqlite3.OperationalError, Exception):
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
        if self.do_mode:
            return self._get_do_mode_system_prompt()
        return self._get_read_only_system_prompt()
    
    def _get_read_only_system_prompt(self) -> str:
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

CRITICAL: You must respond with ONLY a JSON object - no other text before or after.
Do NOT include explanations outside the JSON. Put all reasoning inside the "reasoning" field.

JSON format:
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
    
    def _get_do_mode_system_prompt(self) -> str:
        return """You are a Linux system assistant that can READ, WRITE, and EXECUTE commands to solve problems.

You are in DO MODE - you have the ability to make changes to the system to solve the user's problem.

Your task is to:
1. Understand the user's problem or request
2. Quickly gather essential information (1-3 read commands MAX)
3. Plan and propose a solution with specific commands using "do_commands"
4. Execute the solution with the user's permission
5. Handle failures gracefully with repair attempts

CRITICAL WORKFLOW RULES:
- DO NOT spend more than 3-4 iterations gathering information
- After gathering basic system info (OS, existing packages), IMMEDIATELY propose do_commands
- If you know how to install/configure something, propose do_commands right away
- Be action-oriented: the user wants you to DO something, not just analyze
- You can always gather more info AFTER the user approves the commands if needed

WORKFLOW:
1. Quickly gather essential info (OS version, if package exists) - MAX 2-3 commands
2. IMMEDIATELY propose "do_commands" with your installation/setup plan
3. The do_commands will be shown to the user for approval before execution
4. Commands are executed using a TASK TREE system with auto-repair capabilities:
   - If a command fails, Cortex will automatically diagnose the error
   - Repair sub-tasks may be spawned and executed with additional permission requests
   - Terminal monitoring is available during manual intervention
5. After execution, verify the changes worked and provide a final "answer"
6. If execution_failures appear in history, propose alternative solutions

CRITICAL: You must respond with ONLY a JSON object - no other text before or after.
Do NOT include explanations outside the JSON. Put all reasoning inside the "reasoning" field.

For gathering information (read-only):
{
    "response_type": "command",
    "command": "<shell command to execute>",
    "reasoning": "<why you need this information>"
}

For proposing changes (write/execute):
{
    "response_type": "do_commands",
    "do_commands": [
        {
            "command": "<shell command>",
            "purpose": "<what this command does and why>",
            "requires_sudo": true/false
        }
    ],
    "reasoning": "<overall explanation of the solution>"
}

For final answer:
{
    "response_type": "answer",
    "answer": "<final response to user, summarizing what was done>",
    "reasoning": "<explanation>"
}

For proposing repair commands after failures:
{
    "response_type": "do_commands",
    "do_commands": [
        {
            "command": "<diagnostic or repair command>",
            "purpose": "<why this will help fix the previous failure>",
            "requires_sudo": true/false
        }
    ],
    "reasoning": "<analysis of what went wrong and how this will fix it>"
}

HANDLING FAILURES:
- When you see "execution_failures" in history, analyze the error messages carefully
- Common errors and their fixes:
  * "Permission denied" → Add sudo, check ownership, or run with elevated privileges
  * "No such file or directory" → Create parent directories first (mkdir -p)
  * "Command not found" → Install the package (apt install)
  * "Service not running" → Start the service first (systemctl start)
  * "Configuration syntax error" → Read config file, find and fix the error
- Always provide detailed reasoning when proposing repairs
- If the original approach won't work, suggest an alternative approach
- You may request multiple rounds of commands to diagnose and fix issues

IMPORTANT RULES:
- BE ACTION-ORIENTED: After 2-3 info commands, propose do_commands immediately
- DO NOT over-analyze: You have enough info once you know the OS and if basic packages exist
- For installation tasks: Propose the installation commands right away
- For do_commands, each command should be atomic and specific
- Always include a clear purpose for each command
- Mark requires_sudo: true if the command needs root privileges
- Be careful with destructive commands - always explain what they do
- After making changes, verify they worked before giving final answer
- If something fails, diagnose and try alternative approaches
- Multiple permission requests may be made during a single session for repair commands

ANTI-PATTERNS TO AVOID:
- Don't keep gathering info for more than 3 iterations
- Don't check every possible thing before proposing a solution
- Don't be overly cautious - the user wants action
- If you know how to solve the problem, propose do_commands NOW

PROTECTED PATHS (will require user authentication):
- /etc/* - System configuration
- /boot/* - Boot configuration  
- /usr/bin, /usr/sbin, /sbin, /bin - System binaries
- /root - Root home directory
- /var/log, /var/lib/apt - System data

COMMAND RESTRICTIONS:
- Use SINGLE commands only - no chaining with &&, ||, or ;
- Use pipes (|) sparingly and only for filtering
- No output redirection (>, >>) in read commands
- If you need multiple commands, return them separately in sequence

Examples of READ commands:
- cat /etc/nginx/nginx.conf
- ls -la /var/log/
- systemctl status nginx
- grep -r "error" /var/log/syslog
- dpkg -l | grep nginx
- apt list --installed | grep docker (use apt list, not apt install)

Examples of WRITE/EXECUTE commands (use with do_commands):
- echo 'server_name example.com;' >> /etc/nginx/sites-available/default
- systemctl restart nginx
- apt install -y nginx
- chmod 755 /var/www/html
- mkdir -p /etc/myapp
- cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

Examples of REPAIR commands after failures:
- sudo chown -R $USER:$USER /path/to/file  # Fix ownership issues
- sudo mkdir -p /path/to/directory  # Create missing directories
- sudo apt install -y missing-package  # Install missing dependencies
- journalctl -u service-name -n 50 --no-pager  # Diagnose service issues"""

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
                # Handle execution_failures context from do_commands
                if entry.get("type") == "execution_failures":
                    prompt += f"\n--- EXECUTION FAILURES (Need Repair) ---\n"
                    prompt += f"Message: {entry.get('message', 'Commands failed')}\n"
                    for fail in entry.get("failures", []):
                        prompt += f"\nFailed Command: {fail.get('command', 'unknown')}\n"
                        prompt += f"Purpose: {fail.get('purpose', 'unknown')}\n"
                        prompt += f"Error: {fail.get('error', 'unknown')}\n"
                    prompt += "\nPlease analyze these failures and propose repair commands or alternative approaches.\n"
                    continue
                
                # Handle regular commands
                prompt += f"\n--- Attempt {i} ---\n"
                
                # Check if this is a do_command execution result
                if "executed_by" in entry:
                    prompt += f"Command (executed by {entry['executed_by']}): {entry.get('command', 'unknown')}\n"
                    prompt += f"Purpose: {entry.get('purpose', 'unknown')}\n"
                    if entry.get('success'):
                        truncated_output = self._truncate_output(entry.get('output', ''))
                        prompt += f"Status: SUCCESS\nOutput:\n{truncated_output}\n"
                    else:
                        prompt += f"Status: FAILED\nError: {entry.get('error', 'unknown')}\n"
                else:
                    prompt += f"Command: {entry.get('command', 'unknown')}\n"
                    if entry.get('success'):
                        truncated_output = self._truncate_output(entry.get('output', ''))
                        prompt += f"Output:\n{truncated_output}\n"
                    else:
                        prompt += f"Error: {entry.get('error', 'unknown')}\n"
            
            prompt += "\n"
            
            # Check if there were recent failures
            has_failures = any(
                e.get("type") == "execution_failures" or 
                (e.get("executed_by") and not e.get("success"))
                for e in history[-5:]  # Check last 5 entries
            )
            
            if has_failures:
                prompt += "IMPORTANT: There were command failures. Please:\n"
                prompt += "1. Analyze the error messages to understand what went wrong\n"
                prompt += "2. Propose repair commands using 'do_commands' response type\n"
                prompt += "3. Or suggest an alternative approach if the original won't work\n"
            else:
                prompt += "Based on the above results, either provide another command to gather more information, or provide the final answer.\n"
        else:
            prompt += "Generate a shell command to gather the information needed to answer this question.\n"
        
        prompt += "\nRespond with a JSON object as specified in the system prompt."
        return prompt

    def _parse_llm_response(self, response_text: str) -> SystemCommand:
        """Parse the LLM response into a SystemCommand object."""
        # Try to extract JSON from the response
        original_text = response_text.strip()
        response_text = original_text
        
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) >= 2:
                response_text = parts[1].split("```")[0].strip()
        
        # Try direct JSON parsing first
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text (LLM sometimes adds prose before/after)
            json_match = re.search(r'\{[\s\S]*"response_type"[\s\S]*\}', original_text)
            if json_match:
                try:
                    # Find the complete JSON object by matching braces
                    json_str = json_match.group()
                    # Balance braces to get complete JSON
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end > 0:
                        json_str = json_str[:json_end]
                    
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # If still fails, treat as direct answer
                    return SystemCommand(
                        response_type=LLMResponseType.ANSWER,
                        answer=original_text,
                        reasoning="Could not parse structured response, treating as direct answer"
                    )
            else:
                # No JSON found, treat as direct answer
                return SystemCommand(
                    response_type=LLMResponseType.ANSWER,
                    answer=original_text,
                    reasoning="No JSON structure found, treating as direct answer"
                )
        
        try:
            # Handle do_commands - convert dict list to DoCommand objects
            if data.get("response_type") == "do_commands" and "do_commands" in data:
                data["do_commands"] = [
                    DoCommand(**cmd) if isinstance(cmd, dict) else cmd 
                    for cmd in data["do_commands"]
                ]
            
            return SystemCommand(**data)
        except Exception as e:
            # If SystemCommand creation fails, treat it as a direct answer
            return SystemCommand(
                response_type=LLMResponseType.ANSWER,
                answer=original_text,
                reasoning=f"Failed to create SystemCommand: {e}"
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

    def _call_llm_for_do(self, user_request: str, context: dict | None = None) -> dict:
        """Call LLM to process a natural language request for the interactive session.
        
        This is passed to DoHandler as a callback so it can make LLM calls
        during the interactive session.
        
        Args:
            user_request: The user's natural language request
            context: Optional context dict with executed_commands, session_actions, etc.
        
        Returns:
            Dict with either:
            - {"response_type": "do_commands", "do_commands": [...], "reasoning": "..."}
            - {"response_type": "answer", "answer": "...", "reasoning": "..."}
            - {"response_type": "command", "command": "...", "reasoning": "..."}
        """
        context = context or {}
        
        system_prompt = """You are a Linux system assistant in an interactive session.
The user has just completed some tasks and now wants to do something else.

Based on their request, decide what to do:
1. If they want to EXECUTE commands (install, configure, start, etc.), respond with do_commands
2. If they want INFORMATION (show, explain, how to), respond with an answer
3. If they want to RUN a single read-only command, respond with command

CRITICAL: Respond with ONLY a JSON object - no other text.

For executing commands:
{
    "response_type": "do_commands",
    "do_commands": [
        {"command": "...", "purpose": "...", "requires_sudo": true/false}
    ],
    "reasoning": "..."
}

For providing information:
{
    "response_type": "answer", 
    "answer": "...",
    "reasoning": "..."
}

For running a read-only command:
{
    "response_type": "command",
    "command": "...",
    "reasoning": "..."
}
"""
        
        # Build context-aware prompt
        user_prompt = f"Context:\n"
        if context.get("original_query"):
            user_prompt += f"- Original task: {context['original_query']}\n"
        if context.get("executed_commands"):
            user_prompt += f"- Commands already executed: {', '.join(context['executed_commands'][:5])}\n"
        if context.get("session_actions"):
            user_prompt += f"- Actions in this session: {', '.join(context['session_actions'][:3])}\n"
        
        user_prompt += f"\nUser request: {user_request}\n"
        user_prompt += "\nRespond with a JSON object."
        
        try:
            response_text = self._call_llm(system_prompt, user_prompt)
            
            # Parse the response
            parsed = self._parse_llm_response(response_text)
            
            # Convert to dict
            result = {
                "response_type": parsed.response_type.value,
                "reasoning": parsed.reasoning,
            }
            
            if parsed.response_type == LLMResponseType.DO_COMMANDS and parsed.do_commands:
                result["do_commands"] = [
                    {"command": cmd.command, "purpose": cmd.purpose, "requires_sudo": cmd.requires_sudo}
                    for cmd in parsed.do_commands
                ]
            elif parsed.response_type == LLMResponseType.COMMAND and parsed.command:
                result["command"] = parsed.command
            elif parsed.response_type == LLMResponseType.ANSWER and parsed.answer:
                result["answer"] = parsed.answer
            
            return result
            
        except Exception as e:
            return {
                "response_type": "error",
                "error": str(e),
            }

    def ask(self, question: str) -> str:
        """Ask a natural language question about the system.

        Uses an agentic loop to execute read-only commands and gather information
        to answer the user's question.
        
        In --do mode, can also execute write/modify commands with user confirmation.

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
        
        # Don't cache in do_mode (each run is unique)
        cache_key = f"ask:v2:{question}"
        if self.cache is not None and not self.do_mode:
            cached = self.cache.get_commands(
                prompt=cache_key,
                provider=self.provider,
                model=self.model,
                system_prompt=system_prompt,
            )
            if cached is not None and len(cached) > 0:
                return cached[0]

        # Agentic loop
        history: list[dict[str, Any]] = []
        tried_commands: list[str] = []
        max_iterations = self.MAX_DO_ITERATIONS if self.do_mode else self.MAX_ITERATIONS
        
        if self.debug:
            mode_str = "[DO MODE]" if self.do_mode else ""
            self._debug_print("Ask Query", f"{mode_str} Question: {question}", style="cyan")
        
        # Import console for progress output
        from rich.console import Console
        loop_console = Console()
        
        for iteration in range(max_iterations):
            if self.debug:
                self._debug_print(
                    f"Iteration {iteration + 1}/{max_iterations}",
                    f"Calling LLM ({self.provider}/{self.model})...",
                    style="blue"
                )
            
            # Show progress to user (even without --debug)
            if self.do_mode and iteration > 0:
                loop_console.print(f"[dim]🔄 Analyzing results... (step {iteration + 1}/{max_iterations})[/dim]")
            
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
                    f"Do Commands: {len(parsed.do_commands) if parsed.do_commands else 0}\n"
                    f"Answer: {(parsed.answer[:100] + '...') if parsed.answer and len(parsed.answer) > 100 else parsed.answer or 'N/A'}",
                    style="yellow"
                )
            
            # Show what the LLM decided to do
            if self.do_mode and not self.debug:
                if parsed.response_type == LLMResponseType.COMMAND:
                    loop_console.print(f"[cyan]🔍 Gathering info:[/cyan] {parsed.command}")
                elif parsed.response_type == LLMResponseType.DO_COMMANDS:
                    loop_console.print(f"[cyan]📋 Ready to execute {len(parsed.do_commands or [])} command(s)[/cyan]")
            
            # If LLM provides a final answer, return it
            if parsed.response_type == LLMResponseType.ANSWER:
                answer = parsed.answer or "Unable to determine an answer."
                
                if self.debug:
                    self._debug_print("Final Answer", answer, style="green")
                
                # Cache the response (not in do_mode)
                if self.cache is not None and answer and not self.do_mode:
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
            
            # Handle do_commands in --do mode
            if parsed.response_type == LLMResponseType.DO_COMMANDS and self.do_mode:
                result = self._handle_do_commands(parsed, question, history)
                if result is not None:
                    # If user declined, continue the loop to provide manual instructions
                    if result.startswith("USER_DECLINED:"):
                        # Add to history that user declined
                        history.append({
                            "type": "do_commands_declined",
                            "commands": [(c.command, c.purpose) for c in (parsed.do_commands or [])],
                            "message": "User declined automatic execution. Manual instructions were provided.",
                        })
                        continue
                    return result
            
            # LLM wants to execute a read-only command
            if parsed.command:
                command = parsed.command
                tried_commands.append(command)
                
                if self.debug:
                    self._debug_print("Executing Command", f"$ {command}", style="magenta")
                
                # Validate and execute the command
                success, stdout, stderr = CommandValidator.execute_command(command)
                
                # Show execution result to user
                if self.do_mode and not self.debug:
                    if success:
                        output_lines = len(stdout.split('\n')) if stdout else 0
                        loop_console.print(f"[green]   ✓ Got {output_lines} lines of output[/green]")
                    else:
                        loop_console.print(f"[yellow]   ⚠ Command failed: {stderr[:100]}[/yellow]")
                
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
        result = f"Could not find an answer after {max_iterations} attempts.\n\nTried commands:\n{commands_list}"
        
        if self.debug:
            self._debug_print("Max Iterations Reached", result, style="red")
        
        return result
    
    def _handle_do_commands(
        self, 
        parsed: SystemCommand, 
        question: str,
        history: list[dict[str, Any]]
    ) -> str | None:
        """Handle do_commands response type - execute with user confirmation.
        
        Uses task tree execution for advanced auto-repair capabilities:
        - Spawns repair sub-tasks when commands fail
        - Requests additional permissions during execution
        - Monitors terminals during manual intervention
        - Provides detailed failure reasoning
        
        Returns:
            Result string if completed, None if should continue loop,
            or "USER_DECLINED:..." if user declined.
        """
        if not self._do_handler or not parsed.do_commands:
            return None
        
        from rich.console import Console
        console = Console()
        
        # Prepare commands for analysis
        commands = [
            (cmd.command, cmd.purpose) for cmd in parsed.do_commands
        ]
        
        # Analyze for protected paths
        analyzed = self._do_handler.analyze_commands_for_protected_paths(commands)
        
        # Show reasoning
        console.print()
        console.print(f"[bold cyan]🤖 Cortex Analysis:[/bold cyan] {parsed.reasoning}")
        console.print()
        
        # Show task tree preview
        console.print("[dim]📋 Planned tasks:[/dim]")
        for i, (cmd, purpose, protected) in enumerate(analyzed, 1):
            protected_note = f" [yellow](protected: {', '.join(protected)})[/yellow]" if protected else ""
            console.print(f"[dim]   {i}. {cmd[:60]}...{protected_note}[/dim]")
        console.print()
        
        # Request user confirmation
        if self._do_handler.request_user_confirmation(analyzed):
            # User approved - execute using task tree for better error handling
            run = self._do_handler.execute_with_task_tree(analyzed, question)
            
            # Add execution results to history
            for cmd_log in run.commands:
                history.append({
                    "command": cmd_log.command,
                    "success": cmd_log.status.value == "success",
                    "output": cmd_log.output,
                    "error": cmd_log.error,
                    "purpose": cmd_log.purpose,
                    "executed_by": "cortex",
                })
            
            # Check if there were failures that need LLM input
            failures = [c for c in run.commands if c.status.value == "failed"]
            if failures:
                # Add failure context to history for LLM to help with
                failure_summary = []
                for f in failures:
                    failure_summary.append({
                        "command": f.command,
                        "error": f.error[:500] if f.error else "Unknown error",
                        "purpose": f.purpose,
                    })
                
                history.append({
                    "type": "execution_failures",
                    "failures": failure_summary,
                    "message": f"{len(failures)} command(s) failed during execution. Please analyze and suggest fixes.",
                })
                
                # Continue loop so LLM can suggest next steps
                return None
            
            # Return summary for now - LLM will provide final answer in next iteration
            return None
        else:
            # User declined - provide manual instructions with monitoring
            run = self._do_handler.provide_manual_instructions(analyzed, question)
            
            # Return special marker so we continue the loop
            return f"USER_DECLINED: Manual instructions provided. Run ID: {run.run_id}"
