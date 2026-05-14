import re

COMMAND_PATTERNS = [
    r'(?:npm|npx|yarn|pnpm)\s+(?:run|install|add|build|test|start|dev|lint|exec)\s*\S*',
    r'pip\s+(?:install|uninstall|list|show|freeze|download)\s*\S*',
    r'python\d?\s+(?:-m\s+\S+|script\.py|\S+\.py)',
    r'git\s+(?:clone|commit|push|pull|add|checkout|branch|merge|rebase|log|status|diff|stash|tag|init|remote)',
    r'uvicorn\s+\S+:\S+(?:\s+--reload)?',
    r'gunicorn\s+\S+:\S+',
    r'docker\s+(?:run|build|compose|ps|exec|logs|pull|push|stop|rm|images)',
    r'kubectl\s+\S+',
    r'conda\s+(?:install|create|activate|deactivate|env|list)',
    r'curl\s+[-/\w.:]+\s*\S*',
    r'wget\s+\S+',
    r'make\s+\S*',
    r'echo\s+["\']?[^"\']+["\']?',
    r'cat\s+[/\w.\-]+\S*',
    r'grep\s+[-rni]+\s+\S+',
    r'chmod\s+\d+\s+\S+',
    r'ssh\s+\S+@\S+',
    r'scp\s+\S+\s+\S+',
    r'sudo\s+\S+',
    r'\.\/[\w./-]+',
]


def extract_commands(text: str) -> list[str]:
    seen: set[str] = set()
    commands: list[str] = []
    for pat in COMMAND_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            cmd = m.group(0).strip().rstrip(".,;:'\"")
            if cmd and cmd not in seen and len(cmd) >= 3:
                seen.add(cmd)
                commands.append(cmd)
    return commands
