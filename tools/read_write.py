from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file at the given path."

    async def run(self, path: str, **_: Any) -> ToolResult:
        try:
            content = Path(path).read_text()
            return ToolResult(output=content)
        except Exception as exc:
            return ToolResult(output=None, error=str(exc))

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                },
                "required": ["path"],
            },
        }


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file at the given path. Creates the file if it does not exist."

    async def run(self, path: str, content: str, **_: Any) -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return ToolResult(output=f"written to {path}")
        except Exception as exc:
            return ToolResult(output=None, error=str(exc))

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        }
