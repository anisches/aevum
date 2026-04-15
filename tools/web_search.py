import asyncio
from typing import Any

from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for up-to-date information before answering."

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results

    def _search_sync(self, query: str) -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=self.max_results))

    async def run(self, query: str, **_: Any) -> ToolResult:
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._search_sync, query
            )
            if not results:
                return ToolResult(output="no results found")
            formatted = "\n\n".join(
                f"[{i+1}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
                for i, r in enumerate(results)
            )
            return ToolResult(output=formatted)
        except Exception as exc:
            return ToolResult(output=None, error=str(exc))

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        }
