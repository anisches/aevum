import asyncio
import itertools
import json
import os
import sys
import termios
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
CY  = "\033[36m"
GR  = "\033[32m"
RD  = "\033[31m"
GY  = "\033[90m"
MG  = "\033[35m"

CLEAR    = "\033[2J\033[H"
UP1      = "\033[1A"
CLR_LINE = "\033[2K"

_CLI_CONFIG_PATH = Path.home() / ".aevum" / "cli_config.json"

_CODE_WIDTH = 52


def _render_code_block(code_lines: list[str], lang: str) -> list[str]:
    out = []
    label = f" {lang} " if lang else ""
    top   = f"─{label}" + "─" * (_CODE_WIDTH - len(label))
    out.append(f"  {GY}╭{top}╮{R}")
    for ln in code_lines:
        # strip trailing whitespace, leave leading indent intact
        content = ln.rstrip()
        out.append(f"  {GY}│{R}  {CY}{content}{R}")
    out.append(f"  {GY}╰{'─' * (_CODE_WIDTH + 1)}╯{R}")
    return out


def _render_md(text: str) -> str:
    import re
    lines    = text.split("\n")
    out: list[str] = []
    in_code  = False
    code_buf: list[str] = []
    code_lang = ""
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if not in_code:
                in_code   = True
                code_lang = line.strip()[3:].strip()
                code_buf  = []
            else:
                out.extend(_render_code_block(code_buf, code_lang))
                out.append("")
                in_code = False
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        stripped = line.strip()

        if re.fullmatch(r"[-*_]{3,}", stripped):
            out.append(f"  {GY}{'─' * 46}{R}")
            i += 1
            continue

        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                out.append(f"\n  {MG}{B}{title.upper()}{R}")
            elif level == 2:
                out.append(f"\n  {B}{title}{R}")
            else:
                out.append(f"\n  {CY}{title}{R}")
            out.append(f"  {GY}{'─' * min(len(title) + 2, 46)}{R}")
            i += 1
            continue

        m = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if m:
            indent  = len(m.group(1))
            content = _inline_md(m.group(2))
            pad     = "  " * (indent // 2 + 1)
            out.append(f"{pad}{GY}·{R}  {content}")
            i += 1
            continue

        m = re.match(r"^(\s*)(\d+)\.\s+(.*)", line)
        if m:
            indent  = len(m.group(1))
            num     = m.group(2)
            content = _inline_md(m.group(3))
            pad     = "  " * (indent // 2 + 1)
            out.append(f"{pad}{GY}{num}.{R}  {content}")
            i += 1
            continue

        if not stripped:
            out.append("")
            i += 1
            continue

        out.append(f"  {_inline_md(line)}")
        i += 1

    # unclosed code block fallback
    if in_code and code_buf:
        out.extend(_render_code_block(code_buf, code_lang))

    return "\n".join(out)


def _print_response(rendered: str, delay: float = 0.012) -> None:
    """Print rendered markdown line-by-line with a subtle fade-in."""
    for line in rendered.split("\n"):
        print(line)
        time.sleep(delay)


def _inline_md(text: str) -> str:
    import re
    # bold **...**
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{B}{m.group(1)}{R}", text)
    # italic *...* (not preceded/followed by *)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", lambda m: f"{DIM}{m.group(1)}{R}", text)
    # inline code `...`
    text = re.sub(r"`([^`]+)`", lambda m: f"{CY}{m.group(1)}{R}", text)
    return text


BANNER = f"""{CY}{B}
  ╭───────────────────────────────╮
  │   aevum  ·  select model     │
  ╰───────────────────────────────╯{R}
"""

PROVIDER_ICONS  = {"anthropic": "◆", "gemini": "◈", "ollama": "◉"}
PROVIDER_COLORS = {"anthropic": MG, "gemini": CY, "ollama": GR}

_SPIN_WORDS = [
    "thinking", "pondering", "cooking", "brewing",
    "weaving", "conjuring", "reflecting", "crafting",
    "computing", "simmering", "focusing", "processing",
]
_SPIN_CHARS     = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
_FRAMES_PER_WORD = 12


class _Spinner:
    def __init__(self) -> None:
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        words = itertools.cycle(_SPIN_WORDS)
        word  = next(words)
        frame = 0
        while not self._stop.is_set():
            char = next(_SPIN_CHARS)
            print(f"\r  {DIM}{char}{R} {GY}aevum {word}...{R}", end="", flush=True)
            time.sleep(0.13)
            frame += 1
            if frame >= _FRAMES_PER_WORD:
                frame = 0
                word  = next(words)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        print(f"\r{CLR_LINE}", end="", flush=True)


def _echo_off() -> list | None:
    try:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        return old
    except Exception:
        return None


def _echo_on(old: list | None) -> None:
    if old is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _cls() -> None:
    print(CLEAR, end="", flush=True)


def _print_slow(text: str, delay: float = 0.018) -> None:
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def _dot_transition(label: str, steps: int = 3, delay: float = 0.3) -> None:
    print(f"  {DIM}{label}", end="", flush=True)
    for _ in range(steps):
        time.sleep(delay)
        print(".", end="", flush=True)
    print(f"{R}", flush=True)
    time.sleep(0.1)


def _pick(prompt: str, options: list[str], color: str = CY) -> str:
    print(f"\n  {B}{prompt}{R}\n")
    for i, opt in enumerate(options, 1):
        print(f"  {GY}{i}.{R}  {opt}")
    print()
    while True:
        raw = input(f"  {DIM}›{R} ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            print(f"  {UP1}{CLR_LINE}  {color}✓{R}  {B}{chosen}{R}")
            time.sleep(0.15)
            return chosen
        print(f"  {RD}enter a number 1–{len(options)}{R}")


def _parse_args() -> str | None:
    args = sys.argv[1:]
    if "--url" in args:
        idx = args.index("--url")
        if idx + 1 < len(args):
            return args[idx + 1]
    return None


def _load_cli_config() -> dict | None:
    if _CLI_CONFIG_PATH.exists():
        try:
            return json.loads(_CLI_CONFIG_PATH.read_text())
        except Exception:
            return None
    return None


def _save_cli_config(provider: str, model: str, ollama_url: str) -> None:
    _CLI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CLI_CONFIG_PATH.write_text(json.dumps(
        {"provider": provider, "model": model, "ollama_url": ollama_url},
        indent=2,
    ))


def _select_model(provider: str, ollama_url: str) -> str:
    match provider:
        case "anthropic":
            from providers.anthropic import DEFAULT_MODELS
            return _pick("choose a model", DEFAULT_MODELS, color=MG)
        case "gemini":
            from providers.gemini import DEFAULT_MODELS
            return _pick("choose a model", DEFAULT_MODELS, color=CY)
        case "ollama":
            import httpx
            print(f"\n  {DIM}fetching models from {ollama_url}", end="", flush=True)
            try:
                resp   = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"  {GR}✓{R}")
            except Exception as exc:
                print(f"\n  {RD}✗ could not reach ollama: {exc}{R}")
                sys.exit(1)
            if not models:
                print(f"  {RD}no models found — run: ollama pull <model>{R}")
                sys.exit(1)
            return _pick("choose a model", models, color=GR)
        case _:
            print(f"  {RD}unknown provider: {provider}{R}")
            sys.exit(1)


def _configure(ollama_url: str) -> tuple[str, str]:
    provider = _pick("choose a provider", ["anthropic", "gemini", "ollama"])

    match provider:
        case "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print(f"  {RD}ANTHROPIC_API_KEY is not set.{R}")
                sys.exit(1)
        case "gemini":
            if not os.environ.get("GEMINI_API_KEY"):
                print(f"  {RD}GEMINI_API_KEY is not set.{R}")
                sys.exit(1)

    model = _select_model(provider, ollama_url)
    _save_cli_config(provider, model, ollama_url)
    return provider, model


def _connect_transition(provider: str, model: str) -> None:
    icon  = PROVIDER_ICONS.get(provider, "◆")
    color = PROVIDER_COLORS.get(provider, CY)
    print()
    _dot_transition(f"connecting to {provider}", steps=4, delay=0.22)
    time.sleep(0.1)
    print(f"\n  {color}{B}{icon}  {provider}{R}  {GY}·{R}  {model}")
    print(f"\n  {GY}{'─' * 46}{R}")
    print(f"  {DIM}type  /model  to switch  ·  exit  to quit{R}")
    print(f"  {GY}{'─' * 46}{R}\n")
    time.sleep(0.2)


async def _chat_loop(
    provider_name: str,
    model: str,
    ollama_url: str,
) -> None:
    from src.agent import Agent
    from src.config import Config

    def _make_agent(p: str, m: str) -> Agent:
        return Agent(config=Config(provider=p, model=m, ollama_base_url=ollama_url))

    agent = _make_agent(provider_name, model)
    color = PROVIDER_COLORS.get(provider_name, CY)

    while True:
        try:
            user_input = input(f"  {CY}you{R}  {GY}›{R}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {DIM}goodbye.{R}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print(f"\n  {DIM}goodbye.{R}\n")
            break

        if user_input.lower() == "/model":
            provider_name, model = _configure(ollama_url)
            color = PROVIDER_COLORS.get(provider_name, CY)
            agent = _make_agent(provider_name, model)
            _connect_transition(provider_name, model)
            continue

        print()
        spinner         = _Spinner()
        old_echo        = _echo_off()
        spinner_stopped = False

        _STEP_COLORS = {"thought": GY, "action": CY, "observation": GR}

        def on_step(step_type: str, content: str) -> None:
            nonlocal spinner_stopped
            if not spinner_stopped:
                spinner.stop()
                _echo_on(old_echo)
                spinner_stopped = True
            col = _STEP_COLORS.get(step_type, GY)
            print(f"  {col}{DIM}{content[:80]}{R}")

        spinner.start()
        try:
            full_response = ""
            async for chunk in agent.stream(user_input, on_step=on_step):
                full_response += chunk
            if not spinner_stopped:
                spinner.stop()
                _echo_on(old_echo)
            if full_response:
                print(f"\n  {color}aevum{R}  {GY}›{R}\n")
                _print_response(_render_md(full_response))
            meta = agent.last_meta
            if meta and meta.source == "mine":
                print(f"\n  {GR}{DIM}{meta.reason}{R}")
            print(f"\n  {GY}{'╌' * 46}{R}\n")
        except Exception as exc:
            if not spinner_stopped:
                spinner.stop()
                _echo_on(old_echo)
            print(f"\n  {RD}error: {exc}{R}\n")


def main() -> None:
    url_flag   = _parse_args()
    ollama_url = url_flag or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    _cls()
    _print_slow(BANNER, delay=0.004)

    saved = _load_cli_config()

    if saved:
        provider = saved["provider"]
        model    = saved["model"]
        saved_url = saved.get("ollama_url", ollama_url)
        if saved_url != "http://localhost:11434":
            ollama_url = saved_url
        icon  = PROVIDER_ICONS.get(provider, "◆")
        color = PROVIDER_COLORS.get(provider, CY)
        print(f"  {GY}saved config{R}  {color}{icon}  {provider}{R}  {GY}·{R}  {model}\n")
    else:
        provider, model = _configure(ollama_url)

    _connect_transition(provider, model)
    asyncio.run(_chat_loop(provider, model, ollama_url))


if __name__ == "__main__":
    main()
