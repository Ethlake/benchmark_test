"""Structured logging — dual console + JSONL output via structlog.

Call ``configure_logging()`` once at startup (from the CLI callback).
Every module then uses::

    import structlog
    log = structlog.get_logger()
    log.info("event_name", key=value, ...)

Context binding (``log.bind(run_id=...)``) propagates automatically to
all downstream log calls within that context.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import structlog
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── Rich console (stderr, no auto-highlight) ─────────────────────────

_console = Console(stderr=True, highlight=False)

# ── Events to suppress from console (still go to JSONL) ──────────────

_SUPPRESS_EVENTS: frozenset[str] = frozenset(
    {
        "llm_request",
        "llm_response",
        "tool_schemas",
        "system_prompt",
        "initial_obs",
        "env_reset",
        "env_step_start",
        "env_step_end",
        "turn_saved",
        "message_history",
        "token_usage",
        "pointer_update",
        "artifact_registered",
        "state_header",
        "text_nudge_sent",
        "observation",
        "policy_act_start",
        "policy_act_end",
        "tool_result_raw",
        # Internal env/tool events (verbose — keep in JSONL)
        "env_step_internal",
        "profile_load",
        "profile_dataset",
        "profile_stats",
        "profile_text_stats",
        "dataset_load_start",
        "dataset_load_split_dir",
        "dataset_load_disk",
        "dataset_load_file",
        "dataset_load_hub",
        "dataset_select_columns",
        "dataset_row_limit",
        "dataset_loaded",
        "dataset_ensure_full",
        "recipe_computed",
        "vlm_filter_start",
        "vlm_filter_reload",
        "vlm_mix_dataset",
        "vlm_mix_concat",
        "vlm_modify_reload",
        "quality_filter_reload",
    }
)

# ── Phase→style mapping ──────────────────────────────────────────────

_PHASE_STYLES: dict[str, str] = {
    "preselection": "cyan",
    "filter": "magenta",
    "selection": "blue",
    "postselection": "yellow",
    "save": "green",
}

_TOOL_PHASE: dict[str, str] = {
    "profile_datasets": "preselection",
    "load_dataset": "preselection",
    "analyze_requirements": "preselection",
    "compute_mix_ratio": "preselection",
    "vlm_filter": "filter",
    "quality_filter": "filter",
    "vlm_mix": "filter",
    "vlm_modify": "filter",
    "explore_dataset": "selection",
    "dedup_check": "selection",
    "text_quality_check": "selection",
    "balanced_sampling": "selection",
    "transform_dataset": "selection",
    "save_to_disk": "save",
    "save_to_hf": "save",
    "convert_format": "postselection",
    "submit_finetune": "postselection",
    "submit_eval": "postselection",
    "think": "utility",
}

# "utility" phase isn't in _PHASE_STYLES — falls back to "white"

_LEVEL_STYLES: dict[str, str] = {
    "debug": "dim",
    "info": "",
    "warning": "yellow",
    "error": "red",
    "critical": "bold red",
}


class _SuppressEventsFilter(logging.Filter):
    """Hide high-volume events from console while keeping JSONL intact."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = getattr(record, "msg", None)
        if isinstance(msg, dict):
            event = msg.get("event")
            if event in _SUPPRESS_EVENTS:
                return False
            # Suppress tool-entry-point logs (e.g. "profile_datasets", "compute_mix_ratio")
            # — redundant with policy_decision + env_step rendering.
            if event in _TOOL_PHASE:
                return False
        return True


class _RichStreamHandler(logging.StreamHandler):
    """StreamHandler that suppresses empty strings (Rich side-effect renderer)."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        if msg == "":
            return
        super().emit(record)


# ── Pure-string helpers ───────────────────────────────────────────────


def _tool_style(name: str) -> str:
    phase = _TOOL_PHASE.get(name, "")
    return _PHASE_STYLES.get(phase, "white")


def _truncate(s: str, max_len: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s[:max_len] + "..." if len(s) > max_len else s


def _format_args(args: dict) -> str:
    """Compact arg display: key=val, key=val."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _format_value_inline(value: object, max_len: int = 120) -> str:
    """Format one value for compact single-line display."""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    text = text.replace("\n", " ").strip()
    return _truncate(text, max_len)


def _format_value_block(value: object, max_len: int = 800) -> str:
    """Format one value for multi-line block display (best effort JSON pretty-print)."""
    obj: object | None = None
    if isinstance(value, (dict, list)):
        obj = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw[:1] in {"{", "["}:
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                obj = None
        if obj is None:
            return _truncate(raw.replace("\n", " "), max_len)

    if obj is None:
        return _truncate(str(value).replace("\n", " "), max_len)

    pretty = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(pretty) > max_len:
        return _truncate(pretty.replace("\n", " "), max_len)
    return pretty


# ── Rich console renderer ────────────────────────────────────────────


def _agent_console_renderer(
    _logger: logging.Logger,
    name: str,
    event_dict: dict,
) -> str:
    """structlog renderer — prints Rich objects as a side-effect, returns ''."""
    event = event_dict.get("event", "")

    # ── Episode banner ──
    if event == "episode_start":
        ep = event_dict.get("episode", "?")
        total = event_dict.get("total_episodes", "?")
        _console.print()
        _console.print(Rule(f"[bold cyan]EPISODE {ep}/{total}[/bold cyan]", style="cyan"))
        return ""

    if event in ("episode_end", "episode_done"):
        ep = event_dict.get("episode", "?")
        steps = event_dict.get("steps", event_dict.get("iterations", "?"))
        reward = event_dict.get("total_reward")
        label = f"Episode {ep} done — {steps} steps"
        if reward is not None:
            label += f" — reward {reward}"
        _console.print(Rule(f"[dim]{label}[/dim]", style="dim"))
        return ""

    # ── Agent step: tool call ──
    if event in ("agent_step", "env_step"):
        step = event_dict.get("step", "?")
        tool = event_dict.get("tool", "?")
        args = event_dict.get("tool_args", event_dict.get("args", ""))
        wall = event_dict.get("wall_s", event_dict.get("duration_s", 0))
        status = event_dict.get("status", "ok")
        result_preview = event_dict.get("result_preview", "")
        reward = event_dict.get("reward")

        style = _tool_style(tool)
        phase = _TOOL_PHASE.get(tool, "")

        args_str = _format_args(args) if isinstance(args, dict) else _truncate(str(args), 60)

        # Step header: number + phase tag + tool name
        line = Text()
        line.append(f"  {step:>2} ", style="bold")
        if phase:
            line.append(f" {phase} ", style=f"bold white on {style}")
            line.append(" ")
        line.append(tool, style=f"bold {style}")

        # Status badge + timing
        line.append("  ")
        status_upper = str(status).upper()
        if status_upper in ("OK", "TRUE"):
            line.append(" OK ", style="bold white on green")
        elif status_upper in ("ERR", "ERROR", "FALSE"):
            line.append(" ERR ", style="bold white on red")
        else:
            line.append(f" {status_upper} ", style="bold white on yellow")
        line.append(f"  {float(wall):.1f}s", style="dim")
        if reward is not None and reward != 0:
            line.append(f"  r={reward}", style="dim cyan")

        _console.print(line)

        # Args on a sub-line
        if args_str:
            _console.print(f"     [dim]\u2514\u2500 {_truncate(args_str, 90)}[/dim]")
        if result_preview:
            _console.print(f"     [dim]\u2514\u2500 {_truncate(str(result_preview), 90)}[/dim]")
        return ""

    # ── LLM thought ──
    if event == "thought":
        text = event_dict.get("text", "")
        if text:
            display = _truncate(text, 300)
            _console.print(f"  [dim italic]\U0001f4ad {display}[/dim italic]")
        return ""

    # ── Policy decision (LLM chose tool calls or text) ──
    if event == "policy_decision":
        decision = event_dict.get("decision", "?")
        wall = event_dict.get("wall_s", 0)
        ptok = event_dict.get("prompt_tokens", 0)
        ctok = event_dict.get("completion_tokens", 0)
        has_thought = event_dict.get("has_thought", False)

        line = Text()
        line.append("  \u2192 ", style="dim")

        if decision == "tool_call":
            tools = event_dict.get("tools", [])
            line.append("LLM", style="dim")
            line.append(" \u2192 ", style="dim")
            for i, t in enumerate(tools):
                if i > 0:
                    line.append(", ", style="dim")
                line.append(t, style=f"bold {_tool_style(t)}")
        elif decision == "text":
            line.append("LLM", style="dim")
            line.append(" \u2192 ", style="dim")
            line.append("text response", style="magenta")
            consec = event_dict.get("consecutive_text", 0)
            if consec > 1:
                line.append(f" ({consec}x)", style="yellow")

        line.append(f"  {wall:.1f}s", style="dim")
        line.append(f"  {ptok}\u2192{ctok}tok", style="dim")
        if has_thought:
            line.append("  +thought", style="dim blue")

        _console.print(line)
        return ""

    # ── Agent text response (no tool call) ──
    if event in ("agent_text", "text_response"):
        text = event_dict.get("text", event_dict.get("preview", ""))
        line = Text()
        line.append("  -- ", style="bold dim")
        line.append("text response", style="magenta")
        line.append(f"  {_truncate(str(text), 100)}", style="dim")
        _console.print(line)
        return ""

    # ── LLM call timing ──
    if event == "llm_call":
        wall = event_dict.get("wall_s", 0)
        ptok = event_dict.get("prompt_tokens", 0)
        ctok = event_dict.get("completion_tokens", 0)
        _console.print(f"       [dim]llm: {wall:.1f}s  tokens: {ptok}\u2192{ctok}[/dim]")
        return ""

    # ── Run summary ──
    if event == "run_summary":
        _render_run_summary(event_dict)
        return ""

    # ── LLM retry / error ──
    if event == "llm_retry":
        attempt = event_dict.get("attempt", "?")
        error = event_dict.get("error", "")
        _console.print(f"  [yellow]LLM retry {attempt}: {_truncate(str(error), 80)}[/yellow]")
        return ""

    # ── Default: structured context tag + event + values ──
    _render_default(event_dict)
    return ""


def _render_run_summary(event_dict: dict) -> None:
    """Render run_summary as a Rich table (reads top-level keys)."""
    # Build stats table
    table = Table(
        title="RUN SUMMARY",
        title_style="bold",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    stat_keys = [
        ("stop_reason", "Stop Reason"),
        ("episode", "Episode"),
        ("iterations", "Iterations"),
        ("tool_calls", "Tool Calls"),
        ("errors", "Errors"),
        ("prompt_tokens", "Prompt Tokens"),
        ("completion_tokens", "Completion Tokens"),
        ("total_tokens", "Total Tokens"),
        ("wall_time_s", "Wall Time (s)"),
    ]
    for key, label in stat_keys:
        val = event_dict.get(key)
        if val is not None:
            style = "red" if key == "errors" and val else ""
            table.add_row(label, str(val), style=style)

    _console.print()
    _console.print(table)

    # Tool breakdown sub-table
    tools = event_dict.get("tools", {})
    if tools and isinstance(tools, dict):
        t_table = Table(
            title="Tool Breakdown",
            title_style="bold dim",
            box=box.SIMPLE,
            padding=(0, 1),
            expand=False,
        )
        t_table.add_column("Tool", style="bold")
        t_table.add_column("Calls", justify="right")
        t_table.add_column("Errors", justify="right")
        for tname in sorted(tools):
            info = tools[tname]
            calls = info.get("calls", 0) if isinstance(info, dict) else info
            errs = info.get("errors", 0) if isinstance(info, dict) else 0
            style = _tool_style(tname)
            err_style = "red" if errs else "dim"
            t_table.add_row(
                Text(tname, style=style),
                str(calls),
                Text(str(errs), style=err_style),
            )
        _console.print(t_table)

    # Recipe panel
    recipe = event_dict.get("recipe")
    if recipe and isinstance(recipe, dict):
        recipe_text = json.dumps(recipe, indent=2, ensure_ascii=False)
        if len(recipe_text) > 1200:
            recipe_text = recipe_text[:1200] + "\n..."
        _console.print(
            Panel(
                recipe_text,
                title="[bold]Recipe[/bold]",
                border_style="green",
                expand=False,
            )
        )


def _render_default(event_dict: dict) -> None:
    """Render a generic event with Rich-styled context tags."""
    level = event_dict.get("level", "info")
    event_name = str(event_dict.get("event", ""))

    skip_keys = {"event", "level", "timestamp", "_record", "_from_structlog"}
    extra = {k: v for k, v in event_dict.items() if k not in skip_keys}

    # Build context tag: [category ep=N run=X]
    category = extra.pop("category", None)
    episode = extra.pop("episode", None)
    run_id = extra.pop("run_id", None)

    ctx_text = Text()
    if category or episode is not None or run_id:
        ctx_text.append("[")
        if category:
            ctx_text.append(str(category), style="cyan")
        if episode is not None:
            if category:
                ctx_text.append(" ")
            ctx_text.append(f"ep={episode}", style="dim")
        if run_id:
            if category or episode is not None:
                ctx_text.append(" ")
            ctx_text.append(f"run={run_id}", style="dim")
        ctx_text.append("] ")

    # Separate inline and block values
    ordered_inline = [
        "step",
        "tool",
        "dataset",
        "name",
        "rows",
        "kept",
        "rejected",
        "status",
        "reward",
        "duration_s",
        "path",
    ]
    inline_priority = {k: i for i, k in enumerate(ordered_inline)}
    block_keys = {
        "result_preview",
        "error",
        "warning",
        "warnings",
        "args",
        "columns",
        "rejection_reasons",
        "recipe",
        "payload",
    }

    inline_parts: list[str] = []
    block_parts: list[str] = []
    for key in sorted(extra.keys(), key=lambda k: (inline_priority.get(k, 999), k)):
        value = extra[key]
        inline_val = _format_value_inline(value, max_len=120)
        needs_block = (
            key in block_keys or isinstance(value, (dict, list, tuple)) or len(inline_val) > 90
        )
        if needs_block:
            block_val = _format_value_block(value)
            if "\n" in block_val:
                block_parts.append(
                    f"{key}:\n" + "\n".join(f"      {line}" for line in block_val.splitlines())
                )
            else:
                block_parts.append(f"{key}: {block_val}")
        else:
            inline_parts.append(f"{key}={inline_val}")

    level_style = _LEVEL_STYLES.get(level, "")

    line = Text()
    line.append_text(ctx_text)
    line.append(event_name, style=level_style or None)
    if inline_parts:
        line.append("  ")
        line.append(", ".join(inline_parts), style="dim")

    _console.print(line)

    if block_parts:
        for bp in block_parts:
            _console.print(f"    [dim]{bp}[/dim]")


# ── Logging setup ────────────────────────────────────────────────────


def configure_logging(
    *,
    debug: bool = False,
    log_file: Path | None = None,
) -> None:
    """Set up structlog with dual output: Rich stderr + optional JSONL file.

    Parameters
    ----------
    debug:
        If True, set root log level to DEBUG; otherwise INFO.
    log_file:
        If given, every event is also written as one JSON line to this path.
        The file is opened in append mode so multiple runs can share it.
    """
    level = logging.DEBUG if debug else logging.INFO

    # Shared processors — run before any renderer
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # -- stdlib logging integration ------------------------------------------
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    for name in (
        "LiteLLM",
        "litellm",
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "autogen_core",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # -- stdlib formatter (for stdlib loggers routed through structlog) ------
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _agent_console_renderer,
        ],
    )

    # Replace the root handler with our Rich-aware one
    root = logging.getLogger()
    root.handlers.clear()

    console_handler = _RichStreamHandler(sys.stderr)
    console_handler.addFilter(_SuppressEventsFilter())
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # -- optional JSONL file handler -----------------------------------------
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        jsonl_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler = logging.FileHandler(str(log_file), mode="a")
        file_handler.setFormatter(jsonl_formatter)
        root.addHandler(file_handler)

    root.setLevel(level)


def add_file_handler(log_file: Path) -> logging.Handler:
    """Attach a JSONL file handler to the root logger. Returns it for later removal."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.FileHandler(str(log_file), mode="a")
    handler.setFormatter(jsonl_formatter)
    logging.getLogger().addHandler(handler)
    return handler
