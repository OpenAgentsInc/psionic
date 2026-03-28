#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hermes compatibility probes against an OpenAI-compatible qwen35 backend."
    )
    parser.add_argument("--hermes-root", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--psionic-root", required=True)
    parser.add_argument("--backend-label", default="psionic")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--psionic-revision", default="")
    parser.add_argument("--max-iterations", type=int, default=5)
    return parser.parse_args()


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def import_hermes(hermes_root: Path):
    sys.path.insert(0, str(hermes_root))
    os.chdir(hermes_root)
    from run_agent import AIAgent
    from toolsets import create_custom_toolset
    from tools.registry import registry

    return AIAgent, create_custom_toolset, registry


def first_assistant_tool_turn(messages: list[dict]) -> dict | None:
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return message
    return None


def final_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def lower_text(value: str) -> str:
    return value.lower().strip()


def run_case(
    *,
    case_id: str,
    user_prompt: str,
    system_message: str,
    enabled_toolsets: list[str],
    tool_policy,
    max_iterations: int,
    model: str,
    base_url: str,
    AIAgent,
):
    agent = AIAgent(
        base_url=base_url,
        api_key="dummy",
        provider="custom",
        api_mode="chat_completions",
        model=model,
        enabled_toolsets=enabled_toolsets,
        max_iterations=max_iterations,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    obs = {"stream_calls": 0, "deltas": []}
    orig_stream = agent._interruptible_streaming_api_call
    orig_build = agent._build_api_kwargs

    def wrapped_stream(self, api_kwargs, on_first_delta=None):
        obs["stream_calls"] += 1
        return orig_stream(api_kwargs, on_first_delta=on_first_delta)

    def wrapped_build(self, api_messages):
        kwargs = orig_build(api_messages)
        tool_choice, parallel_tool_calls = tool_policy(api_messages)
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        kwargs["temperature"] = 0
        kwargs["seed"] = 0
        obs.setdefault("request_summaries", []).append(
            {
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "tool_names": [
                    tool.get("function", {}).get("name")
                    for tool in kwargs.get("tools", [])
                    if isinstance(tool, dict)
                ],
            }
        )
        return kwargs

    agent._interruptible_streaming_api_call = MethodType(wrapped_stream, agent)
    agent._build_api_kwargs = MethodType(wrapped_build, agent)

    def stream_callback(text: str):
        obs["deltas"].append(text)

    result = agent.run_conversation(
        user_prompt,
        system_message=system_message,
        stream_callback=stream_callback,
    )
    return {
        "case_id": case_id,
        "stream_calls": obs["stream_calls"],
        "stream_deltas": obs["deltas"],
        "request_summaries": obs.get("request_summaries", []),
        "result": result,
    }


def report_case_result(case: dict, passed: bool, summary: str, details: dict) -> dict:
    result = case["result"]
    return {
        "case_id": case["case_id"],
        "pass": passed,
        "summary": summary,
        "stream_calls": case["stream_calls"],
        "request_summaries": case.get("request_summaries", []),
        "api_calls": result.get("api_calls"),
        "completed": result.get("completed"),
        "failed": result.get("failed", False),
        "final_response": result.get("final_response"),
        "messages": result.get("messages", []),
        "details": details,
    }


def main() -> int:
    args = parse_args()
    hermes_root = Path(args.hermes_root).resolve()
    psionic_root = Path(args.psionic_root).resolve()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    AIAgent, create_custom_toolset, registry = import_hermes(hermes_root)

    psionic_revision = args.psionic_revision or run_git(
        ["git", "rev-parse", "HEAD"], psionic_root
    )
    hermes_revision = run_git(["git", "rev-parse", "HEAD"], hermes_root)

    def required_only(_api_messages):
        return "required", True

    def auto_only(_api_messages):
        return "auto", True

    def required_then_auto(api_messages):
        saw_tool = any(message.get("role") == "tool" for message in api_messages)
        return ("auto" if saw_tool else "required"), True

    reports: list[dict] = []

    registry.register(
        name="get_weather_required",
        toolset="psionic_hermes_qwen35_required_tool_turn",
        schema={
            "name": "get_weather_required",
            "description": "Get the current Paris weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Paris"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {"city": args.get("city"), "condition": "sunny"}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_qwen35_required_tool_turn",
        "Required tool turn probe",
        tools=["get_weather_required"],
    )
    case = run_case(
        case_id="required_tool_turn",
        user_prompt="Use the weather tool for Paris.",
        system_message="You are Hermes. Use tools when needed.",
        enabled_toolsets=["psionic_hermes_qwen35_required_tool_turn"],
        tool_policy=required_then_auto,
        max_iterations=2,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    tool_turn = first_assistant_tool_turn(case["result"].get("messages", []))
    tool_names = [
        tool_call.get("function", {}).get("name")
        for tool_call in (tool_turn or {}).get("tool_calls", [])
    ]
    reports.append(
        report_case_result(
            case,
            passed=tool_names == ["get_weather_required"],
            summary="required tool turn emitted one weather tool call",
            details={"assistant_tool_names": tool_names},
        )
    )

    registry.register(
        name="get_weather_auto",
        toolset="psionic_hermes_qwen35_auto_text_turn",
        schema={
            "name": "get_weather_auto",
            "description": "Get weather information if explicitly requested.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda args, **kwargs: json.dumps({"city": args.get("city"), "unused": True}),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_qwen35_auto_text_turn",
        "Auto text turn probe",
        tools=["get_weather_auto"],
    )
    case = run_case(
        case_id="auto_plain_text_turn",
        user_prompt="Say hello in one short sentence. Do not use tools.",
        system_message="You are Hermes. If the user asks for a greeting, answer directly and do not call tools.",
        enabled_toolsets=["psionic_hermes_qwen35_auto_text_turn"],
        tool_policy=auto_only,
        max_iterations=args.max_iterations,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    messages = case["result"].get("messages", [])
    tool_turn = first_assistant_tool_turn(messages)
    final_text = final_assistant_text(messages)
    reports.append(
        report_case_result(
            case,
            passed=tool_turn is None and bool(final_text),
            summary="auto mode answered with plain text and no tool call",
            details={"final_text": final_text},
        )
    )

    registry.register(
        name="get_weather_loop",
        toolset="psionic_hermes_qwen35_tool_loop",
        schema={
            "name": "get_weather_loop",
            "description": "Get the current weather for Paris.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {"city": args.get("city"), "condition": "sunny", "temperature_c": 22}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_qwen35_tool_loop",
        "Tool loop probe",
        tools=["get_weather_loop"],
    )
    case = run_case(
        case_id="multi_turn_tool_loop",
        user_prompt="Use the weather tool for Paris and then summarize the result.",
        system_message="You are Hermes. Use tools when needed and summarize tool results plainly.",
        enabled_toolsets=["psionic_hermes_qwen35_tool_loop"],
        tool_policy=required_then_auto,
        max_iterations=args.max_iterations,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    messages = case["result"].get("messages", [])
    final_text = final_assistant_text(messages)
    saw_tool_result = any(message.get("role") == "tool" for message in messages)
    reports.append(
        report_case_result(
            case,
            passed=saw_tool_result and "paris" in lower_text(final_text),
            summary="multi-turn tool loop replay reached a final answer",
            details={"final_text": final_text, "saw_tool_result": saw_tool_result},
        )
    )

    registry.register(
        name="get_paris_weather",
        toolset="psionic_hermes_qwen35_parallel_turn",
        schema={
            "name": "get_paris_weather",
            "description": "Get the weather for Paris.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        handler=lambda args, **kwargs: json.dumps({"city": "Paris", "condition": "sunny"}),
        check_fn=lambda: True,
    )
    registry.register(
        name="get_tokyo_weather",
        toolset="psionic_hermes_qwen35_parallel_turn",
        schema={
            "name": "get_tokyo_weather",
            "description": "Get the weather for Tokyo.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        handler=lambda args, **kwargs: json.dumps({"city": "Tokyo", "condition": "rainy"}),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_qwen35_parallel_turn",
        "Parallel turn probe",
        tools=["get_paris_weather", "get_tokyo_weather"],
    )
    case = run_case(
        case_id="parallel_tool_turn",
        user_prompt="Use both weather tools now, then summarize both cities in one answer.",
        system_message=(
            "You are Hermes. When the user asks for both weather tools, emit one tool-call turn "
            "that includes both tool calls before answering."
        ),
        enabled_toolsets=["psionic_hermes_qwen35_parallel_turn"],
        tool_policy=required_then_auto,
        max_iterations=2,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    tool_turn = first_assistant_tool_turn(case["result"].get("messages", []))
    tool_names = [
        tool_call.get("function", {}).get("name")
        for tool_call in (tool_turn or {}).get("tool_calls", [])
    ]
    reports.append(
        report_case_result(
            case,
            passed=tool_names == ["get_paris_weather", "get_tokyo_weather"],
            summary="parallel tool turn emitted both tool calls in one assistant response",
            details={"assistant_tool_names": tool_names},
        )
    )

    registry.register(
        name="lookup_atlantis_weather",
        toolset="psionic_hermes_qwen35_invalid_arguments",
        schema={
            "name": "lookup_atlantis_weather",
            "description": "Attempt to look up the weather for Atlantis.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Atlantis"]}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {
                "error": f"unsupported city {args.get('city')}",
                "supported_cities": ["Paris"],
                "refusal": "Atlantis is unsupported. Do not claim a weather result.",
            }
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_qwen35_invalid_arguments",
        "Invalid argument probe",
        tools=["lookup_atlantis_weather"],
    )
    case = run_case(
        case_id="invalid_argument_truthful_refusal",
        user_prompt="Use the Atlantis weather tool and then explain truthfully what happened.",
        system_message=(
            "You are Hermes. If the Atlantis weather tool returns an error, your next assistant "
            "message must be exactly: Atlantis is unsupported. I do not have a weather result."
        ),
        enabled_toolsets=["psionic_hermes_qwen35_invalid_arguments"],
        tool_policy=required_then_auto,
        max_iterations=3,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    messages = case["result"].get("messages", [])
    final_text = final_assistant_text(messages)
    lowered = lower_text(final_text)
    truthful_refusal = "atlantis" in lowered and any(
        token in lowered
        for token in [
            "unsupported",
            "error",
            "not supported",
            "cannot",
            "can't",
            "do not have access",
            "do not have a weather result",
        ]
    )
    reports.append(
        report_case_result(
            case,
            passed=truthful_refusal,
            summary="invalid-argument case surfaced a truthful failure",
            details={"final_text": final_text},
        )
    )

    registry.register(
        name="get_weather_stream",
        toolset="psionic_hermes_qwen35_streamed_turn",
        schema={
            "name": "get_weather_stream",
            "description": "Get the current weather for Paris.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {"city": args.get("city"), "condition": "sunny"}
        ),
        check_fn=lambda: True,
    )
    create_custom_toolset(
        "psionic_hermes_qwen35_streamed_turn",
        "Streamed tool probe",
        tools=["get_weather_stream"],
    )
    case = run_case(
        case_id="streamed_tool_turn",
        user_prompt="Use the weather tool for Paris.",
        system_message="You are Hermes. Use tools when needed.",
        enabled_toolsets=["psionic_hermes_qwen35_streamed_turn"],
        tool_policy=required_then_auto,
        max_iterations=2,
        model=args.model,
        base_url=args.base_url,
        AIAgent=AIAgent,
    )
    tool_turn = first_assistant_tool_turn(case["result"].get("messages", []))
    reports.append(
        report_case_result(
            case,
            passed=case["stream_calls"] >= 1 and tool_turn is not None,
            summary="streaming chat loop preserved a tool-call turn",
            details={
                "assistant_tool_names": [
                    tool_call.get("function", {}).get("name")
                    for tool_call in (tool_turn or {}).get("tool_calls", [])
                ]
            },
        )
    )

    overall_pass = all(case["pass"] for case in reports)
    output = {
        "report_kind": "psionic_hermes_qwen35_compatibility",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_label": args.backend_label,
        "base_url": args.base_url,
        "model": args.model,
        "model_path": args.model_path,
        "psionic_root": str(psionic_root),
        "psionic_revision": psionic_revision,
        "hermes_root": str(hermes_root),
        "hermes_revision": hermes_revision,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "host": platform.node(),
        "cases": reports,
        "overall_pass": overall_pass,
        "passing_case_count": sum(1 for case in reports if case["pass"]),
        "total_case_count": len(reports),
    }
    report_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
