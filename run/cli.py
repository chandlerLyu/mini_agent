"""Simple local runner for my_agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.loop import AgentLoop
from config import AppConfig
from environment.local import LocalEnvironment
from eval.tasks import DEMO_TASKS
from models.litellm_client import LiteLLMModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run", description="Run my_agent locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local", help="Run the agent against a local working directory.")
    local.add_argument("--task", required=True, help="Task for the agent.")
    local.add_argument("--model", required=True, help="LiteLLM model name, e.g. openai/gpt-4.1-mini.")
    local.add_argument("--cwd", default="demo_repo", help="Working directory for tool execution.")
    local.add_argument("--output", default="data/last_run.json", help="Trajectory output path.")
    local.add_argument("--step-limit", type=int, default=12, help="Maximum steps before stopping.")

    demo = subparsers.add_parser("demo", help="Show example evaluation tasks.")
    demo.add_argument("--list", action="store_true", help="List bundled demo tasks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        for task in DEMO_TASKS:
            print(f"{task['id']}: {task['task']}")
        return 0

    config = AppConfig()
    config.agent.step_limit = args.step_limit
    config.agent.trajectory_path = Path(args.output)
    config.environment.cwd = Path(args.cwd)
    config.model.model_name = args.model

    loop = AgentLoop(
        model=LiteLLMModel(),
        environment=LocalEnvironment(config.environment),
        config=config,
    )
    result = loop.run(args.task)
    print(f"status={result.status}")
    print(f"steps={result.steps}")
    print(f"trajectory={result.trajectory_path}")
    if result.final_answer:
        print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
