#!/usr/bin/env python3
"""Run all RelBench example tasks and summarize results."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "data" / "relbench" / "run_reports"
KEY_VALUE_PATTERN = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_ ]{0,63}):\s*(?P<value>.*)$")
HELPER_FILES = {
    "relbench_amazon_common.py",
    "relbench_avito_common.py",
    "relbench_dataset_utils.py",
    "relbench_stack_task_utils.py",
    "relbench_trial_task_utils.py",
    "relbench_v2_task_utils.py",
}
LOCAL_RUNNER_FALLBACKS = {
    "relbench_post_votes_local_runner.py",
    "relbench_user_badges_local_runner.py",
    "relbench_user_engagement_local_runner.py",
}
CLASSIFICATION_TASKS = {
    "relbench_amazon_user_churn.py",
    "relbench_amazon_item_churn.py",
    "relbench_avito_user_visits.py",
    "relbench_avito_user_clicks.py",
    "relbench_event_user_repeat.py",
    "relbench_event_user_ignore.py",
    "relbench_f1_driver_dnf.py",
    "relbench_f1_driver_top3.py",
    "relbench_hm_user_churn.py",
    "relbench_user_engagement_local_runner.py",
    "relbench_user_badges_local_runner.py",
    "relbench_trial_study_outcome.py",
    "relbench_arxiv_paper_citation.py",
    "relbench_arxiv_author_category.py",
    "relbench_salt_item_plant.py",
    "relbench_salt_item_shippoint.py",
    "relbench_salt_item_incoterms.py",
    "relbench_salt_sales_office.py",
    "relbench_salt_sales_group.py",
    "relbench_salt_sales_payterms.py",
    "relbench_salt_sales_shipcond.py",
    "relbench_salt_sales_incoterms.py",
    "relbench_ratebeer_beer_churn.py",
    "relbench_ratebeer_user_churn.py",
    "relbench_ratebeer_brewer_dormant.py",
    "relbench_mimic_patient_iculengthofstay.py",
}
REGRESSION_TASKS = {
    "relbench_amazon_user_ltv.py",
    "relbench_amazon_item_ltv.py",
    "relbench_avito_ad_ctr.py",
    "relbench_event_user_attendance.py",
    "relbench_f1_driver_position.py",
    "relbench_hm_item_sales.py",
    "relbench_post_votes_local_runner.py",
    "relbench_trial_study_adverse.py",
    "relbench_trial_site_success.py",
    "relbench_arxiv_author_publication.py",
    "relbench_ratebeer_user_count.py",
    "relbench_ratebeer_beer_ratings_total_score.py",
}
LINK_PREDICTION_TASKS = {
    "relbench_arxiv_paper_paper_cocitation.py",
    "relbench_ratebeer_user_beer_liked.py",
    "relbench_ratebeer_user_place_liked.py",
    "relbench_ratebeer_user_beer_favorite.py",
}
V2_TASKS = {
    task
    for task in CLASSIFICATION_TASKS | REGRESSION_TASKS | LINK_PREDICTION_TASKS
    if task.startswith(
        (
            "relbench_arxiv_",
            "relbench_mimic_",
            "relbench_ratebeer_",
            "relbench_salt_",
        )
    )
}
V1_TASKS = (CLASSIFICATION_TASKS | REGRESSION_TASKS | LINK_PREDICTION_TASKS) - V2_TASKS
TASK_GROUPS = {
    "all": CLASSIFICATION_TASKS | REGRESSION_TASKS | LINK_PREDICTION_TASKS,
    "v1": V1_TASKS,
    "classification": CLASSIFICATION_TASKS,
    "regression": REGRESSION_TASKS,
    "link-prediction": LINK_PREDICTION_TASKS,
    "v2": V2_TASKS,
}


@dataclass
class TaskResult:
    task_name: str
    script: str
    status: str
    return_code: int
    duration_seconds: float
    started_at_utc: str
    finished_at_utc: str
    log_path: str
    parsed_fields: dict[str, Any]
    highlights: dict[str, Any]
    skip_reason: str | None
    error_excerpt: list[str]


def parse_training_frame_workers(value: str) -> int:
    lowered = value.strip().lower()
    if lowered in {"all", "max"}:
        return 0
    try:
        workers = int(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "training frame workers must be a non-negative integer or 'all'"
        ) from exc
    if workers < 0:
        raise argparse.ArgumentTypeError(
            "training frame workers must be a non-negative integer or 'all'"
        )
    return workers


def default_training_frame_workers() -> int:
    configured = os.environ.get("RELBench_TRAINING_FRAME_WORKERS")
    if configured is None:
        return 1
    return parse_training_frame_workers(configured)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for logs and reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for task execution.",
    )
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Only run tasks whose filename contains this substring. Can be repeated.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Run a specific runner filename. Can be repeated.",
    )
    parser.add_argument(
        "--task-type",
        choices=sorted(TASK_GROUPS),
        default="all",
        help="Run all tasks or select classification, regression, or link prediction.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream each task's full stdout/stderr while it runs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed task.",
    )
    parser.add_argument(
        "--training-frame-workers",
        type=parse_training_frame_workers,
        default=default_training_frame_workers(),
        metavar="N",
        help=(
            "Build training cutoff frames with N concurrent workers. "
            "Use 1 for sequential execution or 0/all to start one worker per frame."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tasks without executing them.",
    )
    return parser.parse_args()


def discover_tasks(match_filters: list[str], explicit_tasks: list[str], task_type: str) -> list[Path]:
    primary_tasks = {
        path
        for path in EXAMPLES_DIR.glob("relbench_*.py")
        if path.name not in HELPER_FILES and not path.name.endswith("_local_runner.py")
    }
    fallback_tasks = {EXAMPLES_DIR / name for name in LOCAL_RUNNER_FALLBACKS}
    tasks = sorted(primary_tasks | {path for path in fallback_tasks if path.exists()})
    allowed_names = TASK_GROUPS[task_type]
    tasks = [task for task in tasks if task.name in allowed_names]
    if explicit_tasks:
        wanted = set(explicit_tasks)
        tasks = [task for task in tasks if task.name in wanted]
    if match_filters:
        tasks = [
            task
            for task in tasks
            if any(fragment.lower() in task.name.lower() for fragment in match_filters)
        ]
    return tasks


def maybe_literal_eval(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered in {"skipped", "none", "true", "false"}:
        return {"skipped": "skipped", "none": None, "true": True, "false": False}[lowered]

    if value[0] in "[{('\"" or value[0].isdigit() or (value[0] == "-" and value[1:].isdigit()):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def parse_output(stdout: str, stderr: str) -> tuple[dict[str, Any], dict[str, Any], str | None, list[str]]:
    fields: dict[str, Any] = {}
    skip_reason: str | None = None

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "skipping model fit" in stripped.lower() or "insufficient features" in stripped.lower():
            skip_reason = stripped
        match = KEY_VALUE_PATTERN.match(stripped)
        if not match:
            continue
        key = match.group("key").strip().replace(" ", "_")
        value = maybe_literal_eval(match.group("value"))
        fields[key] = value

    highlights: dict[str, Any] = {}
    for key in (
        "catboost_auc",
        "catboost_in_time_auc",
        "catboost_holdout_auc",
        "validation_metrics",
        "test_metrics",
        "validation_nmae",
        "test_nmae",
        "in_time_holdout_mae_2020",
        "out_of_time_mae_2021",
        "feature_count",
        "num_features",
        "target",
        "rows",
        "train_rows",
        "validation_rows",
        "test_rows",
        "val_rows",
        "future_rows",
    ):
        if key in fields:
            highlights[key] = fields[key]

    combined = [line.rstrip() for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    error_excerpt = combined[-20:]
    return fields, highlights, skip_reason, error_excerpt


def format_summary(result: TaskResult) -> str:
    parts = [
        f"status={result.status}",
        f"duration={result.duration_seconds:.1f}s",
    ]
    if result.skip_reason:
        parts.append(f"skip_reason={result.skip_reason}")

    metrics = []
    for key, value in result.highlights.items():
        metrics.append(f"{key}={value}")
    if metrics:
        parts.append("metrics: " + "; ".join(metrics))

    return " | ".join(parts)


def run_task(
    task_path: Path,
    python_executable: str,
    output_dir: Path,
    stream_output: bool,
    training_frame_workers: int = 1,
) -> TaskResult:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{task_path.stem}.log"

    started = datetime.now(timezone.utc)
    started_perf = time.perf_counter()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if training_frame_workers < 0:
        raise ValueError("training_frame_workers must be non-negative")
    env["RELBench_TRAINING_FRAME_WORKERS"] = str(training_frame_workers)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )

    process = subprocess.Popen(
        [python_executable, str(task_path)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    parsed_lines: list[str] = []
    error_tail: deque[str] = deque(maxlen=20)
    with log_path.open("w") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if KEY_VALUE_PATTERN.match(line.strip()) or "skipping model fit" in line.lower() or "insufficient features" in line.lower():
                parsed_lines.append(line)
            error_tail.append(line.rstrip())
            if stream_output:
                print(line, end="", flush=True)
    process.wait()
    duration = time.perf_counter() - started_perf
    finished = datetime.now(timezone.utc)
    fields, highlights, skip_reason, _ = parse_output("".join(parsed_lines), "")
    status = "passed" if process.returncode == 0 else "failed"

    return TaskResult(
        task_name=task_path.stem.removesuffix("_local_runner"),
        script=str(task_path.relative_to(REPO_ROOT)),
        status=status,
        return_code=process.returncode,
        duration_seconds=duration,
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        log_path=str(log_path.relative_to(REPO_ROOT)),
        parsed_fields=fields,
        highlights=highlights,
        skip_reason=skip_reason,
        error_excerpt=list(error_tail),
    )


def build_markdown_report(results: list[TaskResult], started_at_utc: str, finished_at_utc: str) -> str:
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)

    lines = [
        "# RelBench Task Run Report",
        "",
        f"- Started at: `{started_at_utc}`",
        f"- Finished at: `{finished_at_utc}`",
        f"- Tasks: `{len(results)}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{failed}`",
        "",
        "| Task | Status | Duration (s) | Highlights | Log |",
        "| --- | --- | ---: | --- | --- |",
    ]

    for result in results:
        highlights = ", ".join(f"{key}={value}" for key, value in result.highlights.items()) or "n/a"
        lines.append(
            f"| `{result.task_name}` | `{result.status}` | {result.duration_seconds:.1f} | {highlights} | `{result.log_path}` |"
        )

    failures = [result for result in results if result.status == "failed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            lines.append(f"### `{result.task_name}`")
            lines.append("")
            lines.append(f"- Script: `{result.script}`")
            lines.append(f"- Return code: `{result.return_code}`")
            lines.append(f"- Log: `{result.log_path}`")
            lines.append("- Error excerpt:")
            lines.append("```text")
            lines.extend(result.error_excerpt or ["<no output captured>"])
            lines.append("```")
            lines.append("")

    return "\n".join(lines) + "\n"


def print_final_console_report(results: list[TaskResult], markdown_path: Path, json_path: Path) -> None:
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)

    print("\nFinal RelBench results", flush=True)
    print("=" * 80, flush=True)
    print(f"tasks={len(results)} passed={passed} failed={failed}", flush=True)
    for result in results:
        print(f"- {result.task_name}: {format_summary(result)}", flush=True)
    print(f"\nmarkdown_report={markdown_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"json_report={json_path.relative_to(REPO_ROOT)}", flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    tasks = discover_tasks(args.match, args.task, args.task_type)
    if not tasks:
        print("No relbench local runner tasks matched the selection.", file=sys.stderr, flush=True)
        return 1

    if args.dry_run:
        print(f"Discovered relbench {args.task_type} tasks:", flush=True)
        for task in tasks:
            print(f"- {task.relative_to(REPO_ROOT)}", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc)
    results: list[TaskResult] = []

    print(
        f"Running {len(tasks)} relbench {args.task_type} task(s) from {REPO_ROOT} using {args.python}",
        flush=True,
    )
    print(
        "Training frame workers: "
        f"{'all frames' if args.training_frame_workers == 0 else args.training_frame_workers}",
        flush=True,
    )
    print(f"Logs and reports will be written under {args.output_dir.relative_to(REPO_ROOT)}", flush=True)

    for index, task in enumerate(tasks, start=1):
        print(f"\n[{index}/{len(tasks)}] starting {task.stem}", flush=True)
        result = run_task(
            task,
            args.python,
            args.output_dir,
            args.stream_output,
            args.training_frame_workers,
        )
        results.append(result)
        print(f"[{index}/{len(tasks)}] finished {task.stem} | {format_summary(result)}", flush=True)
        if args.stop_on_error and result.status == "failed":
            break

    run_finished = datetime.now(timezone.utc)
    json_path = args.output_dir / "relbench_results.json"
    markdown_path = args.output_dir / "relbench_results.md"

    payload = {
        "started_at_utc": run_started.isoformat(),
        "finished_at_utc": run_finished.isoformat(),
        "task_count": len(results),
        "task_type": args.task_type,
        "training_frame_workers": args.training_frame_workers,
        "passed_count": sum(result.status == "passed" for result in results),
        "failed_count": sum(result.status == "failed" for result in results),
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(build_markdown_report(results, payload["started_at_utc"], payload["finished_at_utc"]))

    print_final_console_report(results, markdown_path, json_path)
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
