#!/usr/bin/env python3
"""Small local API for streaming Hello World run logs into MkDocs UI.

Run:
  python docs/api/modal_stream_server.py

Then open the interactive docs page and click Run.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


@dataclass
class JobState:
    id: str
    cmd: list[str]
    cwd: str
    process: subprocess.Popen[str] | None = None
    log_queue: Queue[str] = field(default_factory=Queue)
    logs: list[str] = field(default_factory=list)
    status: str = "queued"
    return_code: int | None = None


class StartJobRequest(BaseModel):
    example: str = "hello_world"


def _allowed_origins() -> list[str]:
    raw = os.getenv("GRAPHREDUCE_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8001",
            "http://localhost:8001",
        ]
    if raw == "*":
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


app = FastAPI(title="GraphReduce Local Stream API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, JobState] = {}

SCRIPT_MAP: dict[str, str] = {
    "hello_world": "examples/hello_world_local_runner.py",
    "preserve_child_grain": "examples/preserve_child_grain_local_runner.py",
    "all_tables_ml_targets": "examples/all_tables_ml_targets_local_runner.py",
    "predictive_ai_xgboost": "examples/predictive_ai_xgboost_local_runner.py",
    "relbench_user_badges": "examples/relbench_user_badges_local_runner.py",
    "relbench_post_votes": "examples/relbench_post_votes_local_runner.py",
    "relbench_user_engagement": "examples/relbench_user_engagement_local_runner.py",
    "relbench_hm_user_churn": "examples/relbench_hm_user_churn_local_runner.py",
    "relbench_avito_user_clicks": "examples/relbench_avito_user_clicks_local_runner.py",
    "relbench_avito_user_visits": "examples/relbench_avito_user_visits_local_runner.py",
    "relbench_amazon_user_churn": "examples/relbench_amazon_user_churn_local_runner.py",
    "relbench_amazon_item_churn": "examples/relbench_amazon_item_churn_local_runner.py",
    "multi_backend_pandas": "examples/multi_backend_pandas_local_runner.py",
    "multi_backend_sqlite": "examples/multi_backend_sqlite_local_runner.py",
    "multi_backend_duckdb": "examples/multi_backend_duckdb_local_runner.py",
    "multi_backend_pyspark": "examples/multi_backend_pyspark_local_runner.py",
    "custom_pyspark_all_nodes": "examples/custom_pyspark_all_nodes_local_runner.py",
    "custom_pandas_all_nodes": "examples/custom_pandas_all_nodes_local_runner.py",
    "custom_duckdb_all_nodes": "examples/custom_duckdb_all_nodes_local_runner.py",
}


def _stream_reader(job: JobState) -> None:
    assert job.process is not None
    # Read char-by-char so output that uses carriage returns or partial lines
    # still appears in the browser stream.
    buf = ""
    while True:
        ch = job.process.stdout.read(1)  # type: ignore[union-attr]
        if ch == "":
            break
        if ch in ("\n", "\r"):
            if buf:
                job.logs.append(buf)
                job.log_queue.put(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        job.logs.append(buf)
        job.log_queue.put(buf)

    rc = job.process.wait()
    job.return_code = rc
    job.status = "succeeded" if rc == 0 else "failed"
    job.log_queue.put("__JOB_DONE__")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/examples")
def list_examples() -> dict[str, Any]:
    return {"examples": sorted(SCRIPT_MAP.keys())}


@app.post("/jobs")
def start_job(payload: StartJobRequest) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = SCRIPT_MAP.get(payload.example)
    if not script:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unsupported example '{payload.example}'",
                "supported_examples": sorted(SCRIPT_MAP.keys()),
            },
        )
    cmd = [sys.executable, script]

    job_id = str(uuid.uuid4())
    job = JobState(id=job_id, cmd=cmd, cwd=str(repo_root))
    jobs[job_id] = job

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("GRAPHREDUCE_INTERACTIVE", "1")
    try:
        job.process = subprocess.Popen(
            cmd,
            cwd=job.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        job.status = "failed"
        job.return_code = 127
        msg = f"Failed to start command: {exc}"
        job.logs.append(msg)
        job.log_queue.put(msg)
        job.log_queue.put("__JOB_DONE__")
        return {"job_id": job_id}

    job.status = "running"
    thread = threading.Thread(target=_stream_reader, args=(job,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {
        "job_id": job.id,
        "status": job.status,
        "return_code": job.return_code,
        "command": " ".join(job.cmd),
        "cwd": job.cwd,
        "log_lines": len(job.logs),
    }


@app.get("/jobs/{job_id}/stream")
async def stream_logs(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")

    async def event_gen():
        # Replay existing logs first for reconnect support.
        for line in job.logs:
            yield f"data: {line}\n\n"

        while True:
            try:
                line = job.log_queue.get(timeout=0.2)
            except Empty:
                await asyncio.sleep(0.1)
                continue
            if line == "__JOB_DONE__":
                yield "event: done\ndata: complete\n\n"
                break
            yield f"data: {line}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    host = os.getenv("GRAPHREDUCE_RUNNER_HOST", "127.0.0.1")
    port = int(os.getenv("GRAPHREDUCE_RUNNER_PORT", "8001"))
    uvicorn.run(app, host=host, port=port, reload=False)
