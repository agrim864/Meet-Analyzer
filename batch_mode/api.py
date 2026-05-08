from __future__ import annotations

import json
import mimetypes
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .cluster_runner import ClusterBatchRunner
from .config import BatchModeSettings
from .service import BatchMeetingAnalyzer


class AnalyzeRequest(BaseModel):
    source: str = Field(..., description="Local mp4 path or public http(s) mp4 URL such as a Bunny CDN URL.")
    transcript_segments: Optional[List[Dict[str, Any]]] = None
    title: str = ""
    persist_artifacts: bool = True
    writer: str = Field("", description="Report writer preset: mistral, gemini, qwen, or mock.")


class AnalyzeBatchRequest(BaseModel):
    requests: List[AnalyzeRequest] = Field(..., min_length=1, description="Sequential batch of analysis requests.")
    writer: str = Field("", description="Default report writer preset for requests that do not set writer.")
    sequential: bool = Field(True, description="For cluster runs, submit jobs with Slurm dependencies so they run one after another.")


@lru_cache(maxsize=1)
def get_service() -> BatchMeetingAnalyzer:
    return BatchMeetingAnalyzer(BatchModeSettings.from_env())


@lru_cache(maxsize=1)
def get_cluster_runner() -> ClusterBatchRunner:
    return ClusterBatchRunner(BatchModeSettings.from_env())


app = FastAPI(title="Batch Meeting Analyzer", version="0.1.0")


def _content_disposition(filename: str, *, download: bool) -> str:
    disposition = "attachment" if download else "inline"
    safe_filename = filename.replace('"', "").replace("\\", "_").replace("/", "_") or "artifact"
    return f'{disposition}; filename="{safe_filename}"'


@app.get("/health")
def health() -> Dict[str, Any]:
    settings = BatchModeSettings.from_env()
    return {
        "ok": True,
        "embedding_model": settings.embedding_model_name,
        "llm_backend": settings.llm_backend,
        "llm_model": settings.llm_model,
        "qwen_model": settings.qwen_model_name,
        "writer_presets": ["mistral", "gemini", "qwen", "mock"],
        "cluster_host": settings.cluster_host,
        "cluster_project_dir": settings.cluster_project_dir,
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    try:
        result = get_service().analyze(
            source=request.source,
            transcript_segments=request.transcript_segments,
            title=request.title,
            persist_artifacts=request.persist_artifacts,
            writer=request.writer,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/api/analyze/batch")
def analyze_batch(request: AnalyzeBatchRequest) -> Dict[str, Any]:
    try:
        results = get_service().analyze_many(
            requests=[
                {
                    "source": item.source,
                    "transcript_segments": item.transcript_segments,
                    "title": item.title,
                    "persist_artifacts": item.persist_artifacts,
                    "writer": item.writer or request.writer,
                }
                for item in request.requests
            ]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "count": len(results),
        "results": [item.to_dict() for item in results],
    }


@app.post("/api/analyze/cluster")
def analyze_cluster(request: AnalyzeRequest) -> Dict[str, Any]:
    try:
        record = get_cluster_runner().submit(
            source=request.source,
            transcript_segments=request.transcript_segments,
            title=request.title,
            persist_artifacts=request.persist_artifacts,
            writer=request.writer,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "request_id": record.request_id,
        "job_id": record.job_id,
        "status": "submitted",
        "remote_result_json": record.remote_result_json,
        "remote_source": record.remote_source,
        "writer": request.writer or "mistral",
    }


@app.post("/api/analyze/cluster/batch")
def analyze_cluster_batch(request: AnalyzeBatchRequest) -> Dict[str, Any]:
    try:
        records = get_cluster_runner().submit_many(
            requests=[
                {
                    "source": item.source,
                    "transcript_segments": item.transcript_segments,
                    "title": item.title,
                    "persist_artifacts": item.persist_artifacts,
                    "writer": item.writer or request.writer,
                }
                for item in request.requests
            ],
            writer=request.writer,
            sequential=request.sequential,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "count": len(records),
        "sequential": request.sequential,
        "writer": request.writer or "mistral",
        "jobs": [
            {
                "request_id": record.request_id,
                "job_id": record.job_id,
                "status": "submitted",
                "remote_result_json": record.remote_result_json,
                "remote_source": record.remote_source,
                "title": record.title,
            }
            for record in records
        ],
    }


@app.get("/api/jobs")
def cluster_jobs(limit: int = 50, include_status: bool = False) -> Dict[str, Any]:
    try:
        return get_cluster_runner().jobs(limit=limit, include_status=include_status)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/jobs")
def clear_cluster_jobs(confirm: bool = False, delete_remote_artifacts: bool = False) -> Dict[str, Any]:
    try:
        return get_cluster_runner().clear_jobs(
            confirm=confirm,
            delete_remote_artifacts=delete_remote_artifacts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{request_id}/artifacts")
def cluster_job_artifacts(request_id: str) -> Dict[str, Any]:
    try:
        return get_cluster_runner().artifacts(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{request_id}/artifacts/{artifact_path:path}")
def cluster_job_artifact(request_id: str, artifact_path: str, download: bool = False) -> Response:
    try:
        artifact = get_cluster_runner().artifact(request_id, artifact_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = str(artifact.get("filename") or "artifact")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return Response(
        content=artifact["content"],
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(filename, download=download)},
    )


@app.get("/api/jobs/{request_id}")
def cluster_job_status(request_id: str) -> Dict[str, Any]:
    try:
        return get_cluster_runner().status(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/cluster/queue")
def cluster_queue() -> Dict[str, Any]:
    try:
        return get_cluster_runner().queue()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/cluster/accounting")
def cluster_accounting(since: str = "today", limit: int = 20) -> Dict[str, Any]:
    try:
        return get_cluster_runner().accounting(since=since, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/cluster/cancel/{job_id}")
def cluster_cancel_job(job_id: str) -> Dict[str, Any]:
    try:
        return get_cluster_runner().cancel_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{request_id}/result")
def cluster_job_result(request_id: str, include_full: bool = False, download: bool = False) -> Any:
    try:
        result = get_cluster_runner().result(request_id, include_full=include_full or download)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if download:
        return Response(
            content=json.dumps(result, ensure_ascii=True, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": _content_disposition(f"{request_id}_analysis.json", download=True)},
        )
    return result


@app.get("/api/jobs/{request_id}/report")
def cluster_job_report(request_id: str, download: bool = False) -> Any:
    try:
        report = get_cluster_runner().report(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if download:
        return Response(
            content=report.get("markdown", ""),
            media_type="text/markdown",
            headers={"Content-Disposition": _content_disposition("report.md", download=True)},
        )
    return report
