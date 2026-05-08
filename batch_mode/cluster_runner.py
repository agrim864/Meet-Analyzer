from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import paramiko

from .config import BatchModeSettings


@dataclass
class ClusterJobRecord:
    request_id: str
    source: str
    title: str
    job_id: str
    submitted_at: float
    remote_request_json: str
    remote_result_json: str
    remote_source: str
    remote_log_glob: str
    persist_artifacts: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClusterBatchRunner:
    def __init__(self, settings: BatchModeSettings | None = None) -> None:
        self.settings = settings or BatchModeSettings.from_env()
        self.jobs_dir = Path(self.settings.cluster_jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        *,
        source: str,
        title: str = "",
        transcript_segments: Optional[list[dict[str, Any]]] = None,
        persist_artifacts: bool = True,
        writer: str = "",
        dependency_after_job_id: str = "",
    ) -> ClusterJobRecord:
        request_id = uuid.uuid4().hex
        submitted_at = __import__("time").time()
        remote_job_dir = f"{self.settings.cluster_project_dir}/batch_mode/cluster_jobs/{request_id}"
        remote_request_json = f"{remote_job_dir}/request.json"
        remote_result_json = f"{remote_job_dir}/result.json"
        remote_log_glob = f"{self.settings.cluster_log_dir}/batch_api_*"

        payload = {
            "source": source,
            "title": title,
            "transcript_segments": transcript_segments or [],
            "persist_artifacts": persist_artifacts,
        }
        writer_env = self._writer_env(writer)
        if self.settings.llm_base_url and not writer_env:
            llm_backend = "vllm"
        else:
            llm_backend = writer_env.get("BATCH_LLM_BACKEND") or (self.settings.llm_backend if self.settings.llm_backend not in {"", "mock"} else "hf-local")

        client = self._client()
        try:
            self._exec_checked(client, f"mkdir -p {self._shell(remote_job_dir)}")
            self._exec_checked(client, f"mkdir -p {self._shell(self.settings.cluster_log_dir)}")
            remote_source = self._prepare_remote_source(client, request_id=request_id, source=source)
            payload["source"] = remote_source
            if transcript_segments:
                self._write_remote_json(client, remote_request_json, payload)
            else:
                self._prepare_remote_request(
                    client,
                    source=remote_source,
                    request_json=remote_request_json,
                    transcript_json=f"{remote_job_dir}/transcript_segments.json",
                    title=title,
                    persist_artifacts=persist_artifacts,
                )
            exports = [
                    f"cd {self._shell(self.settings.cluster_project_dir)}",
                    f"export PYTHONPATH={self._shell(self.settings.cluster_project_dir)}:${{PYTHONPATH:-}}",
                    f"export BATCH_SOURCE={self._shell(remote_source)}",
                    f"export BATCH_REQUEST_JSON={self._shell(remote_request_json)}",
                    f"export BATCH_RESULT_JSON={self._shell(remote_result_json)}",
                    f"export BATCH_TITLE={self._shell(title)}",
                    f"export HF_API_KEY={self._shell(self.settings.hf_api_key)}",
                    f"export DEEPGRAM_API_KEY={self._shell(self.settings.deepgram_api_key)}",
                    f"export SIGLIP_MODEL_NAME={self._shell(self.settings.siglip_model_name)}",
                    f"export BATCH_LLM_MODEL={self._shell(writer_env.get('BATCH_LLM_MODEL') or self.settings.llm_model or self.settings.qwen_model_name)}",
                    f"export BATCH_LLM_BASE_URL={self._shell(self.settings.llm_base_url)}",
                    f"export BATCH_LLM_API_KEY={self._shell(self._writer_api_key(writer_env))}",
                    f"export BATCH_QWEN_MODEL_NAME={self._shell(self.settings.qwen_model_name)}",
                    f"export BATCH_EMBEDDING_MODEL={self._shell(self.settings.embedding_model_name)}",
                    f"export BATCH_LLM_BACKEND={self._shell(llm_backend)}",
                    f"export BATCH_LLM_SECTION_WORKERS={self._shell(str(__import__('os').getenv('BATCH_LLM_SECTION_WORKERS', '4')))}",
                    "export HF_HUB_OFFLINE=1",
                    "export TRANSFORMERS_OFFLINE=1",
                    "export BATCH_DISABLE_SIGLIP=1",
                    "export PYTHONUNBUFFERED=1",
                    f"export BATCH_OUTPUT_DIR={self._shell(self.settings.cluster_project_dir + '/batch_mode/outputs')}",
                    f"export BATCH_TEMP_DIR={self._shell(self.settings.cluster_project_dir + '/batch_mode/tmp')}",
                    f"export BATCH_CLUSTER_JOBS_DIR={self._shell(self.settings.cluster_project_dir + '/batch_mode/cluster_jobs')}",
                ]
            for key, value in writer_env.items():
                if key in {"BATCH_LLM_BACKEND", "BATCH_LLM_MODEL"}:
                    continue
                exports.append(f"export {key}={self._shell(value)}")
            exports.append(
                f"\"${{CONDA_BIN:-${{HOME}}/anaconda3/bin/conda}}\" run --no-capture-output -n {self._shell(self.settings.cluster_conda_env)} python {self._shell(self.settings.cluster_project_dir + '/batch_mode/myproject/test.py')} --request-json {self._shell(remote_request_json)} --output-json {self._shell(remote_result_json)}"
            )
            wrap_command = " && ".join(exports)
            dependency_part = f"--dependency={self._shell('afterany:' + dependency_after_job_id)} " if dependency_after_job_id else ""
            command = (
                f"sbatch --account {self._shell(self.settings.cluster_account)} "
                f"--partition {self._shell(self.settings.cluster_partition)} "
                f"--gres {self._shell(self.settings.cluster_gres)} "
                f"--time {self._shell(self.settings.cluster_time_limit)} "
                f"--mem {self._shell(self.settings.cluster_memory)} "
                f"--cpus-per-task {self.settings.cluster_cpus_per_task} "
                f"--job-name {self._shell('batch_api')} "
                f"--output {self._shell(self.settings.cluster_log_dir + '/batch_api_%j.out')} "
                f"--error {self._shell(self.settings.cluster_log_dir + '/batch_api_%j.err')} "
                f"{dependency_part}"
                f"--wrap {self._shell(wrap_command)}"
            )
            stdout = self._exec_checked(client, command).strip()
        finally:
            client.close()

        job_id = self._parse_job_id(stdout)
        record = ClusterJobRecord(
            request_id=request_id,
            source=source,
            title=title,
            job_id=job_id,
            submitted_at=submitted_at,
            remote_request_json=remote_request_json,
            remote_result_json=remote_result_json,
            remote_source=payload["source"],
            remote_log_glob=remote_log_glob,
            persist_artifacts=persist_artifacts,
        )
        self._write_local_record(record)
        return record

    def submit_many(
        self,
        requests: list[dict[str, Any]],
        *,
        writer: str = "",
        sequential: bool = True,
    ) -> list[ClusterJobRecord]:
        records: list[ClusterJobRecord] = []
        previous_job_id = ""
        for item in requests:
            record = self.submit(
                source=str(item.get("source", "")),
                title=str(item.get("title", "")),
                transcript_segments=item.get("transcript_segments"),
                persist_artifacts=bool(item.get("persist_artifacts", True)),
                writer=str(item.get("writer", writer) or writer),
                dependency_after_job_id=previous_job_id if sequential else "",
            )
            records.append(record)
            previous_job_id = record.job_id
        return records

    def jobs(self, *, limit: int = 50, include_status: bool = False) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        records: list[ClusterJobRecord] = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                records.append(ClusterJobRecord(**json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        records.sort(key=lambda item: item.submitted_at, reverse=True)

        jobs = []
        for record in records[:limit]:
            item = record.to_dict()
            item.update(
                {
                    "status_url": f"/api/jobs/{record.request_id}",
                    "result_url": f"/api/jobs/{record.request_id}/result",
                    "full_result_url": f"/api/jobs/{record.request_id}/result?include_full=true",
                    "report_url": f"/api/jobs/{record.request_id}/report",
                    "artifacts_url": f"/api/jobs/{record.request_id}/artifacts",
                }
            )
            if include_status:
                try:
                    item["latest_status"] = self.status(record.request_id)
                except Exception as exc:
                    item["latest_status_error"] = str(exc)
            jobs.append(item)

        return {
            "count": len(jobs),
            "total_known": len(records),
            "jobs_dir": str(self.jobs_dir),
            "jobs": jobs,
        }

    def clear_jobs(self, *, confirm: bool = False, delete_remote_artifacts: bool = False) -> Dict[str, Any]:
        if not confirm:
            raise ValueError("Pass confirm=true to clear saved runs.")

        records: list[ClusterJobRecord] = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                records.append(ClusterJobRecord(**json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue

        remote_deleted: list[dict[str, Any]] = []
        remote_errors: list[dict[str, str]] = []
        if delete_remote_artifacts and records:
            client = self._client()
            try:
                for record in records:
                    try:
                        remote_deleted.append(self._delete_remote_run_artifacts(client, record))
                    except Exception as exc:
                        remote_errors.append({"request_id": record.request_id, "error": str(exc)})
            finally:
                client.close()

        deleted_records = []
        delete_errors = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                deleted_records.append(path.name)
                path.unlink()
            except Exception as exc:
                delete_errors.append({"path": str(path), "error": str(exc)})

        return {
            "cleared": True,
            "delete_remote_artifacts": delete_remote_artifacts,
            "deleted_record_count": len(deleted_records),
            "deleted_records": deleted_records,
            "delete_errors": delete_errors,
            "remote_deleted": remote_deleted,
            "remote_errors": remote_errors,
        }

    def status(self, request_id: str) -> Dict[str, Any]:
        record = self._read_local_record(request_id)
        client = self._client()
        try:
            result_summary = self._read_remote_result_summary_if_exists(client, record.remote_result_json)
            if result_summary is not None:
                return {
                    "request_id": request_id,
                    "job_id": record.job_id,
                    "status": "completed",
                    "result_ready": True,
                    "result_summary": result_summary,
                }

            squeue_state = self._exec_best_effort(
                client,
                f"squeue -j {self._shell(record.job_id)} -h -o %T",
            ).strip()
            if squeue_state:
                return {
                    "request_id": request_id,
                    "job_id": record.job_id,
                    "status": squeue_state.lower(),
                    "result_ready": False,
                }

            sacct_state = self._exec_best_effort(
                client,
                f"sacct -j {self._shell(record.job_id)} --format=State --noheader | head -n 1",
            ).strip()
            normalized = sacct_state.split()[0].lower() if sacct_state else "unknown"
            return {
                "request_id": request_id,
                "job_id": record.job_id,
                "status": normalized,
                "result_ready": False,
            }
        finally:
            client.close()

    def result(self, request_id: str, *, include_full: bool = False) -> Dict[str, Any]:
        record = self._read_local_record(request_id)
        client = self._client()
        try:
            result = (
                self._read_remote_json_if_exists(client, record.remote_result_json)
                if include_full
                else self._read_remote_result_summary_if_exists(client, record.remote_result_json)
            )
        finally:
            client.close()
        if result is None:
            raise FileNotFoundError(f"Cluster result is not ready yet for request {request_id}")
        if include_full:
            return result
        return {
            "request_id": request_id,
            "job_id": record.job_id,
            "status": "completed",
            "result_ready": True,
            "result_summary": result,
            "full_result_hint": f"/api/jobs/{request_id}/result?include_full=true",
            "report_hint": f"/api/jobs/{request_id}/report",
        }

    def report(self, request_id: str) -> Dict[str, Any]:
        record = self._read_local_record(request_id)
        client = self._client()
        try:
            report = self._read_remote_report_if_exists(client, record.remote_result_json)
        finally:
            client.close()
        if report is None:
            raise FileNotFoundError(f"Cluster report is not ready yet for request {request_id}")
        return {
            "request_id": request_id,
            "job_id": record.job_id,
            **report,
        }

    def artifacts(self, request_id: str) -> Dict[str, Any]:
        record = self._read_local_record(request_id)
        client = self._client()
        try:
            payload = self._list_remote_artifacts_if_exists(client, record.remote_result_json)
        finally:
            client.close()
        if payload is None:
            raise FileNotFoundError(f"Cluster artifacts are not ready yet for request {request_id}")

        artifacts = []
        for item in payload.get("artifacts", []):
            artifact_path = str(item.get("path", ""))
            encoded = quote(artifact_path)
            artifacts.append(
                {
                    **item,
                    "view_url": f"/api/jobs/{request_id}/artifacts/{encoded}",
                    "download_url": f"/api/jobs/{request_id}/artifacts/{encoded}?download=true",
                }
            )
        return {
            "request_id": request_id,
            "job_id": record.job_id,
            **payload,
            "artifacts": artifacts,
        }

    def artifact(self, request_id: str, artifact_path: str) -> Dict[str, Any]:
        record = self._read_local_record(request_id)
        client = self._client()
        try:
            metadata = self._resolve_remote_artifact_if_exists(
                client,
                remote_result_path=record.remote_result_json,
                artifact_path=artifact_path,
            )
            if metadata is None:
                raise FileNotFoundError(f"Cluster artifact is not ready yet for request {request_id}: {artifact_path}")
            with client.open_sftp() as sftp:
                with sftp.file(metadata["remote_path"], "rb") as handle:
                    content = handle.read()
        finally:
            client.close()
        return {
            "request_id": request_id,
            "job_id": record.job_id,
            **metadata,
            "content": content,
        }

    def queue(self) -> Dict[str, Any]:
        client = self._client()
        try:
            command = (
                "squeue "
                f"-u {self._shell(self.settings.cluster_user)} "
                "--noheader "
                "--Format=jobid:18,name:24,partition:14,state:14,timeused:12,timelimit:12,nodelist:24,reason:40"
            )
            output = self._exec_best_effort(client, command)
        finally:
            client.close()
        jobs = []
        for line in output.splitlines():
            line = line.rstrip()
            if not line:
                continue
            parts = line.split(None, 7)
            jobs.append(
                {
                    "job_id": parts[0] if len(parts) > 0 else "",
                    "name": parts[1] if len(parts) > 1 else "",
                    "partition": parts[2] if len(parts) > 2 else "",
                    "state": parts[3] if len(parts) > 3 else "",
                    "time_used": parts[4] if len(parts) > 4 else "",
                    "time_limit": parts[5] if len(parts) > 5 else "",
                    "node": parts[6] if len(parts) > 6 else "",
                    "reason": parts[7] if len(parts) > 7 else "",
                    "raw": line,
                }
            )
        return {
            "user": self.settings.cluster_user,
            "count": len(jobs),
            "jobs": jobs,
            "raw": output,
        }

    def accounting(self, *, since: str = "today", limit: int = 20) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        client = self._client()
        try:
            command = (
                "sacct "
                f"-u {self._shell(self.settings.cluster_user)} "
                "--format=JobID,JobName,Partition,State,ExitCode,Elapsed "
                f"-S {self._shell(since)} "
                f"| tail -n +3 | head -n {limit}"
            )
            output = self._exec_best_effort(client, command)
        finally:
            client.close()
        jobs = []
        for line in output.splitlines():
            line = line.rstrip()
            if not line:
                continue
            parts = line.split()
            jobs.append(
                {
                    "job_id": parts[0] if len(parts) > 0 else "",
                    "name": parts[1] if len(parts) > 1 else "",
                    "partition": parts[2] if len(parts) > 2 else "",
                    "state": parts[3] if len(parts) > 3 else "",
                    "exit_code": parts[4] if len(parts) > 4 else "",
                    "elapsed": parts[5] if len(parts) > 5 else "",
                    "raw": line,
                }
            )
        return {
            "user": self.settings.cluster_user,
            "since": since,
            "count": len(jobs),
            "jobs": jobs,
            "raw": output,
        }

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        job_id = str(job_id or "").strip()
        if not job_id or not re.fullmatch(r"\d+(?:_\d+)?(?:\.\w+)?", job_id):
            raise ValueError(f"Invalid Slurm job id: {job_id!r}")
        client = self._client()
        try:
            before = self._exec_best_effort(
                client,
                f"squeue -j {self._shell(job_id)} -h -o '%A %T %M %R'",
            ).strip()
            self._exec_checked(client, f"scancel {self._shell(job_id)}")
            after = self._exec_best_effort(
                client,
                f"squeue -j {self._shell(job_id)} -h -o '%A %T %M %R'",
            ).strip()
            sacct = self._exec_best_effort(
                client,
                f"sacct -j {self._shell(job_id)} --format=JobID,JobName,State,ExitCode,Elapsed --noheader",
            ).strip()
        finally:
            client.close()
        return {
            "job_id": job_id,
            "cancel_requested": True,
            "queue_before": before,
            "queue_after": after,
            "accounting": sacct,
        }

    def _prepare_remote_source(self, client: paramiko.SSHClient, *, request_id: str, source: str) -> str:
        parsed = urlparse(source)
        remote_dir = f"{self.settings.cluster_project_dir}/batch_mode/cluster_jobs/{request_id}"
        if parsed.scheme in {"http", "https"}:
            remote_source = f"{remote_dir}/{self._remote_filename_from_url(source)}"
            self._exec_checked(
                client,
                " ".join(
                    [
                        "curl",
                        "-L",
                        "--fail",
                        "--show-error",
                        "--connect-timeout",
                        "30",
                        "--max-time",
                        str(self.settings.batch_download_timeout_sec),
                        "-o",
                        self._shell(remote_source),
                        self._shell(source),
                    ]
                ),
            )
            return remote_source
        if source.startswith("/") and self._remote_file_exists(client, source):
            return source
        local_path = Path(source)
        if not local_path.exists():
            raise FileNotFoundError(f"Video source not found: {source}")
        remote_source = f"{remote_dir}/{local_path.name}"
        with client.open_sftp() as sftp:
            sftp.put(str(local_path), remote_source)
        return remote_source

    def _remote_file_exists(self, client: paramiko.SSHClient, remote_path: str) -> bool:
        command = f"test -f {self._shell(remote_path)} && echo yes || true"
        return self._exec_best_effort(client, command).strip() == "yes"

    def _remote_filename_from_url(self, source: str) -> str:
        parsed = urlparse(source)
        filename = Path(parsed.path).name or "input.mp4"
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
        if not filename:
            filename = "input.mp4"
        if not Path(filename).suffix:
            filename = f"{filename}.mp4"
        return filename

    def _prepare_remote_request(
        self,
        client: paramiko.SSHClient,
        *,
        source: str,
        request_json: str,
        transcript_json: str,
        title: str,
        persist_artifacts: bool,
    ) -> None:
        command_parts = [
            f"cd {self._shell(self.settings.cluster_project_dir)}",
            f"export PYTHONPATH={self._shell(self.settings.cluster_project_dir)}:${{PYTHONPATH:-}}",
            f"export DEEPGRAM_API_KEY={self._shell(self.settings.deepgram_api_key)}",
            f"\"${{CONDA_BIN:-${{HOME}}/anaconda3/bin/conda}}\" run --no-capture-output -n {self._shell(self.settings.cluster_conda_env)} "
            f"python {self._shell(self.settings.cluster_project_dir + '/batch_mode/myproject/prepare_cluster_request.py')} "
            f"--source {self._shell(source)} "
            f"--request-json {self._shell(request_json)} "
            f"--transcript-json {self._shell(transcript_json)} "
            f"--title {self._shell(title)} "
            f"--require-transcript"
            + ("" if persist_artifacts else " --no-persist-artifacts"),
        ]
        self._exec_checked(client, " && ".join(command_parts))

    def _read_remote_json_if_exists(self, client: paramiko.SSHClient, remote_path: str) -> Optional[Dict[str, Any]]:
        script = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"from pathlib import Path\n"
            f"path = Path({remote_path!r})\n"
            f"print(path.read_text(encoding='utf-8') if path.exists() else '')\n"
            f"PY"
        )
        output = self._exec_best_effort(client, script).strip()
        if not output:
            return None
        return json.loads(output)

    def _read_remote_result_summary_if_exists(
        self,
        client: paramiko.SSHClient,
        remote_path: str,
    ) -> Optional[Dict[str, Any]]:
        script = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"from pathlib import Path\n"
            f"path = Path({remote_path!r})\n"
            f"if not path.exists():\n"
            f"    raise SystemExit\n"
            f"data = json.loads(path.read_text(encoding='utf-8'))\n"
            f"output_path = data.get('output_path') or ''\n"
            f"output = Path(output_path) if output_path else None\n"
            f"analysis_path = str(output) if output else ''\n"
            f"report_path = str(output.with_name('report.md')) if output else ''\n"
            f"llm = data.get('llm_output') or {{}}\n"
            f"summary = {{\n"
            f"    'source': data.get('source'),\n"
            f"    'resolved_video_path': data.get('resolved_video_path'),\n"
            f"    'output_path': output_path,\n"
            f"    'analysis_json': analysis_path,\n"
            f"    'report_md': report_path,\n"
            f"    'counts': {{\n"
            f"        'transcript_segments': len(data.get('transcript_segments') or []),\n"
            f"        'frame_events': len(data.get('frame_events') or []),\n"
            f"        'topic_units': len(data.get('topic_units') or []),\n"
            f"        'topic_clusters': len(data.get('topic_clusters') or []),\n"
            f"    }},\n"
            f"    'llm': {{\n"
            f"        'backend': llm.get('backend'),\n"
            f"        'model': llm.get('model'),\n"
            f"        'final_markdown_chars': len(str(llm.get('final_markdown') or '')),\n"
            f"        'error': llm.get('error'),\n"
            f"    }},\n"
            f"    'profiling': data.get('profiling') or {{}},\n"
            f"}}\n"
            f"print(json.dumps(summary, ensure_ascii=True))\n"
            f"PY"
        )
        output = self._exec_best_effort(client, script).strip()
        if not output:
            return None
        return json.loads(output)

    def _read_remote_report_if_exists(
        self,
        client: paramiko.SSHClient,
        remote_result_path: str,
    ) -> Optional[Dict[str, Any]]:
        script = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"from pathlib import Path\n"
            f"result_path = Path({remote_result_path!r})\n"
            f"if not result_path.exists():\n"
            f"    raise SystemExit\n"
            f"data = json.loads(result_path.read_text(encoding='utf-8'))\n"
            f"output_path = data.get('output_path') or ''\n"
            f"report_path = Path(output_path).with_name('report.md') if output_path else None\n"
            f"if not report_path or not report_path.exists():\n"
            f"    raise SystemExit\n"
            f"payload = {{\n"
            f"    'report_md': str(report_path),\n"
            f"    'markdown': report_path.read_text(encoding='utf-8'),\n"
            f"}}\n"
            f"print(json.dumps(payload, ensure_ascii=True))\n"
            f"PY"
        )
        output = self._exec_best_effort(client, script).strip()
        if not output:
            return None
        return json.loads(output)

    def _list_remote_artifacts_if_exists(
        self,
        client: paramiko.SSHClient,
        remote_result_path: str,
    ) -> Optional[Dict[str, Any]]:
        script = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"from pathlib import Path\n"
            f"result_path = Path({remote_result_path!r})\n"
            f"if not result_path.exists():\n"
            f"    raise SystemExit\n"
            f"data = json.loads(result_path.read_text(encoding='utf-8'))\n"
            f"output_path = data.get('output_path') or ''\n"
            f"output = Path(output_path) if output_path else None\n"
            f"output_dir = output.parent if output else None\n"
            f"artifacts = []\n"
            f"if output_dir and output_dir.exists():\n"
            f"    for path in sorted(output_dir.rglob('*')):\n"
            f"        if not path.is_file():\n"
            f"            continue\n"
            f"        stat = path.stat()\n"
            f"        artifacts.append({{\n"
            f"            'path': path.relative_to(output_dir).as_posix(),\n"
            f"            'remote_path': str(path),\n"
            f"            'size_bytes': stat.st_size,\n"
            f"            'modified_time': stat.st_mtime,\n"
            f"        }})\n"
            f"payload = {{\n"
            f"    'output_dir': str(output_dir) if output_dir else '',\n"
            f"    'remote_result_json': str(result_path),\n"
            f"    'artifacts': artifacts,\n"
            f"}}\n"
            f"print(json.dumps(payload, ensure_ascii=True))\n"
            f"PY"
        )
        output = self._exec_best_effort(client, script).strip()
        if not output:
            return None
        return json.loads(output)

    def _resolve_remote_artifact_if_exists(
        self,
        client: paramiko.SSHClient,
        *,
        remote_result_path: str,
        artifact_path: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_artifact = artifact_path.strip().lstrip("/\\")
        if not normalized_artifact or ".." in Path(normalized_artifact).parts:
            raise FileNotFoundError(f"Invalid artifact path: {artifact_path}")
        script = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"from pathlib import Path\n"
            f"result_path = Path({remote_result_path!r})\n"
            f"artifact_path = {normalized_artifact!r}\n"
            f"if not result_path.exists():\n"
            f"    raise SystemExit\n"
            f"data = json.loads(result_path.read_text(encoding='utf-8'))\n"
            f"output_path = data.get('output_path') or ''\n"
            f"output = Path(output_path) if output_path else None\n"
            f"output_dir = output.parent if output else None\n"
            f"if not output_dir or not output_dir.exists():\n"
            f"    raise SystemExit\n"
            f"target = (output_dir / artifact_path).resolve()\n"
            f"root = output_dir.resolve()\n"
            f"try:\n"
            f"    target.relative_to(root)\n"
            f"except ValueError:\n"
            f"    raise SystemExit\n"
            f"if not target.exists() or not target.is_file():\n"
            f"    raise SystemExit\n"
            f"stat = target.stat()\n"
            f"payload = {{\n"
            f"    'path': artifact_path,\n"
            f"    'remote_path': str(target),\n"
            f"    'filename': target.name,\n"
            f"    'size_bytes': stat.st_size,\n"
            f"    'modified_time': stat.st_mtime,\n"
            f"}}\n"
            f"print(json.dumps(payload, ensure_ascii=True))\n"
            f"PY"
        )
        output = self._exec_best_effort(client, script).strip()
        if not output:
            return None
        return json.loads(output)

    def _delete_remote_run_artifacts(
        self,
        client: paramiko.SSHClient,
        record: ClusterJobRecord,
    ) -> Dict[str, Any]:
        script = (
            f"python - <<'PY'\n"
            f"import json\n"
            f"import shutil\n"
            f"from pathlib import Path\n"
            f"project_dir = Path({self.settings.cluster_project_dir!r}).resolve()\n"
            f"result_path = Path({record.remote_result_json!r})\n"
            f"job_dir = result_path.parent.resolve()\n"
            f"allowed_roots = [\n"
            f"    (project_dir / 'batch_mode' / 'cluster_jobs').resolve(),\n"
            f"    (project_dir / 'batch_mode' / 'outputs').resolve(),\n"
            f"]\n"
            f"targets = []\n"
            f"if result_path.exists():\n"
            f"    try:\n"
            f"        data = json.loads(result_path.read_text(encoding='utf-8'))\n"
            f"        output_path = data.get('output_path') or ''\n"
            f"        if output_path:\n"
            f"            targets.append(Path(output_path).parent.resolve())\n"
            f"    except Exception:\n"
            f"        pass\n"
            f"targets.append(job_dir)\n"
            f"deleted = []\n"
            f"skipped = []\n"
            f"for target in dict.fromkeys(targets):\n"
            f"    allowed = False\n"
            f"    for root in allowed_roots:\n"
            f"        try:\n"
            f"            target.relative_to(root)\n"
            f"            allowed = True\n"
            f"            break\n"
            f"        except ValueError:\n"
            f"            continue\n"
            f"    if not allowed:\n"
            f"        skipped.append(str(target))\n"
            f"        continue\n"
            f"    if target.exists():\n"
            f"        shutil.rmtree(target)\n"
            f"        deleted.append(str(target))\n"
            f"print(json.dumps({{'request_id': {record.request_id!r}, 'deleted': deleted, 'skipped': skipped}}, ensure_ascii=True))\n"
            f"PY"
        )
        output = self._exec_best_effort(client, script).strip()
        if not output:
            return {"request_id": record.request_id, "deleted": [], "skipped": []}
        return json.loads(output)

    def _write_remote_json(self, client: paramiko.SSHClient, remote_path: str, payload: Dict[str, Any]) -> None:
        with client.open_sftp() as sftp:
            with sftp.file(remote_path, "w") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, indent=2))

    def _write_local_record(self, record: ClusterJobRecord) -> None:
        path = self.jobs_dir / f"{record.request_id}.json"
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")

    def _read_local_record(self, request_id: str) -> ClusterJobRecord:
        path = self.jobs_dir / f"{request_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown cluster request id: {request_id}")
        return ClusterJobRecord(**json.loads(path.read_text(encoding="utf-8")))

    def _parse_job_id(self, sbatch_stdout: str) -> str:
        tokens = sbatch_stdout.strip().split()
        for token in reversed(tokens):
            if token.isdigit():
                return token
        raise RuntimeError(f"Unable to parse cluster job id from sbatch output: {sbatch_stdout}")

    def _client(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.settings.cluster_host,
            username=self.settings.cluster_user,
            password=self.settings.cluster_password,
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
        )
        return client

    def _writer_env(self, writer: str) -> Dict[str, str]:
        normalized = (writer or "").strip().lower()
        if normalized in {"", "mistral"}:
            return {
                "BATCH_LLM_BACKEND": "hf-local",
                "BATCH_LLM_MODEL": "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
                "BATCH_HF_QUANTIZATION": "",
                "BATCH_LLM_USE_PREWRITTEN": "0",
                "BATCH_LLM_MAX_PROMPT_CHARS": "60000",
                "BATCH_LLM_MAX_INPUT_TOKENS": "24576",
                "BATCH_LLM_MAX_NEW_TOKENS": "3072",
                "BATCH_LLM_SECTION_WORKERS": "1",
            }
        if normalized == "gemini":
            return {
                "BATCH_LLM_BACKEND": "gemini",
                "BATCH_LLM_MODEL": "gemini-2.5-flash",
                "BATCH_LLM_USE_PREWRITTEN": "0",
                "BATCH_LLM_MAX_PROMPT_CHARS": "80000",
                "BATCH_LLM_MAX_NEW_TOKENS": "8192",
                "BATCH_LLM_TIMEOUT_SEC": "900",
                "BATCH_LLM_SECTION_WORKERS": "1",
            }
        if normalized == "qwen":
            return {
                "BATCH_LLM_BACKEND": "hf-local",
                "BATCH_LLM_MODEL": "Qwen/Qwen3-30B-A3B-FP8",
                "BATCH_HF_QUANTIZATION": "",
            }
        if normalized in {"mock", "preserve"}:
            return {"BATCH_LLM_BACKEND": "mock"}
        raise ValueError(f"Unknown writer preset: {writer}. Use mistral, gemini, qwen, or mock.")

    def _writer_api_key(self, writer_env: Dict[str, str]) -> str:
        backend = writer_env.get("BATCH_LLM_BACKEND", "")
        if backend == "gemini":
            import os

            return self.settings.llm_api_key or os.getenv("GEMINI_API_KEY", "").strip()
        return self.settings.llm_api_key or "EMPTY"

    def _exec_checked(self, client: paramiko.SSHClient, command: str) -> str:
        stdin, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if exit_code != 0:
            raise RuntimeError(f"Cluster command failed ({exit_code}): {command}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        return out

    def _exec_best_effort(self, client: paramiko.SSHClient, command: str) -> str:
        stdin, stdout, stderr = client.exec_command(command)
        stdout.channel.recv_exit_status()
        return stdout.read().decode("utf-8", errors="replace")

    def _shell(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"
