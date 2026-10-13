"""
Flask Blueprint with Detector17-specific endpoints.

Registered under /detector17/* in main.py:
  POST /detector17/analyze/<repo_name>       — kick off analysis in background (202 Accepted)
  GET  /detector17/results/<repo_name>       — rows (raw)
  GET  /detector17/contributors/<repo_name>  — aggregated per-contributor KU counts
  GET  /detector17/kus                       — global KU counts for the detector17 taxonomy
  DELETE /detector17/analysis/<repo_name>    — wipe detector17 rows for one repo

Progress is tracked via the shared analysis_status table — poll it with the
legacy GET /analysis_status?repo_name=<name> endpoint.

Threading model (matches legacy /analyze):
  - 1 outer ThreadPoolExecutor (max_workers=5) → up to 5 repos analyzed in parallel
  - 1 inner ThreadPoolExecutor (max_workers=MAX_WORKERS, default 4) per repo
    → files inside that repo analyzed in parallel

Nothing in routes.py is touched — legacy UI keeps working.
"""

import datetime
import logging
import time
import os
import concurrent.futures

from flask import Blueprint, request, jsonify

from core.utils.code_files_loader import read_files_from_dict_list
from core.git_operations import clone_repo, repo_exists, extract_contributions
from core.git_operations.repo import pull_repo
from core.analysis.detector17_analyzer import detector17_file_scan
from core.ml_operations.detector17_loader import load_detector17_model
from config.settings import CLONED_REPO_BASE_PATH
from api.data_db import (
    save_commits_to_db, get_commits_from_db, get_all_repos_from_db,
    update_analysis_status, get_analysis_status,
)
from api.data_db_detector17 import (
    save_analysis_to_db_with_source,
    get_analysis_from_db_by_source,
    get_ku_counts_by_source,
    get_contributor_kus_by_source,
    delete_analysis_by_source,
)


# ─────────────────────────────────────────────────────────────────────────
# Blueprint + model (lazy-loaded, singleton)
# ─────────────────────────────────────────────────────────────────────────
detector17_bp = Blueprint("detector17", __name__, url_prefix="/detector17")

_DETECTOR17_MODEL = None


def _get_model():
    """Lazy-load the Detector17 model on first request (thread-safe via GIL)."""
    global _DETECTOR17_MODEL
    if _DETECTOR17_MODEL is None:
        model_dir = os.getenv(
            "DETECTOR17_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "..", "models", "ku-detector-17"),
        )
        logging.info(f"[detector17] Loading model from {model_dir} ...")
        _DETECTOR17_MODEL = load_detector17_model(model_dir)
        logging.info("[detector17] Model loaded.")
    return _DETECTOR17_MODEL


# ─────────────────────────────────────────────────────────────────────────
# Background executors (match legacy /analyze pattern)
# ─────────────────────────────────────────────────────────────────────────
# Outer: 5 concurrent repo analyses (one task per repo)
_detector17_bg_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="det17-repo"
)

# Inner (per-task): parallelize file analysis within a single repo
try:
    _MAX_FILE_WORKERS = int(os.getenv("DETECTOR17_FILE_WORKERS",
                                      os.getenv("ANALYSIS_WORKERS", "4")))
except (ValueError, TypeError):
    _MAX_FILE_WORKERS = 4


# ─────────────────────────────────────────────────────────────────────────
# Per-file worker (runs inside the inner ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────
def _analyze_single_file_detector17(file, model):
    """
    Run Detector17 inference on a single file and return a dict ready for
    save_analysis_to_db_with_source(). Returns {'error': ..., 'filename': ...}
    on failure so the task can log it and move on.
    """
    try:
        t0 = time.time()
        detector17_file_scan([file], model)
        elapsed = time.time() - t0

        ts = file.timestamp
        ts_str = ts.isoformat() if isinstance(ts, datetime.datetime) else ts

        return {
            "filename":     file.filename,
            "author":       file.author,
            "timestamp":    ts_str,
            "sha":          file.sha,
            "detected_kus": file.ku_results,
            "elapsed_time": elapsed,
        }
    except Exception as e:
        logging.exception(f"[detector17] file {file.filename} failed")
        return {"error": str(e), "filename": file.filename}


# ─────────────────────────────────────────────────────────────────────────
# Background task — one per repo (runs inside _detector17_bg_executor)
# ─────────────────────────────────────────────────────────────────────────
def _analyze_detector17_repository_task(repo_name, repo_url, local_path):
    """
    Background task: clones/pulls the repo, extracts commits, runs Detector17
    on each file (in parallel via inner ThreadPoolExecutor), and saves rows
    with ku_source='detector17'. Updates analysis_status throughout so the
    legacy GET /analysis_status endpoint reflects progress.
    """
    start_time = datetime.datetime.now()
    logging.info(
        f"[detector17] BG TASK: starting analysis for '{repo_name}' "
        f"(workers={_MAX_FILE_WORKERS})"
    )
    update_analysis_status(repo_name, "in-progress",
                           start_time=start_time, progress=0)

    try:
        # 1. Ensure cloned
        if not repo_exists(repo_name):
            logging.info(f"[detector17] cloning {repo_url} → {local_path}")
            clone_repo(repo_url, local_path)
        else:
            logging.info(f"[detector17] pulling {local_path}")
            pull_repo(local_path)

        # 2. Pull commits (reuse DB cache if available)
        commits = get_commits_from_db(repo_name)
        if not commits:
            logging.info(f"[detector17] extracting commits from {local_path}")
            commits_history = extract_contributions(local_path, commit_limit=50)
            save_commits_to_db(repo_name, commits_history)
            commits = get_commits_from_db(repo_name)

        files = read_files_from_dict_list(commits)
        total = len(files)
        if total == 0:
            logging.warning(f"[detector17] no analyzable files for {repo_name}")
            update_analysis_status(
                repo_name, "error",
                start_time=start_time,
                end_time=datetime.datetime.now(),
                error_message="No analyzable files found."
            )
            return

        logging.info(
            f"[detector17] BG TASK: {total} files to analyze for {repo_name}"
        )

        # 3. Model (lazy-loaded once, shared across all threads via GIL)
        model = _get_model()
        analyzed = 0

        # 4. Parallel per-file inference (inner executor, per-task)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_FILE_WORKERS,
            thread_name_prefix=f"det17-file-{repo_name}"
        ) as file_executor:

            futures = {
                file_executor.submit(_analyze_single_file_detector17, file, model):
                    file.filename
                for file in files.values()
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if "error" in result:
                        logging.error(
                            f"[detector17] skipped {result.get('filename')}: "
                            f"{result['error']}"
                        )
                        continue

                    save_analysis_to_db_with_source(
                        repo_name,
                        file_data=result,
                        ku_source="detector17",
                    )
                    analyzed += 1

                    # Update progress every 10 files (and at the end)
                    if analyzed % 10 == 0 or analyzed == total:
                        progress = int(100 * analyzed / total)
                        update_analysis_status(
                            repo_name, "in-progress",
                            start_time=start_time, progress=progress
                        )
                        logging.info(
                            f"[detector17] BG TASK: {analyzed}/{total} "
                            f"({progress}%) for {repo_name}"
                        )
                except Exception as exc:
                    logging.exception(
                        f"[detector17] future failed for {repo_name}: {exc}"
                    )

        # 5. Done
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.info(
            f"[detector17] BG TASK: completed {repo_name} — "
            f"{analyzed}/{total} files in {duration:.1f}s"
        )
        update_analysis_status(
            repo_name, "completed",
            start_time=start_time, end_time=end_time, progress=100
        )

    except Exception as e:
        end_time = datetime.datetime.now()
        logging.exception(f"[detector17] BG TASK: critical error for {repo_name}")
        update_analysis_status(
            repo_name, "error",
            start_time=start_time, end_time=end_time,
            error_message=str(e)
        )


# ─────────────────────────────────────────────────────────────────────────
# POST /detector17/analyze/<repo_name>
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/analyze/<string:repo_name>", methods=["POST"])
def analyze_detector17(repo_name):
    """
    Kick off Detector17 analysis in the background (returns 202 immediately).

    Poll progress with:  GET /analysis_status?repo_name=<repo_name>
    Fetch results with:  GET /detector17/results/<repo_name>
    """
    organization = request.headers.get('X-User-Organization')
    if not organization:
        return jsonify({"error": "X-User-Organization header is required"}), 400

    # 1. Check that repo is registered
    repos = {r["name"]: r for r in get_all_repos_from_db(organization=organization)}
    if repo_name not in repos:
        return jsonify({
            "error": f"Repository '{repo_name}' not registered. "
                     f"Call POST /repos first."
        }), 404

    # 2. Check that no analysis is already running for this repo
    current_status = get_analysis_status(repo_name)
    if current_status and current_status.get("status") == "in-progress":
        logging.warning(
            f"[detector17] analysis for '{repo_name}' is already in progress"
        )
        return jsonify({
            "message": "Analysis is already in progress for this repository.",
            "repo_name": repo_name,
            "status_endpoint": f"/analysis_status?repo_name={repo_name}"
        }), 409

    repo_url = repos[repo_name]["url"]
    local_path = os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", str(repo_name))

    # 3. Submit to background executor and return 202 immediately
    try:
        _detector17_bg_executor.submit(
            _analyze_detector17_repository_task,
            repo_name, repo_url, local_path
        )
        logging.info(
            f"[detector17] submitted BG task for '{repo_name}' "
            f"(up to 5 repos run in parallel)"
        )
        return jsonify({
            "message":         "Detector17 analysis started in the background.",
            "repo_name":       repo_name,
            "ku_source":       "detector17",
            "status_endpoint": f"/analysis_status?repo_name={repo_name}",
            "results_endpoint": f"/detector17/results/{repo_name}",
        }), 202
    except Exception as e:
        logging.exception(f"[detector17] failed to submit BG task for {repo_name}")
        return jsonify({
            "error": f"Failed to start analysis: {e}"
        }), 500


# ─────────────────────────────────────────────────────────────────────────
# GET /detector17/results/<repo_name>
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/results/<string:repo_name>", methods=["GET"])
def get_results(repo_name):
    rows = get_analysis_from_db_by_source(repo_name, ku_source="detector17")
    if rows is None:
        return jsonify({"error": "DB read failed"}), 500
    return jsonify(rows)


# ─────────────────────────────────────────────────────────────────────────
# GET /detector17/contributors/<repo_name>
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/contributors/<string:repo_name>", methods=["GET"])
def get_contributors(repo_name):
    """
    Per-contributor file counts per KU — the shape the React heatmap wants.
    Returns:
      {
        "repo":       "...",
        "ku_source":  "detector17",
        "contributors": {
            "author@email": {"MQTT": 14, "REST_APIs": 3, ...},
            ...
        }
      }
    """
    contrib = get_contributor_kus_by_source(repo_name, ku_source="detector17")
    if contrib is None:
        return jsonify({"error": "DB read failed"}), 500
    return jsonify({
        "repo":          repo_name,
        "ku_source":     "detector17",
        "contributors":  contrib,
    })


# ─────────────────────────────────────────────────────────────────────────
# GET /detector17/kus
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/kus", methods=["GET"])
def get_kus():
    org = request.args.get("organization")
    data = get_ku_counts_by_source(ku_source="detector17", organization=org)
    if data is None:
        return jsonify({"error": "DB read failed"}), 500
    return jsonify(data)


# ─────────────────────────────────────────────────────────────────────────
# DELETE /detector17/analysis/<repo_name>
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/analysis/<string:repo_name>", methods=["DELETE"])
def delete_results(repo_name):
    n = delete_analysis_by_source(repo_name, ku_source="detector17")
    return jsonify({"deleted_rows": n, "repo": repo_name,
                    "ku_source": "detector17"})