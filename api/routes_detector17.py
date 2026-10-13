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

Why we bypass extract_contributions + read_files_from_dict_list:
  - The legacy loader filters for .java only (K1-K27 is Java-focused).
  - It skips files whose SHA already has ANY analysis — so a repo analyzed by
    the legacy /analyze endpoint can never be analyzed by Detector17 afterwards.
  - It uses basename-only as dict key — files with same name in different
    directories overwrite each other.
Detector17 was trained on C/C++, Java, Python, JS, Xtext/EMF, etc. We walk
git log directly with GitPython, filter by a broader extension set, and build
CodeFile objects in-memory (no DB round-trip for raw commits).

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

import git
from flask import Blueprint, request, jsonify

from core.utils.code_file import CodeFile
from core.git_operations import clone_repo, repo_exists
from core.git_operations.repo import pull_repo
from core.analysis.detector17_analyzer import detector17_file_scan
from core.ml_operations.detector17_loader import load_detector17_model
from config.settings import CLONED_REPO_BASE_PATH
from api.data_db import (
    get_all_repos_from_db,
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
_detector17_bg_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="det17-repo"
)

try:
    _MAX_FILE_WORKERS = int(os.getenv("DETECTOR17_FILE_WORKERS",
                                      os.getenv("ANALYSIS_WORKERS", "4")))
except (ValueError, TypeError):
    _MAX_FILE_WORKERS = 4

try:
    _COMMIT_LIMIT = int(os.getenv("DETECTOR17_COMMIT_LIMIT", "50"))
except (ValueError, TypeError):
    _COMMIT_LIMIT = 50


# ─────────────────────────────────────────────────────────────────────────
# File filtering — Detector17 accepts a broader extension set than legacy
# ─────────────────────────────────────────────────────────────────────────
_DETECTOR17_EXTENSIONS = {
    # C/C++ (IoT, Embedded_RTOS, OPC_UA, MQTT native implementations)
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
    # Java (JEE, Eclipse_RCP, EMF, MQTT Java clients)
    '.java',
    # Python (scripts, IoT prototyping)
    '.py',
    # JavaScript/TypeScript (REST_APIs, WebSockets)
    '.js', '.ts', '.jsx', '.tsx', '.mjs',
    # Go/Rust (modern IoT, microservices)
    '.go', '.rs',
    # Xtext/EMF model files
    '.xtend', '.ecore', '.xtext',
    # Build systems (CMake KU)
    '.cmake',
}
_DETECTOR17_SPECIAL_FILENAMES = {'CMakeLists.txt'}
# Skip very large files (>500KB) — likely generated code, slow to tokenize
_MAX_FILE_SIZE_BYTES = 500 * 1024


def _is_analyzable_for_detector17(path: str) -> bool:
    """Return True if this file path is relevant for the 17 custom KUs."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _DETECTOR17_EXTENSIONS:
        return True
    if os.path.basename(path) in _DETECTOR17_SPECIAL_FILENAMES:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Direct git log walker — bypasses legacy extract_contributions pipeline
# ─────────────────────────────────────────────────────────────────────────
def _extract_detector17_files(local_path: str, commit_limit: int = _COMMIT_LIMIT):
    """
    Walk git log directly and build CodeFile objects for Detector17.

    Bypasses the legacy pipeline:
      - extract_contributions (which filters .java only)
      - save_commits_to_db / get_commits_from_db (DB round-trip not needed)
      - read_files_from_dict_list (which skips files whose SHA already has
        legacy analysis, and overwrites basename collisions)

    Returns a dict {unique_key: CodeFile} ready for Detector17 inference.
    The unique_key is "<sha_short>:<relative_path>" so files with the same
    basename in different directories don't collide.
    """
    repo = git.Repo(local_path)
    commits = list(repo.iter_commits('HEAD', max_count=commit_limit))
    logging.info(
        f"[detector17] walking {len(commits)} commits in {local_path} "
        f"(limit={commit_limit})"
    )

    files = {}
    for commit in commits:
        try:
            # Diff vs first parent; for root commit, list all files
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=False)
            else:
                diffs = commit.diff(git.NULL_TREE, create_patch=False, R=True)

            for diff in diffs:
                # Prefer the "new" path (b_path); fall back to old (a_path)
                path = diff.b_path or diff.a_path
                if not path or not _is_analyzable_for_detector17(path):
                    continue

                # Read file content at this commit via the tree blob
                try:
                    blob = commit.tree / path
                    content_bytes = blob.data_stream.read()
                except (KeyError, Exception) as e:
                    # File was deleted in this commit or path doesn't resolve
                    logging.debug(
                        f"[detector17] skip {path} @ {commit.hexsha[:8]}: {e}"
                    )
                    continue

                if len(content_bytes) > _MAX_FILE_SIZE_BYTES:
                    logging.debug(
                        f"[detector17] skip oversized {path} "
                        f"({len(content_bytes)} bytes)"
                    )
                    continue

                try:
                    content = content_bytes.decode('utf-8', errors='ignore')
                except Exception:
                    continue

                if len(content.strip()) == 0:
                    continue

                # Unique key avoids basename-collision overwrites
                unique_key = f"{commit.hexsha[:8]}:{path}"
                files[unique_key] = CodeFile(
                    filename=path,   # full relative path, not basename
                    content=content,
                    author=commit.author.email or commit.author.name or "unknown",
                    timestamp=commit.committed_datetime,
                    sha=commit.hexsha,
                )
        except Exception as e:
            logging.exception(
                f"[detector17] failed on commit {commit.hexsha[:8]}: {e}"
            )

    logging.info(
        f"[detector17] extracted {len(files)} analyzable files from {local_path}"
    )
    return files


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
    Background task: clones/pulls the repo, walks git log directly (bypassing
    the legacy .java-only pipeline), runs Detector17 on each file in parallel,
    and saves rows with ku_source='detector17'. Updates analysis_status
    throughout so the legacy GET /analysis_status endpoint reflects progress.
    """
    start_time = datetime.datetime.now()
    logging.info(
        f"[detector17] BG TASK: starting analysis for '{repo_name}' "
        f"(file_workers={_MAX_FILE_WORKERS}, commit_limit={_COMMIT_LIMIT})"
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
            try:
                pull_repo(local_path)
            except Exception as e:
                logging.warning(f"[detector17] pull failed, continuing: {e}")

        # 2. Build file list directly from git log (bypass legacy loader)
        files = _extract_detector17_files(local_path, commit_limit=_COMMIT_LIMIT)
        total = len(files)
        if total == 0:
            logging.warning(
                f"[detector17] no analyzable files for {repo_name} "
                f"(no C/C++/Java/Python files in last {_COMMIT_LIMIT} commits)"
            )
            update_analysis_status(
                repo_name, "error",
                start_time=start_time,
                end_time=datetime.datetime.now(),
                error_message=(
                    f"No files with supported extensions found in last "
                    f"{_COMMIT_LIMIT} commits."
                )
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
            "message":          "Detector17 analysis started in the background.",
            "repo_name":        repo_name,
            "ku_source":        "detector17",
            "status_endpoint":  f"/analysis_status?repo_name={repo_name}",
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