"""
Flask Blueprint with Detector17-specific endpoints.

Registered under /detector17/* in main.py:
  POST   /detector17/analyze/<repo_name>       — kick off analysis in background (202)
  GET    /detector17/results/<repo_name>       — rows (raw)
  GET    /detector17/contributors/<repo_name>  — aggregated per-contributor KU counts
  GET    /detector17/kus                       — global KU counts for the detector17 taxonomy
  DELETE /detector17/analysis/<repo_name>      — wipe detector17 rows for one repo
  GET    /detector17/ku_risk                   — per-KU risk (binomial, 20% threshold)   [NEW]
  GET    /detector17/employee_risk             — per-employee delta risk if they leave   [NEW]
  GET    /detector17/organizationskills        — per-KU files/authors for an organisation [NEW]

The 3 NEW endpoints mirror the legacy /ku_risk, /employee_risk and
/organizationskills exactly in request/response shape, but compute over
ku_source='detector17' rows only — the legacy K1-K27 endpoints are
untouched and continue to operate on ku_source='legacy'.

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
import threading


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
    # ΝΕΟ: risk & skills για το ku_source='detector17'
    calculate_risks_detector17,
    get_ku_skills_by_organization_detector17,
    # ΝΕΟ: Impact@Loss (per-employee) για το ku_source='detector17'
    calculate_impact_at_loss_detector17,
)


# ─────────────────────────────────────────────────────────────────────────
# Blueprint + model (lazy-loaded, singleton)
# ─────────────────────────────────────────────────────────────────────────
detector17_bp = Blueprint("detector17", __name__, url_prefix="/detector17")


_DETECTOR17_MODEL = None
_DETECTOR17_MODEL_LOCK = threading.Lock()   # ΝΕΟ: thread-safe lazy init


def _get_model():
    """
    Lazy-load the Detector17 model on first request.

    Delegates checkpoint resolution to load_detector17_model(checkpoint_dir=None),
    which reads env vars in this priority order:
      1. DETECTOR17_HF_REPO   → downloads from Hugging Face (ElisavetKanidou/ku-detector-17)
      2. DETECTOR17_MODEL_PATH → local directory fallback

    Thread-safe via a double-checked lock — prevents the 5 concurrent repo
    tasks from each triggering a redundant HF download on cold start.
    """
    global _DETECTOR17_MODEL
    if _DETECTOR17_MODEL is None:
        with _DETECTOR17_MODEL_LOCK:
            if _DETECTOR17_MODEL is None:   # double-check inside the lock
                logging.info(
                    "[detector17] Loading model (resolving from env: "
                    "DETECTOR17_HF_REPO / DETECTOR17_MODEL_PATH)..."
                )
                # Pass None so the loader reads env vars itself and
                # downloads from HF if DETECTOR17_HF_REPO is set.
                _DETECTOR17_MODEL = load_detector17_model(checkpoint_dir=None)
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

# No commit_limit from the caller → analyze the full git history
_COMMIT_LIMIT_DEFAULT = None

# Lower bound for user-supplied commit_limit
_COMMIT_LIMIT_MIN = 1


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
def _extract_detector17_files(local_path: str, commit_limit: int = _COMMIT_LIMIT_DEFAULT):
    """
    Walk git log directly and build CodeFile objects for Detector17.

    Bypasses the legacy pipeline:
      - extract_contributions (which filters .java only)
      - save_commits_to_db / get_commits_from_db (DB round-trip not needed)
      - read_files_from_dict_list (which skips files whose SHA already has
        legacy analysis, and overwrites basename collisions)

    Returns ({unique_key: CodeFile}, total_commits) ready for Detector17 inference.
    The unique_key is "<sha_short>:<relative_path>" so files with the same
    basename in different directories don't collide.
    """
    repo = git.Repo(local_path)
    commits = list(repo.iter_commits('HEAD', max_count=commit_limit))
    logging.info(
        f"[detector17] walking {len(commits)} commits in {local_path} "
        f"(limit={commit_limit or 'full history'})"
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
    return files, len(commits)


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
def _analyze_detector17_repository_task(repo_name, repo_url, local_path, commit_limit):
    """
    Background task: clones/pulls the repo, walks git log directly (bypassing
    the legacy .java-only pipeline), runs Detector17 on each file in parallel,
    and saves rows with ku_source='detector17'. Updates analysis_status
    throughout so the legacy GET /analysis_status endpoint reflects progress.

    `commit_limit` is now resolved from the HTTP request in analyze_detector17()
    (body or query string; env default as fallback) rather than from a
    module-level constant — this lets each caller tune the scope of its scan.
    """
    start_time = datetime.datetime.now()
    logging.info(
        f"[detector17] BG TASK: starting analysis for '{repo_name}' "
        f"(file_workers={_MAX_FILE_WORKERS}, commit_limit={commit_limit})"
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
        files, total_commits = _extract_detector17_files(local_path, commit_limit=commit_limit)
        total = len(files)
        if total == 0:
            logging.warning(
                f"[detector17] no analyzable files for {repo_name} "
                f"(no C/C++/Java/Python files in last {commit_limit} commits)"
            )
            update_analysis_status(
                repo_name, "error",
                start_time=start_time,
                end_time=datetime.datetime.now(),
                error_message=(
                    f"No files with supported extensions found in last "
                    f"{commit_limit} commits."
                )
            )
            return

        logging.info(
            f"[detector17] BG TASK: {total} files to analyze for {repo_name}"
        )

        # 3. Model (lazy-loaded once, shared across all threads via GIL)
        model = _get_model()
        analyzed = 0

        # Commit progress: a commit counts as done when all its files are done
        # (commits with no analyzable files are done from the start).
        files_left_per_commit = {}
        for f in files.values():
            files_left_per_commit[f.sha] = files_left_per_commit.get(f.sha, 0) + 1
        commits_done = total_commits - len(files_left_per_commit)
        next_commit_log = (commits_done // 100 + 1) * 100

        # 4. Parallel per-file inference (inner executor, per-task)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_FILE_WORKERS,
            thread_name_prefix=f"det17-file-{repo_name}"
        ) as file_executor:

            futures = {
                file_executor.submit(_analyze_single_file_detector17, file, model):
                    file.sha
                for file in files.values()
            }

            for future in concurrent.futures.as_completed(futures):
                sha = futures[future]
                files_left_per_commit[sha] -= 1
                if files_left_per_commit[sha] == 0:
                    commits_done += 1
                    if commits_done >= next_commit_log or commits_done == total_commits:
                        logging.info(
                            f"[detector17] BG TASK: {commits_done}/{total_commits} "
                            f"commits analyzed for {repo_name}"
                        )
                        next_commit_log = (commits_done // 100 + 1) * 100
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

    The caller SHOULD provide `commit_limit` to control how far back in git
    history the scan walks. Resolution order:
      1. JSON body:   {"commit_limit": <int>}
      2. Query param: ?commit_limit=<int>
      3. Not provided → the full git history is analyzed
    `commit_limit` must be an integer >= 1; otherwise the request
    is rejected with 400.

    Poll progress with:  GET /analysis_status?repo_name=<repo_name>
    Fetch results with:  GET /detector17/results/<repo_name>
    """
    organization = request.headers.get('X-User-Organization')
    if not organization:
        return jsonify({"error": "X-User-Organization header is required"}), 400

    # ─── Resolve commit_limit (body → query → env default) ──────────────
    raw_commit_limit = None
    body = request.get_json(silent=True) or {}
    if "commit_limit" in body:
        raw_commit_limit = body["commit_limit"]
    elif request.args.get("commit_limit") is not None:
        raw_commit_limit = request.args.get("commit_limit")

    if raw_commit_limit is None:
        commit_limit = _COMMIT_LIMIT_DEFAULT
        commit_limit_source = "default"
    else:
        try:
            commit_limit = int(raw_commit_limit)
        except (TypeError, ValueError):
            return jsonify({
                "error": "commit_limit must be an integer"
            }), 400
        if commit_limit < _COMMIT_LIMIT_MIN:
            return jsonify({
                "error": f"commit_limit must be at least {_COMMIT_LIMIT_MIN}"
            }), 400
        commit_limit_source = "user"

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
            repo_name, repo_url, local_path, commit_limit
        )
        logging.info(
            f"[detector17] submitted BG task for '{repo_name}' "
            f"(up to 5 repos run in parallel, commit_limit={commit_limit}, "
            f"source={commit_limit_source})"
        )
        return jsonify({
            "message":            "Detector17 analysis started in the background.",
            "repo_name":          repo_name,
            "ku_source":          "detector17",
            "commit_limit":       commit_limit,
            "commit_limit_source": commit_limit_source,
            "status_endpoint":    f"/analysis_status?repo_name={repo_name}",
            "results_endpoint":   f"/detector17/results/{repo_name}",
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


# ═════════════════════════════════════════════════════════════════════════
# ΝΕΟ — Risk & Skills endpoints για το Detector17 (ku_source='detector17')
# Μιμούνται ακριβώς το σχήμα των legacy /ku_risk, /employee_risk,
# /organizationskills, αλλά υπολογίζονται ΜΟΝΟ πάνω στα Detector17 rows.
# Τα legacy endpoints δεν θίγονται.
# ═════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# GET /detector17/ku_risk
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/ku_risk", methods=["GET"])
def get_ku_risk_detector17():
    """
    Calculates and returns the risk associated with each Detector17 KU.
    Same binomial model (20% departure threshold, p_individual_leave=0.1)
    as the legacy /ku_risk endpoint, but computed over ku_source='detector17'
    rows ONLY — so results reflect the 17 custom IoT/JEE/DT KUs.

    Headers:
      X-User-Organization (required)
    Query Parameters (optional):
      repo_name — filter risk to a single repository within the organization.
    """
    organization = request.headers.get("X-User-Organization")
    if not organization:
        return jsonify({"error": "X-User-Organization header is required"}), 400

    repo_name = request.args.get("repo_name")   # None → whole organization

    try:
        risk_data = calculate_risks_detector17(
            organization=organization, repo_name=repo_name
        )
        if "error" in risk_data:
            return jsonify(risk_data), 500

        ku_risk_list = [
            {"ku_name": ku, **data}
            for ku, data in risk_data.get("ku_risk", {}).items()
        ]
        return jsonify(ku_risk_list), 200

    except Exception as e:
        logging.exception("[detector17] /ku_risk failed")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# GET /detector17/employee_risk
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/employee_risk", methods=["GET"])
def get_employee_risk_detector17():
    """
    Calculates the risk associated with the hypothetical departure of each
    employee, considering ONLY KUs from the Detector17 taxonomy. Mirrors
    the legacy /employee_risk endpoint in shape and semantics.

    Headers:
      X-User-Organization (required)
    Query Parameters (optional):
      repo_name — filter to a single repository within the organization.
    """
    organization = request.headers.get("X-User-Organization")
    if not organization:
        return jsonify({"error": "X-User-Organization header is required"}), 400

    repo_name = request.args.get("repo_name")   # None → whole organization

    try:
        risk_data = calculate_risks_detector17(
            organization=organization, repo_name=repo_name
        )
        if "error" in risk_data:
            return jsonify(risk_data), 500

        employee_risk_list = [
            {"employee_name": employee, **data}
            for employee, data in risk_data.get("employee_risk", {}).items()
        ]
        return jsonify(employee_risk_list), 200

    except Exception as e:
        logging.exception("[detector17] /employee_risk failed")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────
# GET /detector17/organizationskills
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/organizationskills", methods=["GET"])
def get_organization_skills_detector17():
    """
    Returns a summary of 'knowledge skills' for the organization, computed
    over Detector17 KUs only (ku_source='detector17'). For each KU, returns
    the total number of unique files and unique authors.

    Headers:
      X-User-Organization (required)
    """
    organization_name = request.headers.get("X-User-Organization")
    if not organization_name:
        return jsonify({"error": "X-User-Organization header is required"}), 400

    try:
        skills_data = get_ku_skills_by_organization_detector17(organization_name)
        if skills_data is None:
            return jsonify({
                "error": "Failed to retrieve Detector17 skills for the organization"
            }), 500
        return jsonify(skills_data), 200

    except Exception as e:
        logging.exception("[detector17] /organizationskills failed")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# ─────────────────────────────────────────────────────────────────────────
# ΝΕΟ — GET /detector17/impact_at_loss (per-employee, Detector17 KUs)
# Αντικαθιστά εννοιολογικά το /detector17/employee_risk χωρίς να το
# αντικαθιστά τεχνικά — το /detector17/employee_risk παραμένει διαθέσιμο
# για συμβατότητα. Impact@Loss ∈ [0, 1] και πάντα μη αρνητικό.
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/impact_at_loss", methods=["GET"])
def get_impact_at_loss_detector17():
    """
    Returns the Impact@Loss score per employee, considering ONLY KUs from
    the Detector17 taxonomy (ku_source='detector17').

    Impact@Loss = AVG_i( my_importance[i] × IMPACT[i] )
      where my_importance[i] = KU_mine[i] / KU_total[i]
      and   IMPACT[i]        = KU_total[i] / total_files_in_scope

    Headers:
      X-User-Organization (required)
    Query Parameters (optional):
      repo_name — restrict calculation to a single repository.
    """
    organization = request.headers.get("X-User-Organization")
    if not organization:
        return jsonify({"error": "X-User-Organization header is required"}), 400

    repo_name = request.args.get("repo_name")   # None → whole organization

    try:
        data = calculate_impact_at_loss_detector17(
            organization=organization, repo_name=repo_name
        )
        if isinstance(data, dict) and "error" in data:
            return jsonify(data), 500
        return jsonify(data), 200

    except Exception as e:
        logging.exception("[detector17] /impact_at_loss failed")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500