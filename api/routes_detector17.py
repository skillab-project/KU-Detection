"""
Flask Blueprint with Detector17-specific endpoints.

Registered under /detector17/* in main.py:
  POST /detector17/analyze/<repo_name>       — kick off analysis with ku_source='detector17'
  GET  /detector17/results/<repo_name>       — rows (raw)
  GET  /detector17/contributors/<repo_name>  — aggregated per-contributor KU counts
  GET  /detector17/kus                       — global KU counts for the detector17 taxonomy
  DELETE /detector17/analysis/<repo_name>    — wipe detector17 rows for one repo

These endpoints live NEXT TO the existing /repos/*, /analysis/* routes.
Nothing in routes.py is touched — legacy UI keeps working.
"""

import datetime
import logging
import time
import os

from flask import Blueprint, request, jsonify

from core.utils.code_files_loader import read_files_from_dict_list
from core.git_operations import clone_repo, repo_exists, extract_contributions
from core.git_operations.repo import pull_repo, get_history_repo
from core.analysis.detector17_analyzer import detector17_file_scan
from core.ml_operations.detector17_loader import load_detector17_model
from config.settings import CLONED_REPO_BASE_PATH
from api.data_db import (
    save_commits_to_db, get_commits_from_db, get_all_repos_from_db,
    update_analysis_status,
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
    """Lazy-load the Detector17 model on first request."""
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
# POST /detector17/analyze/<repo_name>
# ─────────────────────────────────────────────────────────────────────────
@detector17_bp.route("/analyze/<string:repo_name>", methods=["POST"])
def analyze_detector17(repo_name):
    """
    Trigger Detector17 analysis for a repo already registered in /repos.
    Writes rows with ku_source='detector17'. Does NOT touch legacy rows.
    Blocking (not streamed) — for a demo UI that polls status.
    """
    repos = {r["name"]: r for r in get_all_repos_from_db()}
    if repo_name not in repos:
        return jsonify({"error": f"Repository '{repo_name}' not registered. "
                                 f"Call /repos first."}), 404

    repo_url = repos[repo_name]["url"]
    local_path = os.path.join(CLONED_REPO_BASE_PATH, repo_name)

    try:
        # 1. Ensure cloned
        if not repo_exists(local_path):
            clone_repo(repo_url, local_path)
        else:
            pull_repo(local_path)

        # 2. Pull commits (reuse legacy git_operations)
        commits = get_commits_from_db(repo_name)
        if not commits:
            commits_history = get_history_repo(local_path)
            save_commits_to_db(repo_name, commits_history)
            commits = get_commits_from_db(repo_name)

        files = read_files_from_dict_list(commits)
        total = len(files)
        if total == 0:
            return jsonify({"error": "No analyzable files found."}), 400

        # 3. Run the model
        model = _get_model()
        start = datetime.datetime.now()
        update_analysis_status(repo_name, "in-progress",
                                start_time=start, progress=0)

        analyzed = 0
        for file in files.values():
            try:
                t0 = time.time()
                detector17_file_scan([file], model)
                elapsed = time.time() - t0

                ts = file.timestamp
                ts_str = ts.isoformat() if isinstance(ts, datetime.datetime) else ts

                save_analysis_to_db_with_source(
                    repo_name,
                    file_data={
                        "filename":      file.filename,
                        "author":        file.author,
                        "timestamp":     ts_str,
                        "sha":           file.sha,
                        "detected_kus":  file.ku_results,
                        "elapsed_time":  elapsed,
                    },
                    ku_source="detector17",
                )
                analyzed += 1
                if analyzed % 10 == 0:
                    progress = int(100 * analyzed / total)
                    update_analysis_status(repo_name, "in-progress",
                                            start_time=start, progress=progress)
            except Exception as e:
                logging.exception(f"[detector17] file {file.filename} failed")

        end = datetime.datetime.now()
        update_analysis_status(repo_name, "completed",
                                start_time=start, end_time=end, progress=100)

        return jsonify({
            "status":     "completed",
            "repo":       repo_name,
            "ku_source":  "detector17",
            "files":      analyzed,
            "duration_s": (end - start).total_seconds(),
        })

    except Exception as e:
        logging.exception("[detector17] analysis failed")
        update_analysis_status(repo_name, "error",
                                end_time=datetime.datetime.now(),
                                error_message=str(e))
        return jsonify({"error": str(e)}), 500


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
