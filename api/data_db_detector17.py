"""
DB operations specific to Detector17 (ku_source='detector17').

Mirrors the aggregation functions of data_db.py but filtered by ku_source,
so the UI can switch between the two taxonomies without crosstalk.

Also provides a save function that stamps every row with ku_source.
"""

import json
import logging
from datetime import datetime
from collections import defaultdict

from api.data_db import get_db_connection


# ─────────────────────────────────────────────────────────────────────────
# SAVE — new row with explicit ku_source
# ─────────────────────────────────────────────────────────────────────────

def save_analysis_to_db_with_source(repo_name, file_data, ku_source: str):
    """
    Insert a row into analysis_results tagging it with ku_source.
    ku_source must be either 'legacy' or 'detector17'.
    """
    if ku_source not in ("legacy", "detector17"):
        raise ValueError(f"Invalid ku_source: {ku_source}")

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        detected_kus_json = json.dumps(file_data["detected_kus"], default=str)
        ts = file_data["timestamp"]
        ts_str = ts.isoformat() if isinstance(ts, datetime) else ts

        cur.execute('''
            INSERT INTO analysis_results
                (repo_name, filename, author, timestamp, sha, detected_kus,
                 elapsed_time, ku_source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            repo_name,
            file_data["filename"],
            file_data["author"],
            ts_str,
            file_data["sha"],
            detected_kus_json,
            file_data["elapsed_time"],
            ku_source,
        ))
        conn.commit()
        cur.close()
    except Exception as e:
        logging.exception(f"Failed to save analysis for {file_data.get('filename')}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────
# READ — same shape as get_analysis_from_db but filtered by ku_source
# ─────────────────────────────────────────────────────────────────────────

def get_analysis_from_db_by_source(repo_name: str, ku_source: str = "detector17"):
    """
    Returns analysis rows for a repo, filtered by ku_source.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            WHERE repo_name = %s AND ku_source = %s
            ORDER BY timestamp DESC
        ''', (repo_name, ku_source))
        rows = cur.fetchall()
        cur.close()

        out = []
        for filename, author, ts, sha, detected_kus, elapsed in rows:
            if isinstance(detected_kus, str):
                detected_kus = json.loads(detected_kus)
            ts_out = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            out.append({
                "filename": filename,
                "author": author,
                "timestamp": ts_out,
                "sha": sha,
                "detected_kus": detected_kus,
                "elapsed_time": elapsed,
            })
        return out
    except Exception as e:
        logging.exception(f"get_analysis_from_db_by_source failed for {repo_name}")
        return None
    finally:
        conn.close()


def get_ku_counts_by_source(ku_source: str = "detector17", organization: str = None):
    """
    Returns [{"ku_id": "...", "count": N}, ...] filtered by ku_source.
    """
    base_query = """
        SELECT ku.key AS ku_id, COUNT(*) AS ku_count
        FROM analysis_results ar
        JOIN repositories r ON ar.repo_name = r.name,
             LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE ku.value = '1' AND ar.ku_source = %s
    """
    params = [ku_source]
    if organization:
        base_query += " AND r.organization = %s"
        params.append(organization)
    base_query += " GROUP BY ku_id ORDER BY ku_count DESC;"

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()
        return [{"ku_id": r[0], "count": int(r[1])} for r in rows]
    except Exception as e:
        logging.exception("get_ku_counts_by_source failed")
        return None
    finally:
        conn.close()


def get_contributor_kus_by_source(repo_name: str, ku_source: str = "detector17"):
    """
    Returns per-contributor file counts per KU for a repo.
    Shape: { author_email: { ku_id: n_files_detected } }
    This is what Elisavet's React UI heatmap consumes.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute('''
            SELECT ar.author, ku.key AS ku_id, COUNT(DISTINCT ar.filename) AS n_files
            FROM analysis_results ar,
                 LATERAL jsonb_each_text(ar.detected_kus) AS ku
            WHERE ar.repo_name = %s
              AND ar.ku_source = %s
              AND ku.value = '1'
            GROUP BY ar.author, ku.key
            ORDER BY ar.author, n_files DESC
        ''', (repo_name, ku_source))
        rows = cur.fetchall()
        cur.close()

        contrib = defaultdict(dict)
        for author, ku_id, n_files in rows:
            contrib[author][ku_id] = int(n_files)
        return dict(contrib)
    except Exception as e:
        logging.exception("get_contributor_kus_by_source failed")
        return None
    finally:
        conn.close()


def delete_analysis_by_source(repo_name: str, ku_source: str):
    """Clear only the Detector17 rows for a repo — leaves legacy intact."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            'DELETE FROM analysis_results WHERE repo_name = %s AND ku_source = %s',
            (repo_name, ku_source)
        )
        n = cur.rowcount
        conn.commit()
        cur.close()
        return n
    finally:
        conn.close()
