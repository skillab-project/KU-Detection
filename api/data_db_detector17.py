"""
DB operations specific to Detector17 (ku_source='detector17').

Mirrors the aggregation functions of data_db.py but filtered by ku_source,
so the UI can switch between the two taxonomies without crosstalk.

Also provides a save function that stamps every row with ku_source.
"""

import json
import math
import logging
from datetime import datetime
from collections import defaultdict

from api.data_db import get_db_connection, _probability_of_loss


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


# ═════════════════════════════════════════════════════════════════════════
# RISK & SKILLS — Detector17 variants
# Mirror calculate_risks() and get_ku_skills_by_organization() from
# data_db.py exactly (same binomial model, same thresholds, same shapes)
# but filtered to ku_source='detector17' so the two taxonomies never mix.
# ═════════════════════════════════════════════════════════════════════════

def calculate_risks_detector17(organization=None, repo_name=None):
    """
    Calculates KU Risk and Employee Risk ONLY for Detector17 KUs.

    Returns the same shape as calculate_risks() in data_db.py:
        {
          "ku_risk": {
             "<ku_name>": {
                 "ku_risk":  float,
                 "probability_of_loss": float,
                 "impact": float,
                 "employee_count": int,
                 "file_frequency": int,
                 "employees_at_risk_threshold": int
             }, ...
          },
          "employee_risk": {
             "<author>": {
                 "absolute_employee_risk": float,
                 "relative_employee_risk": float,
                 "ku_count": int
             }, ...
          }
        }

    Uses the shared _probability_of_loss() helper from data_db.py
    (binomial model, 20% threshold, p_individual_leave=0.1).
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        base_query_select = '''
            SELECT ar.filename, ar.author, ar.detected_kus
            FROM analysis_results ar
            JOIN repositories r ON ar.repo_name = r.name
            WHERE ar.ku_source = 'detector17'
        '''
        base_query_count = '''
            SELECT COUNT(DISTINCT ar.filename)
            FROM analysis_results ar
            JOIN repositories r ON ar.repo_name = r.name
            WHERE ar.ku_source = 'detector17'
        '''

        conditions = []
        params = []
        if organization:
            conditions.append("r.organization = %s")
            params.append(organization)
        if repo_name:
            conditions.append("ar.repo_name = %s")
            params.append(repo_name)

        where_extra = ""
        if conditions:
            where_extra = " AND " + " AND ".join(conditions)

        cur.execute(base_query_select + where_extra, tuple(params))
        analysis_data = cur.fetchall()

        cur.execute(base_query_count + where_extra, tuple(params))
        total_files_result = cur.fetchone()
        total_files = total_files_result[0] if total_files_result else 0

        cur.close()
        conn.close()
        conn = None

        if not analysis_data or total_files == 0:
            return {"ku_risk": {}, "employee_risk": {}}

        knowledge_units = defaultdict(lambda: {'freq': 0, 'authors': set()})
        author_ku_map = defaultdict(set)

        for filename, author, detected_kus in analysis_data:
            # detected_kus is JSONB → already a dict when returned by psycopg2
            if isinstance(detected_kus, str):
                detected_kus = json.loads(detected_kus)
            for ku, is_present in detected_kus.items():
                if int(is_present) == 1:
                    knowledge_units[ku]['freq'] += 1
                    knowledge_units[ku]['authors'].add(author)
                    author_ku_map[author].add(ku)

        # ─── Per-KU risk ────────────────────────────────────────────────
        ku_risk_results = {}
        for ku, data in knowledge_units.items():
            emps = len(data['authors'])
            freq = data['freq']
            p_L = _probability_of_loss(emps)
            impact = freq / total_files
            ku_risk = p_L * impact
            ku_risk_results[ku] = {
                "ku_risk": ku_risk,
                "probability_of_loss": p_L,
                "impact": impact,
                "employee_count": emps,
                "file_frequency": freq,
                "employees_at_risk_threshold": max(1, math.ceil(0.20 * emps)),
            }

        # ─── Per-employee risk (delta if they leave) ────────────────────
        employee_risk_results = {}
        for author, kus in author_ku_map.items():
            total_delta_risk = 0.0
            total_before_risk = 0.0
            for ku in kus:
                ku_data = knowledge_units[ku]
                freq = ku_data['freq']
                emps_before = len(ku_data['authors'])
                impact = freq / total_files
                p_L_before = _probability_of_loss(emps_before)
                risk_before = p_L_before * impact

                employees_leaving = max(1, math.ceil(0.20 * emps_before))
                emps_after = max(0, emps_before - employees_leaving)
                p_L_after = _probability_of_loss(emps_after)
                risk_after = p_L_after * impact

                delta_risk = risk_after - risk_before
                total_delta_risk += delta_risk
                total_before_risk += risk_before

            absolute_risk = total_delta_risk
            relative_risk = (absolute_risk / total_before_risk
                             if total_before_risk > 0 else 0.0)
            employee_risk_results[author] = {
                "absolute_employee_risk": absolute_risk,
                "relative_employee_risk": relative_risk,
                "ku_count": len(kus),
            }

        return {
            "ku_risk": ku_risk_results,
            "employee_risk": employee_risk_results,
        }

    except Exception as e:
        logging.exception("calculate_risks_detector17 failed")
        return {"error": str(e)}
    finally:
        if conn is not None:
            conn.close()


def get_ku_skills_by_organization_detector17(organization_name):
    """
    Returns per-KU summary (total_files, total_authors) for a given
    organization, filtered by ku_source='detector17'.

    Shape: [{"ku_name": str, "total_files": int, "total_authors": int}, ...]
    Mirror of get_ku_skills_by_organization() in data_db.py.
    """
    sql_query = """
        SELECT
            ku.key AS ku_name,
            COUNT(DISTINCT ar.filename) AS total_files,
            COUNT(DISTINCT ar.author) AS total_authors
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE
            r.organization = %s
            AND ku.value = '1'
            AND ar.ku_source = 'detector17'
        GROUP BY
            ku_name
        ORDER BY
            ku_name;
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query, (organization_name,))
        rows = cur.fetchall()
        cur.close()
        return [
            {"ku_name": row[0],
             "total_files": int(row[1]),
             "total_authors": int(row[2])}
            for row in rows
        ]
    except Exception as e:
        logging.exception(
            f"get_ku_skills_by_organization_detector17 failed for {organization_name}"
        )
        return None
    finally:
        if conn is not None:
            conn.close()
