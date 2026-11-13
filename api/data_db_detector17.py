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
    Calculates KU Risk (per Knowledge Unit) ONLY for Detector17 KUs.

    SCOPE:
      - files/impact → μόνο η ΤΡΕΧΟΥΣΑ έκδοση κάθε αρχείου (latest ανά filename).
      - authors/employee_count → ΟΛΟΙ οι ιστορικοί contributors των αρχείων που
        έχουν το KU στην τρέχουσα έκδοση.

    Το employee-level risk δεν υπολογίζεται πλέον εδώ — χρησιμοποιείται
    το Impact@Loss. Επιστρέφεται κενό dict για συμβατότητα.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        base_query = '''
            SELECT ar.filename, ar.author, ar.detected_kus, ar.timestamp
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

        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        cur.execute(base_query, tuple(params))
        analysis_data = cur.fetchall()
        cur.close()
        conn.close()
        conn = None

        if not analysis_data:
            return {"ku_risk": {}, "employee_risk": {}}

        # (1) Τρέχουσα έκδοση: latest row ανά filename
        latest_by_file = {}   # filename -> (timestamp, detected_kus)
        # (2) Όλοι οι ιστορικοί authors ανά filename
        authors_by_file = defaultdict(set)   # filename -> {authors}

        for filename, author, detected_kus, ts in analysis_data:
            authors_by_file[filename].add(author)
            if filename not in latest_by_file or ts > latest_by_file[filename][0]:
                latest_by_file[filename] = (ts, detected_kus)

        total_files = len(latest_by_file)   # αρχεία τρέχουσας έκδοσης
        if total_files == 0:
            return {"ku_risk": {}, "employee_risk": {}}

        knowledge_units = defaultdict(lambda: {'files': set(), 'authors': set()})
        for filename, (ts, detected_kus) in latest_by_file.items():
            if isinstance(detected_kus, str):
                detected_kus = json.loads(detected_kus)
            for ku, is_present in detected_kus.items():
                if int(is_present) == 1:
                    knowledge_units[ku]['files'].add(filename)
                    # ΟΛΟΙ όσοι άγγιξαν ΠΟΤΕ αυτό το (τρέχον) αρχείο
                    knowledge_units[ku]['authors'].update(authors_by_file[filename])

        ku_risk_results = {}
        for ku, data in knowledge_units.items():
            emps = len(data['authors'])          # όλοι οι ιστορικοί contributors
            freq = len(data['files'])            # distinct τρέχοντα αρχεία με το KU
            p_L = _probability_of_loss(emps)
            impact = freq / total_files          # ∈ [0, 1]
            ku_risk = p_L * impact
            ku_risk_results[ku] = {
                "ku_risk": ku_risk,
                "probability_of_loss": p_L,
                "impact": impact,
                "employee_count": emps,
                "file_frequency": freq,
                "employees_at_risk_threshold": max(1, math.ceil(0.20 * emps)),
            }

        return {
            "ku_risk": ku_risk_results,
            "employee_risk": {}   # deprecated — χρησιμοποιείται το Impact@Loss
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



# ═════════════════════════════════════════════════════════════════════════
# ΝΕΟ — Impact@Loss (Detector17 variant, ku_source='detector17')
# ─────────────────────────────────────────────────────────────────────────
# Mirror του calculate_impact_at_loss() από data_db.py, φιλτραρισμένο στα
# Detector17 KUs. Ο legacy calculate_risks_detector17 ΔΕΝ τροποποιείται.
#
# Για κάθε author και κάθε Detector17 KU στο οποίο έχει συνεισφέρει:
#     my_importance[i] = KU_mine[i] / KU_total[i]
#     IMPACT[i]        = KU_total[i] / total_files
#     Impact@Loss      = AVG_i( my_importance[i] × IMPACT[i] )
# ═════════════════════════════════════════════════════════════════════════

def calculate_impact_at_loss_detector17(organization=None, repo_name=None):
    """
    Υπολογίζει το Impact@Loss ανά employee ΜΟΝΟ για Detector17 KUs
    (ku_source='detector17').

    Returns:
        [
          {"employee_name": str,
           "impact_at_loss": float,     # ∈ [0, 1]
           "ku_count": int},
          ...
        ]
        Ταξινομημένο κατά impact_at_loss φθίνοντα.

    Σε περίπτωση σφάλματος DB, επιστρέφει {"error": str}.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        base_where = " WHERE ar.ku_source = 'detector17' "
        conditions = []
        params = []
        if organization:
            conditions.append("r.organization = %s")
            params.append(organization)
        if repo_name:
            conditions.append("ar.repo_name = %s")
            params.append(repo_name)
        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""

        # 1) total distinct files στο scope (denominator του IMPACT)
        count_query = (
            "SELECT COUNT(DISTINCT ar.filename) "
            "FROM analysis_results ar "
            "JOIN repositories r ON ar.repo_name = r.name "
            + base_where + where_extra
        )
        cur.execute(count_query, tuple(params))
        row = cur.fetchone()
        total_files = int(row[0]) if row and row[0] else 0

        # 2) raw rows για να χτίσουμε KU_total[i] και KU_mine[i, author]
        select_query = (
            "SELECT ar.filename, ar.author, ar.detected_kus "
            "FROM analysis_results ar "
            "JOIN repositories r ON ar.repo_name = r.name "
            + base_where + where_extra
        )
        cur.execute(select_query, tuple(params))
        rows = cur.fetchall()

        cur.close()
        conn.close()
        conn = None

        if total_files == 0 or not rows:
            return []

        ku_files_total = defaultdict(set)
        author_ku_files = defaultdict(lambda: defaultdict(set))

        for filename, author, detected_kus in rows:
            if isinstance(detected_kus, str):
                try:
                    detected_kus = json.loads(detected_kus)
                except Exception:
                    continue
            if not detected_kus:
                continue
            for ku, is_present in detected_kus.items():
                try:
                    if int(is_present) == 1:
                        ku_files_total[ku].add(filename)
                        author_ku_files[author][ku].add(filename)
                except (TypeError, ValueError):
                    continue

        results = []
        for author, kus in author_ku_files.items():
            per_ku_scores = []
            for ku, mine_filenames in kus.items():
                ku_total = len(ku_files_total.get(ku, ()))
                if ku_total == 0:
                    continue
                mine = len(mine_filenames)
                my_importance = mine / ku_total
                impact = ku_total / total_files
                per_ku_scores.append(my_importance * impact)

            if per_ku_scores:
                impact_at_loss = sum(per_ku_scores) / len(per_ku_scores)
            else:
                impact_at_loss = 0.0

            results.append({
                "employee_name": author,
                "impact_at_loss": impact_at_loss,
                "ku_count": len(per_ku_scores),
            })

        results.sort(key=lambda x: x["impact_at_loss"], reverse=True)
        return results

    except Exception as e:
        logging.exception("calculate_impact_at_loss_detector17 failed")
        return {"error": str(e)}
    finally:
        if conn is not None:
            conn.close()