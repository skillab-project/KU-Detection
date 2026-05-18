import json
import os
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
import time
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import subprocess
import math
import urllib.request
import urllib.error 

from scipy.stats import binom  # <-- ΝΕΟ import

# --- Βιβλιοθήκες για ML και Ανάλυση ---
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer

# --- Imports από το project ---
from core.ml_operations.loader import load_codebert_model
from core.analysis.codebert_sliding_window import codebert_sliding_window
from config.settings import CLONED_REPO_BASE_PATH, CODEBERT_BASE_PATH

# Database connection settings
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# Load model
model = load_codebert_model(CODEBERT_BASE_PATH, 27)

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def create_tables():
    table_check_query = '''
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'repositories'
    );
    '''

    commands = [
        '''
        CREATE TABLE repositories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            url VARCHAR(255),
            organization VARCHAR(255),
            description TEXT,
            comments TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_status VARCHAR(255),
            analysis_start_time TIMESTAMP,
            analysis_end_time TIMESTAMP,
            analysis_progress INTEGER,
            analysis_error_message TEXT
        )
        ''',
        '''
        CREATE TABLE commits (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255),
            author VARCHAR(255),
            file_content TEXT,
            changed_lines INTEGER[],
            temp_filepath VARCHAR(255),
            timestamp TIMESTAMP,
            sha VARCHAR(255)
        )
        ''',
        '''
        CREATE TABLE analysis_results (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255),
            filename VARCHAR(255),
            author VARCHAR(255),
            timestamp TIMESTAMP,
            sha VARCHAR(255),
            detected_kus JSONB,
            elapsed_time FLOAT
        )
        '''
    ]

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(table_check_query)
        (table_exists,) = cur.fetchone()
        if table_exists:
            print("Tables already exist. Skipping table creation.")
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='repositories' AND column_name='organization'
            """)
            if cur.fetchone() is None:
                print("Adding 'organization' column to existing 'repositories' table.")
                cur.execute("ALTER TABLE repositories ADD COLUMN organization VARCHAR(255);")
                conn.commit()
            return

        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn is not None:
            conn.close()

def execute_sql_script(cursor, script_path):
    """
    Εκτελεί ένα SQL script που περιέχει πολλαπλές εντολές.
    Κάθε εντολή αναμένεται να τερματίζεται με ερωτηματικό (;).
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        sql_commands = f.read().split(';')
        for command in sql_commands:
            command = command.strip()
            if command: # Εκτελεί μόνο μη κενές εντολές
                try:
                    cursor.execute(command)
                except psycopg2.Error as e:
                    logging.error(f"Σφάλμα κατά την εκτέλεση SQL εντολής: {command[:100]}...")
                    logging.exception(e)
                    raise # Επανεμφάνιση σφάλματος για να αποτύχει η συναλλαγή

def initialize_database():
    conn = None
    cur = None
    seed_file = "/tmp/seed_data.sql" # Ορισμός της διαδρομής του προσωρινού αρχείου

    try:
        conn = get_db_connection() # Υποθέτει ότι αυτή η συνάρτηση υπάρχει και λειτουργεί
        cur = conn.cursor()

        # Έλεγχος αν η βάση δεδομένων είναι ήδη γεμάτη
        cur.execute("SELECT COUNT(*) FROM repositories;")
        count = cur.fetchone()[0]

        if count > 0:
            logging.info(f"Η βάση δεδομένων έχει ήδη δεδομένα ({count} repos βρέθηκαν). Παραλείπεται η αρχικοποίηση.")
            return

        logging.info("Η βάση δεδομένων είναι άδεια. Πραγματοποιείται λήψη δεδομένων seed από το Hugging Face...")

        seed_url = "https://huggingface.co/datasets/nnikolaidis/skillab-ku-analysis-2/resolve/main/seed_data.sql"
        urllib.request.urlretrieve(seed_url, seed_file)
        logging.info(f"Το αρχείο seed κατέβηκε επιτυχώς στο {seed_file}. Φορτώνεται στη βάση δεδομένων...")

        # Φόρτωση δεδομένων seed και καθαρισμός εντός μιας συναλλαγής
        # Η conn.set_isolation_level δεν χρειάζεται εδώ, καθώς η commit/rollback είναι ρητή
        
        execute_sql_script(cur, seed_file)
        logging.info("Τα δεδομένα seed φορτώθηκαν επιτυχώς.")

        # Αφαίρεση apache repositories
        logging.info("Αφαιρούνται τα 'apache' repositories...")
        cur.execute("""
            DELETE FROM analysis_results
            WHERE repo_name IN (SELECT name FROM repositories WHERE organization = 'apache');
        """)
        cur.execute("""
            DELETE FROM commits
            WHERE repo_name IN (SELECT name FROM repositories WHERE organization = 'apache');
        """)
        cur.execute("DELETE FROM repositories WHERE organization = 'apache';")
        logging.info("Τα Apache repositories αφαιρέθηκαν.")

        # Οριστική αποθήκευση όλων των αλλαγών
        conn.commit()
        logging.info("Η αρχικοποίηση και ο καθαρισμός της βάσης δεδομένων ολοκληρώθηκαν με commit.")

        # Επαλήθευση (προαιρετικό)
        cur.execute("SELECT organization, COUNT(*) FROM repositories GROUP BY organization;")
        rows = cur.fetchall()
        for row in rows:
            logging.info(f"  Οργανισμός: {row[0]}, Repositories: {row[1]}")

        # Καθαρισμός του προσωρινού αρχείου seed
        if os.path.exists(seed_file):
            os.remove(seed_file)
            logging.info(f"Αφαιρέθηκε το προσωρινό αρχείο seed: {seed_file}")

    except urllib.error.URLError as e:
        logging.critical(f"Αποτυχία λήψης δεδομένων seed από {seed_url}: {e.reason}")
        raise # Επανεμφάνιση σφάλματος για να σταματήσει η εφαρμογή αν η λήψη είναι κρίσιμη
    except FileNotFoundError:
        logging.critical(f"Το προσωρινό αρχείο seed δεν βρέθηκε μετά την προσπάθεια λήψης: {seed_file}")
        raise
    except psycopg2.Error as e:
        logging.exception(f"Σφάλμα PostgreSQL κατά την αρχικοποίηση της βάσης δεδομένων: {e.pgerror}")
        if conn:
            conn.rollback() # Αναίρεση της συναλλαγής σε περίπτωση σφάλματος βάσης δεδομένων
            logging.warning("Η συναλλαγή της βάσης δεδομένων αναιρέθηκε λόγω σφάλματος.")
        raise
    except Exception as e:
        logging.exception("Μη αναμενόμενο σφάλμα κατά την αρχικοποίηση της βάσης δεδομένων:")
        if conn:
            conn.rollback() # Αναίρεση για οποιοδήποτε άλλο σφάλμα
            logging.warning("Η συναλλαγή της βάσης δεδομένων αναιρέθηκε λόγω μη αναμενόμενου σφάλματος.")
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def save_repo_to_db(name, url=None, organization=None, description=None, comments=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO repositories (name, url, organization, description, comments)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET url = EXCLUDED.url,
                organization = EXCLUDED.organization,
                description = EXCLUDED.description,
                comments = EXCLUDED.comments,
                updated_at = CURRENT_TIMESTAMP
        ''', (name, url, organization, description, comments))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

def delete_repo_from_db(repo_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM analysis_results WHERE repo_name = %s', (repo_name,))
        cur.execute('DELETE FROM commits WHERE repo_name = %s', (repo_name,))
        cur.execute('DELETE FROM repositories WHERE name = %s', (repo_name,))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        conn.close()


def get_all_repos_from_db(organization=None):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                base_query = '''
                    SELECT name, url, organization, description, comments, created_at, updated_at, 
                           analysis_status, analysis_start_time, analysis_end_time, 
                           analysis_progress, analysis_error_message
                    FROM repositories
                '''

                params = []

                if organization:
                    base_query += " WHERE organization = %s"
                    params.append(organization)

                base_query += " ORDER BY name;"

                cur.execute(base_query, tuple(params))

                rows = cur.fetchall()
                repos = []
                for row in rows:
                    repo = {
                        "name": row[0],
                        "url": row[1],
                        "organization": row[2],
                        "description": row[3],
                        "comments": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                        "updated_at": row[6].isoformat() if row[6] else None,
                        "analysis_status": row[7],
                        "analysis_start_time": row[8].isoformat() if row[8] else None,
                        "analysis_end_time": row[9].isoformat() if row[9] else None,
                        "analysis_progress": row[10],
                        "analysis_error_message": row[11]
                    }
                    repos.append(repo)
                return repos
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_commits_to_db(repo_name, commits):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        for commit in commits:
            cur.execute('''
                INSERT INTO commits (repo_name, sha, author, file_content, changed_lines, temp_filepath, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                repo_name,
                commit.get('sha'),
                commit.get('author'),
                commit.get('file_content'),
                commit.get('changed_lines'),
                commit.get('temp_filepath'),
                commit.get('timestamp')
            ))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

def get_commits_from_db(repo_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT sha, author, file_content, changed_lines, temp_filepath, timestamp
            FROM commits
            WHERE repo_name = %s
        ''', (repo_name,))
        rows = cur.fetchall()
        cur.close()

        commits = []
        for row in rows:
            commit = {
                "sha": row[0],
                "author": row[1],
                "file_content": row[2],
                "changed_lines": row[3],
                "temp_filepath": row[4],
                "timestamp": row[5]
            }
            commits.append(commit)

        return commits
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        conn.close()


def getdetected_kus(organization=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if organization:
            cur.execute('''
                SELECT ar.detected_kus, ar.author
                FROM analysis_results ar
                JOIN repositories r ON ar.repo_name = r.name
                WHERE r.organization = %s
            ''', (organization,))
        else:
            cur.execute('''
                SELECT detected_kus, author
                FROM analysis_results
            ''')

        rows = cur.fetchall()

        detected_kus_list = []
        for row in rows:
            detected_kus = json.loads(json.dumps(row[0]))
            author = row[1]
            detected_kus_list.append({"kus": detected_kus, "author": author})

        cur.close()
        return detected_kus_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        conn.close()


def save_analysis_to_db(repo_name, file_data):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        detected_kus_serialized = json.dumps(file_data["detected_kus"], default=str)
        timestamp_serialized = file_data["timestamp"].isoformat() if isinstance(file_data["timestamp"], datetime) else file_data["timestamp"]

        cur.execute('''
            INSERT INTO analysis_results (repo_name, filename, author, timestamp, sha, detected_kus, elapsed_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (
            repo_name,
            file_data["filename"],
            file_data["author"],
            timestamp_serialized,
            file_data["sha"],
            detected_kus_serialized,
            file_data["elapsed_time"]
        ))

        conn.commit()
        cur.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


def get_analysis_from_db(repo_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
            SELECT filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            WHERE repo_name = %s
        ''', (repo_name,))
        rows = cur.fetchall()

        analysis_data = []

        for row in rows:
            filename, author, timestamp, sha, detected_kus, elapsed_time = row

            if isinstance(detected_kus, str):
                detected_kus_deserialized = json.loads(detected_kus)
            else:
                detected_kus_deserialized = detected_kus

            timestamp_deserialized = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

            analysis_data.append({
                "filename": filename,
                "author": author,
                "timestamp": timestamp_deserialized.isoformat() if timestamp_deserialized else None,
                "sha": sha,
                "detected_kus": detected_kus_deserialized,
                "elapsed_time": elapsed_time
            })

        cur.close()

        return analysis_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        conn.close()


def get_allanalysis_from_db(organization=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        base_query = '''
            SELECT ar.filename, ar.author, ar.timestamp, ar.sha, ar.detected_kus, ar.elapsed_time
            FROM analysis_results ar
            JOIN repositories r ON ar.repo_name = r.name
        '''

        params = []
        if organization:
            base_query += " WHERE r.organization = %s"
            params.append(organization)

        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()

        analysis_data = []

        for row in rows:
            filename, author, timestamp, sha, detected_kus, elapsed_time = row

            if isinstance(detected_kus, str):
                detected_kus_deserialized = json.loads(detected_kus)
            else:
                detected_kus_deserialized = detected_kus

            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

            analysis_data.append({
                "filename": filename,
                "author": author,
                "timestamp": timestamp_str,
                "sha": sha,
                "detected_kus": detected_kus_deserialized,
                "elapsed_time": elapsed_time
            })

        cur.close()

        return analysis_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        conn.close()

def get_commits_timestamps_from_db(repo_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
            SELECT DISTINCT timestamp
            FROM analysis_results
            WHERE repo_name = %s
            ORDER BY timestamp ASC
        ''', (repo_name,))

        rows = cur.fetchall()
        cur.close()

        timestamps = [row[0].isoformat() for row in rows]

        return timestamps
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        conn.close()

def get_analysis_withsha_db(sha):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
            SELECT filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            WHERE sha = %s
        ''', (sha,))
        rows = cur.fetchall()

        analysis_data = []

        for row in rows:
            filename, author, timestamp, sha, detected_kus, elapsed_time = row

            if isinstance(detected_kus, str):
                detected_kus_deserialized = json.loads(detected_kus)
            else:
                detected_kus_deserialized = detected_kus

            timestamp_deserialized = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

            analysis_data.append({
                "filename": filename,
                "author": author,
                "timestamp": timestamp_deserialized.isoformat() if timestamp_deserialized else None,
                "sha": sha,
                "detected_kus": detected_kus_deserialized,
                "elapsed_time": elapsed_time
            })

        cur.close()

        return analysis_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        conn.close()

def update_analysis_status(repo_name, status, start_time=None, end_time=None, progress=None, error_message=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            UPDATE repositories
            SET analysis_status = %s,
                analysis_start_time = %s,
                analysis_end_time = %s,
                analysis_progress = %s,
                analysis_error_message = %s
            WHERE name = %s
        ''', (status, start_time, end_time, progress, error_message, repo_name))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"An error occurred updating analysis status: {e}")
    finally:
        conn.close()

def get_analysis_status(repo_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT analysis_status, analysis_start_time, analysis_end_time, analysis_progress, analysis_error_message
            FROM repositories
            WHERE name = %s
        ''', (repo_name,))
        result = cur.fetchone()
        cur.close()
        if result:
            status, start_time, end_time, progress, error_message = result
            return {
                "status": status,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "progress": progress,
                "error_message": error_message
            }
        else:
            return None
    except Exception as e:
        print(f"An error occurred getting analysis status: {e}")
        return None


def analyze_repository_background(repo_name, files):
    analysis_results = []
    total_files = len(files)
    analyzed_files_count = 0
    start_time = datetime.datetime.now()

    logging.info(f"Starting analysis for repository: {repo_name}")
    update_analysis_status(repo_name, 'in-progress', start_time=start_time, progress=0)

    for file in files.values():
        try:
            logging.debug(f"Analyzing file: {file.filename}")
            file_start_time = time.time()
            results = codebert_sliding_window([file], 35, 35, 1, 25, model)
            file_end_time = time.time()
            elapsed_time = file_end_time - file_start_time

            if isinstance(file.timestamp, datetime.datetime):
                timestmp = file.timestamp.isoformat()
            else:
                timestmp = file.timestamp

            file_data = {
                "filename": file.filename,
                "author": file.author,
                "timestamp": timestmp,
                "sha": file.sha,
                "detected_kus": file.ku_results,
                "elapsed_time": elapsed_time
            }
            analysis_results.append(file_data)
            analyzed_files_count += 1
            logging.info(f"Successfully analyzed file {analyzed_files_count}/{total_files}: {file.filename}")

            save_analysis_to_db(repo_name, file_data)

            progress = int((analyzed_files_count / total_files) * 100)
            update_analysis_status(repo_name, 'in-progress', start_time=start_time, progress=progress)

            print(f"Yielding: {json.dumps({'progress': progress, 'file_data': file_data})}")
            yield f"data: {json.dumps({'progress': progress, 'file_data': file_data})}\n\n"

        except Exception as e:
            logging.exception(
                f"Error analyzing file: {file.filename}. Total analyzed before error: {analyzed_files_count}.")
            update_analysis_status(repo_name, 'error', start_time=start_time, end_time=datetime.datetime.now(),
                                   error_message=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

    end_time = datetime.datetime.now()
    logging.info(f"Analysis completed for repository: {repo_name}. Total files analyzed: {len(analysis_results)}")
    update_analysis_status(repo_name, 'completed', start_time=start_time, end_time=end_time, progress=100)
    yield f"data: {json.dumps({'progress': 100, 'message': 'Analysis completed'})}\n\n"

def get_ku_counts_from_db(organization=None):
    base_query = """
        SELECT
            ku.key AS ku_id,
            COUNT(*) AS ku_count
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE
            ku.value = '1'
    """
    params = []
    if organization:
        base_query += " AND r.organization = %s"
        params.append(organization)

    base_query += """
        GROUP BY
            ku_id
        ORDER BY
            ku_count DESC;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()
        ku_counts = [{"ku_id": row[0], "count": int(row[1])} for row in rows]
        return ku_counts
    except Exception as e:
        print(f"An error occurred while getting KU counts: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

def get_organization_project_counts(organization=None):
    base_query = """
        SELECT
            organization,
            COUNT(*) AS project_count
        FROM
            repositories
        WHERE
            organization IS NOT NULL AND organization != ''
    """
    params = []
    if organization:
        base_query += " AND organization = %s"
        params.append(organization)

    base_query += """
        GROUP BY
            organization
        ORDER BY
            project_count DESC;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()

        org_counts = [{"organization": row[0], "count": row[1]} for row in rows]
        return org_counts

    except Exception as e:
        print(f"An error occurred while getting organization counts: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

def get_ku_counts_by_organization(organization=None):
    base_query = """
        SELECT
            r.organization,
            ku.key AS ku_id,
            COUNT(*) AS ku_count
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE
            r.organization IS NOT NULL AND r.organization != '' AND ku.value = '1'
    """
    params = []
    if organization:
        base_query += " AND r.organization = %s"
        params.append(organization)

    base_query += """
        GROUP BY
            r.organization, ku_id
        ORDER BY
            r.organization, ku_count DESC;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()

        organizations_data = {}
        for row in rows:
            org_name, ku_id, ku_count = row
            if org_name not in organizations_data:
                organizations_data[org_name] = {
                    "organization": org_name,
                    "ku_counts": []
                }
            organizations_data[org_name]["ku_counts"].append({
                "ku_id": ku_id,
                "count": ku_count
            })
        return list(organizations_data.values())

    except Exception as e:
        print(f"An error occurred while getting KU counts by organization: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

def get_monthly_analysis_counts_by_org(organization=None):
    base_query = """
        SELECT
            r.organization,
            DATE_TRUNC('month', ar.timestamp)::date AS analysis_month,
            COUNT(ar.id) AS analysis_count
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name
        WHERE
            r.organization IS NOT NULL AND r.organization != ''
    """
    params = []
    if organization:
        base_query += " AND r.organization = %s"
        params.append(organization)

    base_query += """
        GROUP BY
            r.organization, analysis_month
        ORDER BY
            r.organization, analysis_month;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()

        organizations_data = {}
        for row in rows:
            org_name, month_date, analysis_count = row
            if org_name not in organizations_data:
                organizations_data[org_name] = {
                    "organization": org_name,
                    "monthly_counts": []
                }
            organizations_data[org_name]["monthly_counts"].append({
                "month": month_date.strftime('%Y-%m'),
                "count": analysis_count
            })
        return list(organizations_data.values())

    except Exception as e:
        print(f"An error occurred while getting monthly analysis counts by org: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

def get_ku_counts_per_repository(organization=None):
    base_query = """
        SELECT
            ar.repo_name,
            ku.key AS ku_id,
            COUNT(*) as ku_count
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE
            ku.value = '1'
    """
    params = []
    if organization:
        base_query += " AND r.organization = %s"
        params.append(organization)

    base_query += """
        GROUP BY
            ar.repo_name, ku_id;
    """

    repos_with_ku_counts = defaultdict(dict)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()

        for repo_name, ku_id, ku_count in rows:
            repos_with_ku_counts[repo_name][ku_id] = ku_count

        return dict(repos_with_ku_counts)

    except Exception as e:
        logging.error(f"An error occurred while getting KU counts per repository: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


def cluster_repositories_by_kus(num_clusters: int, organization=None):
    try:
        repos_data = get_ku_counts_per_repository(organization=organization)
        if not repos_data or len(repos_data) < num_clusters:
            raise ValueError("Not enough repositories with detected KUs to form the requested number of clusters.")

        repo_names = list(repos_data.keys())

        all_kus = sorted(list(set.union(*(set(d.keys()) for d in repos_data.values()))))
        df = pd.DataFrame(0, index=repo_names, columns=all_kus, dtype=np.int32)
        for repo, ku_counts in repos_data.items():
            for ku, count in ku_counts.items():
                df.loc[repo, ku] = count

        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(df)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        pca = PCA(n_components=2, random_state=42)
        coordinates_2d = pca.fit_transform(tfidf_matrix.toarray())

        results = []
        for repo_name, cluster_label, coords in zip(repo_names, cluster_labels, coordinates_2d):
            results.append({
                "repo_name": repo_name,
                "cluster": int(cluster_label),
                "coordinates": {
                    "x": float(coords[0]),
                    "y": float(coords[1])
                }
            })

        return results

    except ValueError as ve:
        logging.warning(f"Clustering validation error: {ve}")
        raise ve
    except Exception as e:
        logging.exception(f"An unexpected error occurred during K-Means clustering: {e}")
        return None

def get_entire_analysis_table():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('''
            SELECT id, repo_name, filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            ORDER BY repo_name, timestamp;
        ''')
        rows = cur.fetchall()
        cur.close()

        all_results = []

        for row in rows:
            (id, repo_name, filename, author, timestamp, sha, detected_kus, elapsed_time) = row

            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

            all_results.append({
                "id": id,
                "repo_name": repo_name,
                "filename": filename,
                "author": author,
                "timestamp": timestamp_str,
                "sha": sha,
                "detected_kus": detected_kus,
                "elapsed_time": elapsed_time
            })

        return all_results

    except Exception as e:
        print(f"An error occurred while fetching the entire analysis_results table: {e}")
        return None

    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


def get_analysis_results(start_date_str=None, end_date_str=None, organization=None):
    try:
        base_query = '''
            SELECT
                ar.id,
                ar.repo_name,
                r.organization,
                ar.filename,
                ar.author,
                ar.timestamp,
                ar.sha,
                ar.detected_kus,
                ar.elapsed_time
            FROM
                analysis_results ar
            JOIN
                repositories r ON ar.repo_name = r.name
        '''

        conditions = []
        params = []

        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m')
            conditions.append("ar.timestamp >= %s")
            params.append(start_date)

        if end_date_str:
            end_date_exclusive = datetime.strptime(end_date_str, '%Y-%m') + relativedelta(months=1)
            conditions.append("ar.timestamp < %s")
            params.append(end_date_exclusive)

        if organization:
            conditions.append("r.organization = %s")
            params.append(organization)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY ar.repo_name, ar.timestamp;"

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(base_query, tuple(params))

        rows = cur.fetchall()
        cur.close()

        all_results = []
        for row in rows:
            (id, repo_name, org, filename, author, timestamp, sha, detected_kus, elapsed_time) = row

            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

            all_results.append({
                "id": id,
                "repo_name": repo_name,
                "organization": org,
                "filename": filename,
                "author": author,
                "timestamp": timestamp_str,
                "sha": sha,
                "detected_kus": detected_kus,
                "elapsed_time": elapsed_time
            })

        return all_results

    except Exception as e:
        print(f"An error occurred while fetching analysis_results: {e}")
        return None

    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# ΒΟΗΘΗΤΙΚΗ ΣΥΝΑΡΤΗΣΗ: Υπολογισμός πιθανότητας απώλειας με το 20% threshold
# ─────────────────────────────────────────────────────────────────────────────

def _probability_of_loss(emps: int, p_individual_leave: float = 0.1, threshold_pct: float = 0.20) -> float:
    """
    Υπολογίζει την πιθανότητα να φύγει τουλάχιστον το `threshold_pct` (π.χ. 20%)
    των employees που κατέχουν ένα KU.

    Μοντελοποίηση:
    - Κάθε employee φεύγει ανεξάρτητα με πιθανότητα `p_individual_leave`.
    - X ~ Binomial(n=emps, p=p_individual_leave)
    - Υπολογίζουμε P(X >= k), όπου k = ceil(threshold_pct * emps).
    - Ειδική περίπτωση: αν emps == 0, δεν υπάρχει κίνδυνος → 0.0

    Args:
        emps: Αριθμός employees που κατέχουν το KU.
        p_individual_leave: Πιθανότητα ένας employee να φύγει (default 10%).
        threshold_pct: Το κατώφλι ως ποσοστό (default 20%).

    Returns:
        P(X >= ceil(threshold_pct * emps))
    """
    if emps == 0:
        return 0.0

    # Ελάχιστος αριθμός αναχωρήσεων που ορίζει «κίνδυνο»
    k = max(1, math.ceil(threshold_pct * emps))

    # P(X >= k) = 1 - P(X <= k-1)  [binomial survival function]
    p_loss = 1.0 - binom.cdf(k - 1, n=emps, p=p_individual_leave)

    return p_loss


def calculate_risks(organization=None):
    """
    Calculates KU Risk and Employee Risk.

    Νέα λογική πιθανότητας απώλειας (p_L):
    Αντί να απαιτείται να φύγουν *όλοι* οι employees που κατέχουν ένα KU,
    θεωρούμε ότι αρκεί να φύγει το 20% από αυτούς.
    Χρησιμοποιούμε Binomial Distribution:
        p_L = P(X >= ceil(0.20 * emps))
        όπου X ~ Bin(emps, p=0.10)

    Can be filtered by a specific organization.
    """
    try:
        # --- Βήμα 1: Συλλογή και προετοιμασία δεδομένων ---
        conn = get_db_connection()
        cur = conn.cursor()

        base_query_select = '''
            SELECT ar.filename, ar.author, ar.detected_kus
            FROM analysis_results ar
            JOIN repositories r ON ar.repo_name = r.name
        '''
        base_query_count = '''
            SELECT COUNT(DISTINCT ar.filename)
            FROM analysis_results ar
            JOIN repositories r ON ar.repo_name = r.name
        '''

        params = []
        where_clause = ""

        if organization:
            where_clause = " WHERE r.organization = %s"
            params.append(organization)

        cur.execute(base_query_select + where_clause, tuple(params))
        analysis_data = cur.fetchall()

        cur.execute(base_query_count + where_clause, tuple(params))
        total_files_result = cur.fetchone()
        total_files = total_files_result[0] if total_files_result else 0

        cur.close()
        conn.close()

        if not analysis_data or total_files == 0:
            return {
                "ku_risk": {},
                "employee_risk": {}
            }

        # --- Βήμα 2: Δόμηση πληροφορίας (Aggregation) ---
        knowledge_units = defaultdict(lambda: {'freq': 0, 'authors': set()})
        author_ku_map = defaultdict(set)

        for filename, author, detected_kus in analysis_data:
            for ku, is_present in detected_kus.items():
                if int(is_present) == 1:
                    knowledge_units[ku]['freq'] += 1
                    knowledge_units[ku]['authors'].add(author)
                    author_ku_map[author].add(ku)

        # --- Βήμα 3: Υπολογισμός KU Risk ---
        # ΑΛΛΑΓΗ: p_L = P(≥20% των employees φεύγουν) αντί P(όλοι φεύγουν)
        ku_risk_results = {}

        for ku, data in knowledge_units.items():
            emps = len(data['authors'])
            freq = data['freq']

            # Νέος υπολογισμός πιθανότητας απώλειας
            p_L = _probability_of_loss(emps)

            impact = freq / total_files
            ku_risk = p_L * impact

            ku_risk_results[ku] = {
                "ku_risk": ku_risk,
                "probability_of_loss": p_L,
                "impact": impact,
                "employee_count": emps,
                "file_frequency": freq,
                # Επιπλέον πληροφορία: πόσοι employees αρκεί να φύγουν
                "employees_at_risk_threshold": max(1, math.ceil(0.20 * emps))
            }

        # --- Βήμα 4: Υπολογισμός Employee Risk ---
        # ΑΛΛΑΓΗ: αντί να αφαιρούμε 1 employee, αφαιρούμε ceil(20% * emps_before).
        # Αυτό μοντελοποιεί τον κίνδυνο αν φύγει ο employee ΜΑΖΙ με το 20% της ομάδας του.
        employee_risk_results = {}

        for author, kus in author_ku_map.items():
            total_delta_risk = 0.0
            total_before_risk = 0.0

            for ku in kus:
                ku_data = knowledge_units[ku]
                freq = ku_data['freq']
                emps_before = len(ku_data['authors'])
                impact = freq / total_files

                # Risk πριν την αναχώρηση (με το νέο μοντέλο)
                p_L_before = _probability_of_loss(emps_before)
                risk_before = p_L_before * impact

                # Risk μετά: αφαιρούμε ceil(20% * emps_before) employees
                # (minimum 1, για να μην μένουμε με τον ίδιο αριθμό)
                employees_leaving = max(1, math.ceil(0.20 * emps_before))
                emps_after = max(0, emps_before - employees_leaving)

                p_L_after = _probability_of_loss(emps_after)
                risk_after = p_L_after * impact

                delta_risk = risk_after - risk_before
                total_delta_risk += delta_risk
                total_before_risk += risk_before

            absolute_risk = total_delta_risk
            relative_risk = absolute_risk / total_before_risk if total_before_risk > 0 else 0.0

            employee_risk_results[author] = {
                "absolute_employee_risk": absolute_risk,
                "relative_employee_risk": relative_risk,
                "ku_count": len(kus)
            }

        return {
            "ku_risk": ku_risk_results,
            "employee_risk": employee_risk_results
        }

    except Exception as e:
        logging.exception("An error occurred during risk calculation")
        return {"error": str(e)}

def get_ku_counts_by_developer(developer_name, organization=None):
    all_kus = {f"K{i}": 0 for i in range(1, 28)}

    base_query = """
        SELECT
            ku.key AS ku_name,
            COUNT(DISTINCT ar.filename) AS file_count
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE
            ar.author = %s
            AND ku.value = '1'
    """
    params = [developer_name]

    if organization:
        base_query += " AND r.organization = %s"
        params.append(organization)

    base_query += """
        GROUP BY
            ku_name;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()

        for ku_name, file_count in rows:
            if ku_name in all_kus:
                all_kus[ku_name] = 1

        return all_kus

    except Exception as e:
        logging.error(f"An error occurred while getting KU counts for developer {developer_name}: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


def get_all_developer_ku_vectors(start_date_str=None, end_date_str=None, organization=None):
    base_query = """
        SELECT
            ar.author,
            r.name,
            r.organization,
            ARRAY_AGG(DISTINCT ku.key ORDER BY ku.key) as present_kus
        FROM
            analysis_results ar
        JOIN
            repositories r ON ar.repo_name = r.name,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
    """

    conditions = ["ku.value = '1'"]
    params = []

    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m')
            conditions.append("ar.timestamp >= %s")
            params.append(start_date)
        except ValueError:
            logging.warning(f"Invalid start_date format provided: {start_date_str}")
            pass

    if end_date_str:
        try:
            end_date_exclusive = datetime.strptime(end_date_str, '%Y-%m') + relativedelta(months=1)
            conditions.append("ar.timestamp < %s")
            params.append(end_date_exclusive)
        except ValueError:
            logging.warning(f"Invalid end_date format provided: {end_date_str}")
            pass

    if organization:
        conditions.append("r.organization = %s")
        params.append(organization)

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += """
        GROUP BY
            ar.author, r.name, r.organization
        ORDER BY
            ar.author, r.name;
    """

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()
        cur.close()

        results = []
        all_kus_template = {f"K{i}": 0 for i in range(1, 28)}

        for author, repo_name, organization_val, present_kus_list in rows:
            ku_vector = all_kus_template.copy()

            if present_kus_list:
                for ku in present_kus_list:
                    if ku in ku_vector:
                        ku_vector[ku] = 1

            results.append({
                "developer_name": author,
                "organization": organization_val,
                "repo_name": repo_name,
                "ku_vector": ku_vector
            })

        return results

    except Exception as e:
        logging.error(f"An error occurred while getting KU vectors for all developers: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

def get_ku_skills_by_organization(organization_name):
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
        GROUP BY
            ku_name
        ORDER BY
            ku_name;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query, (organization_name,))
        rows = cur.fetchall()
        cur.close()

        skills_data = [
            {"ku_name": row[0], "total_files": int(row[1]), "total_authors": int(row[2])}
            for row in rows
        ]
        return skills_data

    except Exception as e:
        logging.error(f"An error occurred while getting KU skills for organization {organization_name}: {e}")
        return None
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()