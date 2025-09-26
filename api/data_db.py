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

# --- ΔΙΟΡΘΩΣΗ: Η συνάρτηση πλέον δέχεται τον οργανισμό ως παράμετρο ---
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

def get_all_repos_from_db():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT name, url, organization, description, comments, created_at, updated_at, 
                           analysis_status, analysis_start_time, analysis_end_time, 
                           analysis_progress, analysis_error_message
                    FROM repositories;
                ''')
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

        # Convert the list of tuples to a list of dictionaries
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


def getdetected_kus():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

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

        # Εκτέλεση του query για ανάκτηση των δεδομένων
        cur.execute('''
            SELECT filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            WHERE repo_name = %s
        ''', (repo_name,))
        rows = cur.fetchall()

        # Λίστα για αποθήκευση των αποτελεσμάτων
        analysis_data = []

        # Επεξεργασία των δεδομένων
        for row in rows:
            filename, author, timestamp, sha, detected_kus, elapsed_time = row

            # Αν η στήλη detected_kus είναι JSON string, κάνουμε deserialization
            if isinstance(detected_kus, str):
                detected_kus_deserialized = json.loads(detected_kus)
            else:
                detected_kus_deserialized = detected_kus  # Είναι ήδη αντικείμενο Python

            # Μετατροπή του timestamp αν χρειάζεται
            timestamp_deserialized = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

            # Προσθήκη του λεξικού στη λίστα
            analysis_data.append({
                "filename": filename,
                "author": author,
                "timestamp": timestamp_deserialized.isoformat() if timestamp_deserialized else None,
                "sha": sha,
                "detected_kus": detected_kus_deserialized,
                "elapsed_time": elapsed_time
            })

        cur.close()

        # Επιστροφή της λίστας αποτελεσμάτων (analysis_data) ως απάντηση
        return analysis_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        conn.close()


def get_allanalysis_from_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Εκτέλεση του query για ανάκτηση όλων των δεδομένων από τον πίνακα analysis_results
        cur.execute('''
            SELECT ar.filename, ar.author, ar.timestamp, ar.sha, ar.detected_kus, ar.elapsed_time
            FROM analysis_results ar
            JOIN repositories r ON ar.repo_name = r.name;
        ''')
        rows = cur.fetchall()

        # Λίστα για αποθήκευση των αποτελεσμάτων
        analysis_data = []

        # Επεξεργασία των δεδομένων
        for row in rows:
            filename, author, timestamp, sha, detected_kus, elapsed_time = row

            # Αν η στήλη detected_kus είναι JSON string, κάνουμε deserialization
            if isinstance(detected_kus, str):
                detected_kus_deserialized = json.loads(detected_kus)
            else:
                detected_kus_deserialized = detected_kus  # Είναι ήδη αντικείμενο Python

            # Μετατροπή του timestamp αν χρειάζεται σε string ISO format
            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

            # Προσθήκη του λεξικού στη λίστα
            analysis_data.append({
                "filename": filename,
                "author": author,
                "timestamp": timestamp_str,
                "sha": sha,
                "detected_kus": detected_kus_deserialized,
                "elapsed_time": elapsed_time
            })

        cur.close()

        # Επιστροφή της λίστας αποτελεσμάτων (analysis_data) ως απάντηση
        return analysis_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        conn.close()

def get_commits_timestamps_from_db(repo_name):
    try:
        conn = get_db_connection()  # Σύνδεση με τη βάση δεδομένων
        cur = conn.cursor()

        # Εκτέλεση του query για να πάρουμε μοναδικά timestamps από τα commits
        cur.execute('''
            SELECT DISTINCT timestamp
            FROM analysis_results
            WHERE repo_name = %s
            ORDER BY timestamp ASC
        ''', (repo_name,))

        rows = cur.fetchall()
        cur.close()

        # Επιστρέφουμε μια λίστα με τα timestamps
        timestamps = [row[0].isoformat() for row in rows]  # row[0] είναι το timestamp

        return timestamps
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        conn.close()  # Κλείνουμε τη σύνδεση

def get_analysis_withsha_db(sha):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Εκτέλεση του query για ανάκτηση των δεδομένων
        cur.execute('''
            SELECT filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            WHERE sha = %s
        ''', (sha,))
        rows = cur.fetchall()

        # Λίστα για αποθήκευση των αποτελεσμάτων
        analysis_data = []

        # Επεξεργασία των δεδομένων
        for row in rows:
            filename, author, timestamp, sha, detected_kus, elapsed_time = row

            # Αν η στήλη detected_kus είναι JSON string, κάνουμε deserialization
            if isinstance(detected_kus, str):
                detected_kus_deserialized = json.loads(detected_kus)
            else:
                detected_kus_deserialized = detected_kus  # Είναι ήδη αντικείμενο Python

            # Μετατροπή του timestamp αν χρειάζεται
            timestamp_deserialized = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

            # Προσθήκη του λεξικού στη λίστα
            analysis_data.append({
                "filename": filename,
                "author": author,
                "timestamp": timestamp_deserialized.isoformat() if timestamp_deserialized else None,
                "sha": sha,
                "detected_kus": detected_kus_deserialized,
                "elapsed_time": elapsed_time
            })

        cur.close()

        # Επιστροφή της λίστας αποτελεσμάτων (analysis_data) ως απάντηση
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

            # Αποθήκευση αποτελέσματος αμέσως μετά την ανάλυση
            save_analysis_to_db(repo_name, file_data)


            # Update progress
            progress = int((analyzed_files_count / total_files) * 100)
            update_analysis_status(repo_name, 'in-progress', start_time=start_time, progress=progress)

            # Send progress and data update to frontend every file
            print(f"Yielding: {json.dumps({'progress': progress, 'file_data': file_data})}")  # Debugging line
            yield f"data: {json.dumps({'progress': progress, 'file_data': file_data})}\n\n"

        except Exception as e:
            logging.exception(
                f"Error analyzing file: {file.filename}. Total analyzed before error: {analyzed_files_count}.")
            update_analysis_status(repo_name, 'error', start_time=start_time, end_time=datetime.datetime.now(),
                                   error_message=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

    # Save results to the database
    #save_analysis_to_db(repo_name, analysis_results)
    end_time = datetime.datetime.now()
    logging.info(f"Analysis completed for repository: {repo_name}. Total files analyzed: {len(analysis_results)}")
    update_analysis_status(repo_name, 'completed', start_time=start_time, end_time=end_time, progress=100)
    yield f"data: {json.dumps({'progress': 100, 'message': 'Analysis completed'})}\n\n"

def get_ku_counts_from_db():
    sql_query = """
        SELECT
            ku.key AS ku_id,
            COUNT(*) AS ku_count
        FROM
            analysis_results,
            LATERAL jsonb_each_text(detected_kus) AS ku
        WHERE
            ku.value = '1'
        GROUP BY
            ku_id
        ORDER BY
            ku_count DESC;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
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

def get_organization_project_counts():
    """
    Ανακτά από τη βάση δεδομένων το πλήθος των projects ανά οργανισμό,
    βασιζόμενο στην στήλη 'organization'.
    """
    sql_query = """
        SELECT
            organization,
            COUNT(*) AS project_count
        FROM
            repositories
        WHERE
            organization IS NOT NULL AND organization != ''
        GROUP BY
            organization
        ORDER BY
            project_count DESC;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
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

def get_ku_counts_by_organization():
    """
    Επιστρέφει τα στατιστικά των KUs ομαδοποιημένα ανά οργανισμό.
    """
    sql_query = """
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
        GROUP BY
            r.organization, ku_id
        ORDER BY
            r.organization, ku_count DESC;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
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

def get_monthly_analysis_counts_by_org():
    """
    Επιστρέφει το πλήθος των αναλύσεων ανά μήνα, ομαδοποιημένο ανά οργανισμό.
    """
    sql_query = """
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
        GROUP BY
            r.organization, analysis_month
        ORDER BY
            r.organization, analysis_month;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
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

def get_ku_counts_per_repository():
    """
    Ανακτά το πλήθος των commits για κάθε KU, ομαδοποιημένο ανά repository.
    Επιστρέφει ένα λεξικό της μορφής:
    { 'repo_name_1': {'KU1': 10, 'KU5': 3}, 'repo_name_2': {'KU1': 5, 'KU8': 12} }
    """
    sql_query = """
        SELECT
            repo_name,
            ku.key AS ku_id,
            COUNT(*) as ku_count
        FROM
            analysis_results,
            LATERAL jsonb_each_text(detected_kus) AS ku
        WHERE
            ku.value = '1'
        GROUP BY
            repo_name, ku_id;
    """
    repos_with_ku_counts = defaultdict(dict)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
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


def cluster_repositories_by_kus(num_clusters: int):
    """
    Ομαδοποιεί τα repositories χρησιμοποιώντας K-Means πάνω σε TF-IDF-weighted KU counts
    και μειώνει τις διαστάσεις με PCA για 2D οπτικοποίηση.
    """
    try:
        # 1. Ανάκτηση του πλήθους των KUs για κάθε repository
        repos_data = get_ku_counts_per_repository() # <--- ΣΩΣΤΗ ΚΛΗΣΗ
        if not repos_data or len(repos_data) < num_clusters:
            raise ValueError("Not enough repositories with detected KUs to form the requested number of clusters.")

        repo_names = list(repos_data.keys())

        # 2. Δημιουργία του πίνακα χαρακτηριστικών (με ακατέργαστα counts)
        all_kus = sorted(list(set.union(*(set(d.keys()) for d in repos_data.values()))))
        df = pd.DataFrame(0, index=repo_names, columns=all_kus, dtype=np.int32)
        for repo, ku_counts in repos_data.items():
            for ku, count in ku_counts.items():
                df.loc[repo, ku] = count

        # 3. Μετατροπή των ακατέργαστων counts σε σταθμισμένα scores με TF-IDF
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(df)

        # 4. Εκτέλεση του K-Means πάνω στα σταθμισμένα δεδομένα
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # 5. Μείωση διαστάσεων με PCA πάνω στα σταθμισμένα δεδομένα
        pca = PCA(n_components=2, random_state=42)
        coordinates_2d = pca.fit_transform(tfidf_matrix.toarray())

        # 6. Σύνθεση της τελικής απάντησης
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
    """
    Ανακτά ΟΛΕΣ τις εγγραφές και τις στήλες από τον πίνακα analysis_results.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Απλό query για να πάρουμε όλα τα δεδομένα από τον πίνακα
        cur.execute('''
            SELECT id, repo_name, filename, author, timestamp, sha, detected_kus, elapsed_time
            FROM analysis_results
            ORDER BY repo_name, timestamp;
        ''')
        rows = cur.fetchall()
        cur.close()

        # Λίστα για την αποθήκευση των αποτελεσμάτων
        all_results = []

        # Επεξεργασία κάθε γραμμής
        for row in rows:
            # Αποσυμπίεση των πεδίων από τη γραμμή
            (id, repo_name, filename, author, timestamp, sha, detected_kus, elapsed_time) = row

            # Το detected_kus είναι τύπου JSONB, οπότε η psycopg2 το μετατρέπει ήδη σε dict.
            # Το timestamp είναι datetime object, το μετατρέπουμε σε string.
            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

            # Προσθήκη του λεξικού στη λίστα
            all_results.append({
                "id": id,
                "repo_name": repo_name,
                "filename": filename,
                "author": author,
                "timestamp": timestamp_str,
                "sha": sha,
                "detected_kus": detected_kus, # Είναι ήδη dict
                "elapsed_time": elapsed_time
            })

        return all_results

    except Exception as e:
        print(f"An error occurred while fetching the entire analysis_results table: {e}")
        return None  # Επιστρέφουμε None σε περίπτωση σφάλματος

    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


def get_analysis_results(start_date_str=None, end_date_str=None):
    """
    Ανακτά εγγραφές από τον πίνακα analysis_results με προαιρετικό φιλτράρισμα ημερομηνίας.
    - Αν δεν δοθεί καμία ημερομηνία, επιστρέφει όλες τις εγγραφές.
    - Αν δοθεί μόνο start_date, επιστρέφει εγγραφές από εκείνη την ημερομηνία και μετά.
    - Αν δοθεί μόνο end_date, επιστρέφει εγγραφές μέχρι εκείνη την ημερομηνία.
    - Αν δοθούν και οι δύο, επιστρέφει εγγραφές εντός του εύρους.
    Επιστρέφει επίσης το πεδίο 'organization' από τον πίνακα 'repositories'.
    """
    try:
        # --- ΑΛΛΑΓΗ 1: Ενημέρωση του SQL Query για να περιλαμβάνει JOIN και το πεδίο organization ---
        base_query = '''
            SELECT
                ar.id,
                ar.repo_name,
                r.organization, -- ΠΡΟΣΘΗΚΗ: Το πεδίο που θέλουμε από τον πίνακα repositories
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
            # Χρησιμοποιούμε το alias 'ar' για σαφήνεια
            conditions.append("ar.timestamp >= %s")
            params.append(start_date)

        if end_date_str:
            end_date_exclusive = datetime.strptime(end_date_str, '%Y-%m') + relativedelta(months=1)
            # Χρησιμοποιούμε το alias 'ar' για σαφήνεια
            conditions.append("ar.timestamp < %s")
            params.append(end_date_exclusive)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        # Χρησιμοποιούμε aliases και εδώ για συνέπεια
        base_query += " ORDER BY ar.repo_name, ar.timestamp;"

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(base_query, tuple(params))

        rows = cur.fetchall()
        cur.close()

        all_results = []
        for row in rows:
            # --- ΑΛΛΑΓΗ 2: Ενημέρωση της αποσυμπίεσης του tuple για να περιλαμβάνει το organization ---
            (id, repo_name, organization, filename, author, timestamp, sha, detected_kus, elapsed_time) = row

            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

            # --- ΑΛΛΑΓΗ 3: Προσθήκη του organization στο τελικό λεξικό ---
            all_results.append({
                "id": id,
                "repo_name": repo_name,
                "organization": organization,  # <-- Η νέα πληροφορία
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
def calculate_risks():
    """
    Calculates KU Risk and Employee Risk based on the entire analysis dataset.
    This function implements the logic described in the provided documentation.
    """
    try:
        # --- Βήμα 1: Συλλογή και προετοιμασία δεδομένων ---
        conn = get_db_connection()
        cur = conn.cursor()

        # Ανάκτηση όλων των σχετικών δεδομένων: filename, author, detected_kus
        cur.execute('''
            SELECT filename, author, detected_kus
            FROM analysis_results
        ''')
        analysis_data = cur.fetchall()

        # Ανάκτηση του συνολικού αριθμού μοναδικών αρχείων στο έργο
        cur.execute('SELECT COUNT(DISTINCT filename) FROM analysis_results')
        total_files = cur.fetchone()[0]

        cur.close()
        conn.close()

        if not analysis_data or total_files == 0:
            return {"error": "No analysis data found to calculate risks."}

        # --- Βήμα 2: Δόμηση πληροφορίας (Aggregation) ---
        # knowledge_units: {'KU_name': {'freq': count, 'authors': {set of authors}}}
        # author_ku_map: {'author_name': {set of KUs}}
        knowledge_units = defaultdict(lambda: {'freq': 0, 'authors': set()})
        author_ku_map = defaultdict(set)

        for filename, author, detected_kus in analysis_data:
            for ku, is_present in detected_kus.items():
                if int(is_present) == 1:
                    knowledge_units[ku]['freq'] += 1
                    knowledge_units[ku]['authors'].add(author)
                    author_ku_map[author].add(ku)

        # --- Βήμα 3: Υπολογισμός KU Risk ---
        # Σταθερή πιθανότητα αποχώρησης ενός προγραμματιστή
        P_A = 0.1
        ku_risk_results = {}

        for ku, data in knowledge_units.items():
            emps = len(data['authors'])
            freq = data['freq']

            # P(L) = P_A ^ emps (Πιθανότητα απώλειας)
            p_L = P_A ** emps

            # Impact = freq / totalFiles (Αντίκτυπος)
            impact = freq / total_files

            # KU Risk = P(L) * Impact
            ku_risk = p_L * impact

            ku_risk_results[ku] = {
                "ku_risk": ku_risk,
                "probability_of_loss": p_L,
                "impact": impact,
                "employee_count": emps,
                "file_frequency": freq
            }

        # --- Βήμα 4: Υπολογισμός Employee Risk ---
        employee_risk_results = {}

        for author, kus in author_ku_map.items():
            total_delta_risk = 0.0  # (Σ ΔRisk_KUj) for Absolute Risk
            total_before_risk = 0.0  # (Σ Risk_KUj_before) for Relative Risk

            for ku in kus:
                # Ανάκτηση δεδομένων για το KU
                ku_data = knowledge_units[ku]
                freq = ku_data['freq']
                emps_before = len(ku_data['authors'])
                impact = freq / total_files

                # Υπολογισμός Risk_KUj_before
                p_L_before = P_A ** emps_before
                risk_before = p_L_before * impact

                # Υπολογισμός Risk_KUj_after
                emps_after = emps_before - 1
                if emps_after == 0:
                    # Ειδική περίπτωση: ο εργαζόμενος ήταν ο μοναδικός κάτοχος
                    p_L_after = 1.0
                else:
                    p_L_after = P_A ** emps_after

                risk_after = p_L_after * impact

                # Υπολογισμός ΔRisk_KUj (Μεταβολή στον κίνδυνο)
                delta_risk = risk_after - risk_before

                total_delta_risk += delta_risk
                total_before_risk += risk_before

            # Υπολογισμός Absolute & Relative Risk για τον εργαζόμενο
            absolute_risk = total_delta_risk

            if total_before_risk > 0:
                relative_risk = absolute_risk / total_before_risk
            else:
                relative_risk = 0.0  # Αν ο αρχικός κίνδυνος ήταν 0

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
def get_ku_counts_by_developer(developer_name):
    """
    Για έναν συγκεκριμένο προγραμματιστή, επιστρέφει ένα λεξικό με όλα τα KUs
    (K1 έως K27) και την τιμή τους να είναι το πλήθος των *μοναδικών αρχείων*
    στα οποία εντοπίστηκε το καθένα. Τα KUs που δεν βρέθηκαν έχουν τιμή 0.
    """
    # 1. Δημιουργούμε το τελικό λεξικό με όλα τα KUs αρχικοποιημένα στο 0.
    # Αυτό εγγυάται ότι η απάντηση θα έχει πάντα την ίδια δομή.
    all_kus = {f"K{i}": 0 for i in range(1, 28)}

    # 2. Το SQL query που θα "ξεδιπλώσει" το JSON και θα μετρήσει
    #    τα μοναδικά αρχεία για τα KUs που βρέθηκαν.
    sql_query = """
        SELECT
            ku.key AS ku_name,
            COUNT(DISTINCT ar.filename) AS file_count
        FROM
            analysis_results ar,
            LATERAL jsonb_each_text(ar.detected_kus) AS ku
        WHERE
            ar.author = %s
            AND ku.value = '1'
        GROUP BY
            ku_name;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Περνάμε το όνομα του developer ως παράμετρο για ασφάλεια (αποφυγή SQL Injection)
        cur.execute(sql_query, (developer_name,))
        rows = cur.fetchall()
        cur.close()

        # 3. Ενημερώνουμε το αρχικό λεξικό με τα αποτελέσματα από τη βάση.
        #    Μόνο τα KUs που βρέθηκαν θα ενημερωθούν, τα υπόλοιπα θα παραμείνουν 0.
        for ku_name, file_count in rows:
            if ku_name in all_kus:
                all_kus[ku_name] = int(file_count)

        # 4. Επιστρέφουμε το πλήρες λεξικό.
        return all_kus

    except Exception as e:
        logging.error(f"An error occurred while getting KU counts for developer {developer_name}: {e}")
        return None  # Επιστρέφουμε None σε περίπτωση σφάλματος
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()