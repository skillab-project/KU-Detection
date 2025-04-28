import json
import os
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
import time
import logging
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

        # Check if tables already exists
        cur.execute(table_check_query)
        (table_exists,) = cur.fetchone()
        if table_exists:
            print("Tables already exists. Skipping table creation.")
            return

        # Create tables
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn is not None:
            conn.close()

def save_repo_to_db(name, url=None, description=None, comments=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO repositories (name, url, description, comments)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET url = EXCLUDED.url,
                description = EXCLUDED.description,
                comments = EXCLUDED.comments,
                updated_at = CURRENT_TIMESTAMP
        ''', (name, url, description, comments))
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

        # Διαγραφή από τον πίνακα analysis_results
        cur.execute('''
            DELETE FROM analysis_results WHERE repo_name = %s
        ''', (repo_name,))

        # Διαγραφή από τον πίνακα commits
        cur.execute('''
            DELETE FROM commits WHERE repo_name = %s
        ''', (repo_name,))

        # Διαγραφή από τον πίνακα repositories
        cur.execute('''
            DELETE FROM repositories WHERE name = %s
        ''', (repo_name,))

        conn.commit()
        cur.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e  # Ξαναπετάμε το σφάλμα για να το διαχειριστεί η διαδρομή Flask
    finally:
        conn.close()

def get_all_repos_from_db():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT name, url, description, comments, created_at, updated_at, analysis_status, analysis_start_time, analysis_end_time, analysis_progress, analysis_error_message
                    FROM repositories
                    WHERE url LIKE 'https://github.com/apache/%';
                ''')

                rows = cur.fetchall()

                repos = []
                for row in rows:
                    repo = {
                        "name": row[0],
                        "url": row[1],
                        "description": row[2],
                        "comments": row[3],
                        "created_at": row[4].isoformat() if row[4] else None,
                        "updated_at": row[5].isoformat() if row[5] else None,
                        "analysis_status": row[6],
                        "analysis_start_time": row[7].isoformat() if row[7] else None,
                        "analysis_end_time": row[8].isoformat() if row[8] else None,
                        "analysis_progress": row[9],
                        "analysis_error_message": row[10]
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
            JOIN repositories r ON ar.repo_name = r.name  
            WHERE r.url LIKE 'https://github.com/apache/%';
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