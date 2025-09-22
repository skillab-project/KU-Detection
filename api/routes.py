import os
from dotenv import load_dotenv
import concurrent.futures
import time
import logging
from flask import Flask, request, jsonify
import json
import datetime

# Φόρτωσε τις μεταβλητές από το αρχείο .env
load_dotenv()

from flask_cors import CORS
from api.data_db import (
    save_commits_to_db,
    get_commits_from_db,
    save_analysis_to_db,
    save_repo_to_db,
    get_all_repos_from_db,
    get_analysis_from_db,
    delete_repo_from_db,
    getdetected_kus,
    get_commits_timestamps_from_db,
    update_analysis_status,
    get_analysis_status,
    get_allanalysis_from_db,
    get_ku_counts_from_db,
    get_organization_project_counts,
    get_ku_counts_by_organization,
    get_monthly_analysis_counts_by_org,
    cluster_repositories_by_kus,
    get_analysis_results,
    calculate_risks,
)
from core.git_operations import clone_repo, repo_exists, extract_contributions
from core.git_operations.repo import pull_repo, get_history_repo
from core.utils.code_files_loader import read_files_from_dict_list
from flask_swagger_ui import get_swaggerui_blueprint
from core.ml_operations.loader import load_codebert_model
from core.analysis.codebert_sliding_window import codebert_sliding_window
from config.settings import CLONED_REPO_BASE_PATH, CODEBERT_BASE_PATH

app = Flask(__name__)
CORS(app)

# Δημιουργία ενός global ThreadPoolExecutor ---
# Δημιουργούμε τον executor μία φορά όταν ξεκινάει η εφαρμογή.
# Αυτός θα διαχειρίζεται όλες τις background εργασίες μας.
try:
    MAX_WORKERS = int(os.getenv('ANALYSIS_WORKERS', 4))
except (ValueError, TypeError):
    MAX_WORKERS = 4
# Αυτός ο executor θα τρέχει τις κύριες αναλύσεις (μία ανά repository)
background_task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5) # 5 ταυτόχρονες αναλύσεις repo


# Load model
logging.info("Loading CodeBERT model...")
model = load_codebert_model(CODEBERT_BASE_PATH, 27)
logging.info("CodeBERT model loaded.")


# ΒΟΗΘΗΤΙΚΗ ΣΥΝΑΡΤΗΣΗ: Αυτή εκτελείται σε κάθε thread για ένα αρχείο
def analyze_single_file(file, repo_url, model):
    """Analyzes a single file and returns the results."""
    try:
        logging.debug(f"Analyzing file in thread: {file.filename}")
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
            "elapsed_time": elapsed_time,
            "repoUrl": repo_url,
        }
        return file_data
    except Exception as e:
        logging.exception(f"Error analyzing file in thread: {file.filename}")
        return {"error": str(e), "filename": file.filename, "repoUrl": repo_url}


def analyze_repository_task(repo_url, files, model):
    """
    This function runs in a background thread, completely detached from the
    HTTP request. It performs the analysis and updates the database.
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    total_files = len(files)
    analyzed_files_count = 0
    start_time = datetime.datetime.now()

    logging.info(f"BACKGROUND TASK: Starting analysis for {repo_name} with {total_files} files.")
    update_analysis_status(repo_name, "in-progress", start_time=start_time, progress=0)

    try:
        # Χρησιμοποιούμε έναν τοπικό executor για την παραλληλοποίηση των αρχείων
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Δημιουργούμε τα future objects
            futures = {executor.submit(analyze_single_file, file, repo_url, model): file for file in files.values()}

            for future in concurrent.futures.as_completed(futures):
                try:
                    result_data = future.result()

                    if "error" in result_data:
                        logging.error(f"A thread failed for file {result_data.get('filename')}: {result_data['error']}")
                        continue # Προχωράμε στο επόμενο

                    analyzed_files_count += 1
                    save_analysis_to_db(repo_name, result_data)
                    progress = int((analyzed_files_count / total_files) * 100)
                    update_analysis_status(repo_name, "in-progress", start_time=start_time, progress=progress)

                    logging.info(
                        f"BACKGROUND TASK: Successfully analyzed file {analyzed_files_count}/{total_files}: {result_data['filename']}")

                except Exception as exc:
                    # Αν ένα future αποτύχει για κάποιο λόγο
                    logging.error(f'BACKGROUND TASK: A file analysis generated an exception: {exc}')

        # Όταν ολοκληρωθεί ο βρόγχος, η ανάλυση τελείωσε επιτυχώς
        end_time = datetime.datetime.now()
        logging.info(f"BACKGROUND TASK: Analysis completed for repository: {repo_name}.")
        update_analysis_status(repo_name, "completed", start_time=start_time, end_time=end_time, progress=100)

    except Exception as e:
        # Αν συμβεί κάποιο μεγάλο λάθος στην όλη διαδικασία
        end_time = datetime.datetime.now()
        logging.exception(f"BACKGROUND TASK: A critical error occurred during analysis for {repo_name}.")
        update_analysis_status(
            repo_name, "error", start_time=start_time, end_time=end_time,
            error_message=str(e)
        )


def init_routes(app):
    # Swagger UI Configuration
    SWAGGER_URL = "/swagger"
    API_URL = "/static/swagger.json"
    swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={"app_name": "KU-Detection-Back-End API"})
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    # ( /commits, /repos, /detected_kus, etc. )
    @app.route("/commits", methods=["POST"])
    def list_commits():
        data = request.get_json()
        repo_url = data.get("repo_url")
        commit_limit = data.get("limit", 50)
        if not repo_url:
            return jsonify({"error": "Repository URL is required"}), 400
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        try:
            if not repo_exists(repo_name):
                clone_repo(repo_url, os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", str(repo_name)))
            else:
                pull_repo(os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", str(repo_name)))
            commits = extract_contributions(os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", repo_name), commit_limit=commit_limit)
            save_commits_to_db(repo_name, commits)
            return jsonify(commits), 200
        except Exception as e:
            logging.exception("Error during git operations in list_commits")
            return jsonify({"error": str(e)}), 500

    @app.route("/repos", methods=["POST"])
    def create_repo():
        data = request.json
        repo_name = data.get("repo_name")
        url = data.get("url", "")
        organization = data.get("organization", None)  # Παίρνουμε τον οργανισμό από το request
        description = data.get("description", "")
        comments = data.get("comments", "")
        try:
            # Περνάμε τον οργανισμό στη συνάρτηση αποθήκευσης
            save_repo_to_db(repo_name, url, organization, description, comments)
            return jsonify({"message": "Repository created successfully"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/detected_kus", methods=["GET"])
    def get_detected_kus():
        try:
            kus_list = getdetected_kus()
            if kus_list is not None:
                return jsonify(kus_list), 200
            else:
                return jsonify({"error": "Failed to retrieve detected KUs"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/repos/<string:repo_name>", methods=["PUT"])
    def edit_repo(repo_name):
        data = request.json
        url = data.get("url", "")
        organization = data.get("organization", None)  # Παίρνουμε τον οργανισμό από το request
        description = data.get("description", "")
        comments = data.get("comments", "")
        try:
            # Περνάμε τον οργανισμό στη συνάρτηση αποθήκευσης
            save_repo_to_db(repo_name, url, organization, description, comments)
            return jsonify({"message": "Repository updated successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/timestamps", methods=["GET"])
    def get_timestamps():
        try:
            repo_name = request.args.get("repo_name")
            if not repo_name:
                return jsonify({"error": "Repository name is required"}), 400
            timestamps = get_commits_timestamps_from_db(repo_name)
            if timestamps is None:
                return jsonify({"error": "Failed to retrieve timestamps"}), 500
            return jsonify(timestamps), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/historytime", methods=["GET"])
    def historytime():
        try:
            repo_url = request.args.get("repo_url")
            if not repo_url:
                return jsonify({"error": "Missing 'repo_url' parameter"}), 400
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            commit_history = get_history_repo(repo_url, repo_name, CLONED_REPO_BASE_PATH)
            commit_dates = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in commit_history]
            return jsonify({"repo_name": repo_name, "commit_dates": commit_dates}), 200
        except Exception as e:
            logging.exception("Error in historytime")
            return jsonify({"error": str(e)}), 500

    @app.route("/delete_repo/<string:repo_name>", methods=["DELETE"])
    def delete_repo(repo_name):
        try:
            delete_repo_from_db(repo_name)
            return jsonify({"message": f"Repository '{repo_name}' and related data deleted successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/repos", methods=["GET"])
    def list_repos():
        try:
            repos = get_all_repos_from_db()
            return jsonify(repos), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    @app.route("/ku_risk", methods=["GET"])
    def get_ku_risk_endpoint():
        """
        Calculates and returns the risk associated with each Knowledge Unit (KU).
        The risk is a product of the probability of loss and the impact of that loss.
        """
        try:
            risk_data = calculate_risks()
            if "error" in risk_data:
                return jsonify(risk_data), 500

            # Μετατροπή σε λίστα για ευκολότερη διαχείριση στο frontend
            ku_risk_list = [
                {"ku_name": ku, **data} for ku, data in risk_data.get("ku_risk", {}).items()
            ]

            return jsonify(ku_risk_list), 200

        except Exception as e:
            logging.exception("Error in /ku_risk endpoint")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    @app.route("/employee_risk", methods=["GET"])
    def get_employee_risk_endpoint():
        """
        Calculates and returns the risk associated with the hypothetical departure
        of each employee, in both absolute and relative terms.
        """
        try:
            risk_data = calculate_risks()
            if "error" in risk_data:
                return jsonify(risk_data), 500

            # Μετατροπή σε λίστα για ευκολότερη διαχείριση στο frontend
            employee_risk_list = [
                {"employee_name": employee, **data} for employee, data in risk_data.get("employee_risk", {}).items()
            ]

            return jsonify(employee_risk_list), 200

        except Exception as e:
            logging.exception("Error in /employee_risk endpoint")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    @app.route("/analyze", methods=["GET"])
    def analyze():
        repo_url = request.args.get("repo_url")
        if not repo_url:
            logging.error("No repository URL provided.")
            return jsonify({"error": "Repository URL is required"}), 400

        repo_name = repo_url.split("/")[-1].replace(".git", "")
        logging.info(f"Received analysis request for repository: {repo_name}")

        # Έλεγχος αν μια ανάλυση ήδη τρέχει
        current_status = get_analysis_status(repo_name)
        if current_status and current_status.get('status') == 'in-progress':
            logging.warning(f"Analysis for {repo_name} is already in progress.")
            return jsonify({"message": "Analysis is already in progress for this repository."}), 409 # 409 Conflict

        commits = get_commits_from_db(repo_name)
        if not commits:
            logging.error(f"No commits found for repository: {repo_name}")
            return jsonify({"error": "No commits found for the repository"}), 400

        try:
            files = read_files_from_dict_list(commits)
            logging.info(f"Found {len(files)} files. Submitting analysis task to background executor.")

            # Υποβάλλουμε την εργασία στον global executor
            background_task_executor.submit(analyze_repository_task, repo_url, files, model)

            # Επιστρέφουμε αμέσως μια απάντηση 202 Accepted
            return jsonify({
                "message": "Analysis started in the background.",
                "repo_name": repo_name,
                "status_endpoint": f"/analysis_status?repo_name={repo_name}"
            }), 202

        except Exception as e:
            logging.exception(f"Failed to start analysis for repository: {repo_name}")
            return jsonify({"error": "An error occurred while trying to start the analysis"}), 500

    @app.route("/analysis_status", methods=["GET"])
    def analysis_status_endpoint():
        repo_name = request.args.get("repo_name")
        if not repo_name:
            return jsonify({"error": "Repository name is required"}), 400
        status_info = get_analysis_status(repo_name)
        if status_info:
            return jsonify(status_info), 200
        else:
            # Είναι καλό να επιστρέφουμε μια κατάσταση 'not_started' αντί για 404
            return jsonify({"status": "not_started", "progress": 0}), 200

    @app.route("/analyzedb", methods=["GET"])
    def analyzedb():
        try:
            repo_name = request.args.get("repo_name")
            if not repo_name:
                return jsonify({"error": "repo_name parameter is required"}), 400
            analysis_data = get_analysis_from_db(repo_name)
            if analysis_data is None:
                return jsonify({"error": "Failed to retrieve analysis data"}), 500
            return jsonify(analysis_data), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/analyzeall", methods=["GET"])
    def analyzeall():
        try:
            analysis_data = get_allanalysis_from_db()
            if analysis_data is None:
                return jsonify({"error": "Failed to retrieve all analysis data"}), 500
            return jsonify(analysis_data), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route("/ku_statistics", methods=["GET"])
    def get_ku_statistics():
        """
        Επιστρέφει μια λίστα με όλα τα μοναδικά KUs και το πλήθος
        των εμφανίσεών τους σε όλα τα projects.
        """
        try:
            # Κάλεσε τη νέα συνάρτηση που έφτιαξες στο data_db.py
            ku_counts = get_ku_counts_from_db()

            if ku_counts is not None:
                # Αν όλα πήγαν καλά, στείλε τα δεδομένα ως JSON
                return jsonify(ku_counts), 200
            else:
                # Αν η συνάρτηση επέστρεψε None (π.χ. λόγω σφάλματος)
                return jsonify({"error": "Failed to retrieve KU statistics"}), 500
        except Exception as e:
            # Γενικό σφάλμα
            logging.exception("Error in /ku_statistics endpoint")
            return jsonify({"error": str(e)}), 500

    @app.route("/organization_stats", methods=["GET"])
    def get_organization_statistics():
        """
        Επιστρέφει μια λίστα με τα ονόματα των οργανισμών (π.χ. 'apache')
        και τον αριθμό των projects που έχουμε αποθηκευμένα για τον καθένα.
        """
        try:
            # Κάλεσε τη νέα συνάρτηση από το data_db.py
            org_counts = get_organization_project_counts()

            if org_counts is not None:
                return jsonify(org_counts), 200
            else:
                return jsonify({"error": "Failed to retrieve organization statistics"}), 500
        except Exception as e:
            logging.exception("Error in /organization_stats endpoint")
            return jsonify({"error": str(e)}), 500

    @app.route("/ku_by_organization", methods=["GET"])
    def get_ku_by_organization_stats():
        """
        Επιστρέφει μια λίστα οργανισμών, και για τον καθένα, μια λίστα
        με τα KUs που εντοπίστηκαν στα projects του και το πλήθος τους.
        """
        try:
            data = get_ku_counts_by_organization()
            if data is not None:
                return jsonify(data), 200
            else:
                return jsonify({"error": "Failed to retrieve KU statistics by organization"}), 500
        except Exception as e:
            logging.exception("Error in /ku_by_organization endpoint")
            return jsonify({"error": str(e)}), 500

    @app.route("/monthly_analysis_stats", methods=["GET"])
    def get_monthly_analysis_statistics():
        """
        Επιστρέφει μια λίστα οργανισμών, και για τον καθένα, το πλήθος
        των αναλύσεων που έγιναν ανά μήνα στα projects του.
        """
        try:
            data = get_monthly_analysis_counts_by_org()
            if data is not None:
                return jsonify(data), 200
            else:
                return jsonify({"error": "Failed to retrieve monthly analysis statistics"}), 500
        except Exception as e:
            logging.exception("Error in /monthly_analysis_stats endpoint")
            return jsonify({"error": str(e)}), 500

    @app.route("/cluster_repos", methods=["POST"])
    def cluster_repos():
        """
        Εκτελεί ομαδοποίηση K-Means στα repositories με βάση τα KUs τους.
        Δέχεται ένα JSON body με το κλειδί 'num_clusters'.
        Example: {"num_clusters": 5}
        """
        data = request.get_json()
        if not data or "num_clusters" not in data:
            return jsonify({"error": "Missing 'num_clusters' in request body"}), 400

        try:
            num_clusters = int(data["num_clusters"])
            if num_clusters < 2:
                return jsonify({"error": "Number of clusters must be at least 2"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "'num_clusters' must be an integer"}), 400

        try:
            # Κάλεσε τη νέα συνάρτηση που εκτελεί την ομαδοποίηση
            clustered_data = cluster_repositories_by_kus(num_clusters)

            if clustered_data is not None:
                return jsonify(clustered_data), 200
            else:
                # Γενικό σφάλμα αν η συνάρτηση επέστρεψε None για άγνωστο λόγο
                return jsonify({"error": "Failed to perform clustering due to an internal error"}), 500

        except ValueError as ve:
            # Πιάνει το σφάλμα για λίγα repositories (π.χ. ζητάς 5 clusters ενώ έχεις μόνο 3 repos)
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logging.exception("Error in /cluster_repos endpoint")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    @app.route("/analysis_results", methods=["GET"])
    def get_analysis_results_endpoint():
        """
        Επιστρέφει τα δεδομένα του πίνακα analysis_results.
        Μπορεί να φιλτραριστεί με προαιρετικές παραμέτρους query:
        - start_date (YYYY-MM)
        - end_date (YYYY-MM)
        Αν δεν δοθούν παράμετροι, επιστρέφει όλα τα δεδομένα.
        """
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Έλεγχος της μορφής των παραμέτρων, μόνο αν έχουν δοθεί
        try:
            if start_date:
                datetime.datetime.strptime(start_date, '%Y-%m')
            if end_date:
                datetime.datetime.strptime(end_date, '%Y-%m')
        except ValueError:
            return jsonify({"error": "Invalid date format. Please use YYYY-MM."}), 400

        try:
            # Κλήση της νέας, ευέλικτης συνάρτησης
            data = get_analysis_results(start_date, end_date)

            if data is not None:
                return jsonify(data), 200
            else:
                return jsonify({"error": "Failed to retrieve analysis results"}), 500

        except Exception as e:
            logging.exception("Error in /analysis_results endpoint")
            return jsonify({"error": str(e)}), 500


init_routes(app)

if __name__ == "__main__":
    # Για production, θα χρησιμοποιούσες έναν WSGI server όπως Gunicorn ή uWSGI
    app.run(debug=True, host='0.0.0.0', port=5000)