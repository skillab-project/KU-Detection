import os

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
)
from core.git_operations import clone_repo, repo_exists, extract_contributions
from core.git_operations.repo import pull_repo, get_history_repo
from core.utils.code_files_loader import read_files_from_dict_list
from flask_swagger_ui import get_swaggerui_blueprint  # Import the Swagger UI blueprint
from core.ml_operations.loader import load_codebert_model
from core.analysis.codebert_sliding_window import codebert_sliding_window
from config.settings import CLONED_REPO_BASE_PATH, CODEBERT_BASE_PATH
import threading
import time
import logging
from flask import Flask, request, jsonify, Response
import json
import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes.  This is generally better than disabling it.

# Load model
model = load_codebert_model(CODEBERT_BASE_PATH, 27)


def analyze_repository_background(repo_url, files):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    analysis_results = []
    total_files = len(files)
    analyzed_files_count = 0
    start_time = datetime.datetime.now()

    logging.info(f"Starting analysis for repository: {repo_name}")
    update_analysis_status(
        repo_name, "in-progress", start_time=start_time, progress=0
    )

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
                "elapsed_time": elapsed_time,
                "repoUrl": repo_url,
            }
            analysis_results.append(file_data)
            analyzed_files_count += 1

            logging.info(
                f"Successfully analyzed file {analyzed_files_count}/{total_files}: {file.filename}"
            )

            # Save results using repo_name
            save_analysis_to_db(repo_name, file_data)

            # Update progress using repo_name
            progress = int((analyzed_files_count / total_files) * 100)
            update_analysis_status(
                repo_name, "in-progress", start_time=start_time, progress=progress
            )

            # Send progress and data update to frontend, including repoUrl
            yield f"data: {json.dumps({'progress': progress, 'file_data': file_data, 'repoUrl': repo_url})}\n\n"

        except Exception as e:
            logging.exception(
                f"Error analyzing file: {file.filename}. Total analyzed before error: {analyzed_files_count}."
            )
            update_analysis_status(
                repo_name,
                "error",
                start_time=start_time,
                end_time=datetime.datetime.now(),
                error_message=str(e),
            )
            yield f"data: {json.dumps({'error': str(e), 'repoUrl': repo_url})}\n\n"
            return

    # Final update after all files are processed
    end_time = datetime.datetime.now()
    logging.info(
        f"Analysis completed for repository: {repo_name}. Total files analyzed: {len(analysis_results)}"
    )
    update_analysis_status(
        repo_name, "completed", start_time=start_time, end_time=end_time, progress=100
    )
    yield f"data: {json.dumps({'progress': 100, 'message': 'Analysis completed', 'repoUrl': repo_url})}\n\n"


def init_routes(app):
    # Swagger UI Configuration
    SWAGGER_URL = "/swagger"  # URL for exposing Swagger UI
    API_URL = "/static/swagger.json"  # URL for the Swagger JSON definition

    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={"app_name": "KU-Detection-Back-End API"},
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    @app.route("/commits", methods=["POST"])
    def list_commits():
        """
        Retrieve commits from a repository.
        """
        data = request.get_json()
        repo_url = data.get("repo_url")
        commit_limit = data.get("limit", 50)

        if not repo_url:
            return jsonify({"error": "Repository URL is required"}), 400

        repo_name = repo_url.split("/")[-1].replace(".git", "")

        try:  # Added try-except block
            if not repo_exists(repo_name):
                clone_repo(
                    repo_url,
                    os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", str(repo_name)),
                )
            else:
                # Pull the latest changes if the repository already exists
                pull_repo(os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", str(repo_name)))

            commits = extract_contributions(
                os.path.join(CLONED_REPO_BASE_PATH, "fake_session_id", repo_name),
                commit_limit=commit_limit,
            )
            save_commits_to_db(repo_name, commits)
            return jsonify(commits), 200  # Added status code 200

        except Exception as e:  # Catch any exception during git operations
            logging.exception("Error during git operations in list_commits") # Log with traceback
            return jsonify({"error": str(e)}), 500


    @app.route("/repos", methods=["POST"])
    def create_repo():
        """
        Create a new repository entry.
        """
        data = request.json
        repo_name = data.get("repo_name")
        url = data.get("url", "")  # Default to empty string if 'url' is not provided
        description = data.get("description", "") # Default
        comments = data.get("comments", "")  # Default

        try:
            save_repo_to_db(repo_name, url, description, comments)
            return jsonify({"message": "Repository created successfully"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/detected_kus", methods=["GET"])
    def get_detected_kus():
        """
        Retrieve detected KUs.
        """
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
        """
        Edit an existing repository entry.
        """
        data = request.json
        url = data.get("url", "") # Default
        description = data.get("description", "")# Default
        comments = data.get("comments", "")# Default

        try:
            save_repo_to_db(repo_name, url, description, comments)
            return jsonify({"message": "Repository updated successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/timestamps", methods=["GET"])
    def get_timestamps():
        """
        Retrieve timestamps for commits.
        """
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
        """
        Retrieve commit history timestamps.
        """
        try:
            repo_url = request.args.get("repo_url")
            if not repo_url:
                return jsonify({"error": "Missing 'repo_url' parameter"}), 400

            repo_name = repo_url.split("/")[-1].replace(".git", "")
            commit_history = get_history_repo(repo_url, repo_name, CLONED_REPO_BASE_PATH)
            commit_dates = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in commit_history]
            return jsonify({"repo_name": repo_name, "commit_dates": commit_dates}), 200

        except Exception as e:
            logging.exception("Error in historytime") #Added logging
            return jsonify({"error": str(e)}), 500


    @app.route("/delete_repo/<string:repo_name>", methods=["DELETE"])
    def delete_repo(repo_name):
        """
        Delete a repository entry.
        """
        try:
            delete_repo_from_db(repo_name)
            return jsonify({"message": f"Repository '{repo_name}' and related data deleted successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/repos", methods=["GET"])
    def list_repos():
        """
        List all repository entries.
        """
        try:
            repos = get_all_repos_from_db()
            return jsonify(repos), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    def start_analysis_in_background(repo_name, files):
        analysis_thread = threading.Thread(
            target=analyze_repository_background, args=(repo_name, files)
        )
        analysis_thread.start()

    @app.route("/analyze", methods=["GET"])
    def analyze():
        """
        Analyze a repository.
        """
        repo_url = request.args.get("repo_url")

        if not repo_url:
            logging.error("No repository URL provided.")
            return jsonify({"error": "Repository URL is required"}), 400

        repo_name = repo_url.split("/")[-1].replace(".git", "")
        logging.info(f"Starting analysis for repository: {repo_name}")

        commits = get_commits_from_db(repo_name)

        if not commits:
            logging.error(f"No commits found for repository: {repo_name}")
            return jsonify({"error": "No commits found for the repository"}), 400

        try:
            files = read_files_from_dict_list(commits)
            logging.info(f"Retrieved {len(files)} files for analysis.")
            return Response(
                analyze_repository_background(repo_url, files), content_type="text/event-stream"
            )

        except Exception as e:
            logging.exception(f"Unexpected error during analysis for repository: {repo_name}")
            return jsonify({"error": "An error occurred during analysis"}), 500

    @app.route("/analysis_status", methods=["GET"])
    def analysis_status_endpoint():
        """
        Get analysis status for a repository.
        """
        repo_name = request.args.get("repo_name")

        if not repo_name:
            return jsonify({"error": "Repository name is required"}), 400

        status_info = get_analysis_status(repo_name)

        if status_info:
            return jsonify(status_info), 200
        else:
            return jsonify({"status": "not found"}), 404

    @app.route("/analyzedb", methods=["GET"])
    def analyzedb():
        """
        Get analysis data from the database.
        """
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
        """
        Get all analysis data.
        """
        try:
            analysis_data = get_allanalysis_from_db()

            if analysis_data is None:
                return jsonify({"error": "Failed to retrieve all analysis data"}), 500

            return jsonify(analysis_data), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


init_routes(app)  # Call init_routes AFTER defining it

if __name__ == "__main__":
    app.run(debug=True)