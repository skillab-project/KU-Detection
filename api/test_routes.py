import unittest
from unittest.mock import patch, MagicMock
import json
import os
import datetime
from flask import Flask
import sys
import logging

# Suppress logging during tests (adjust if needed for debugging)
logging.disable(logging.CRITICAL)

# Προσθήκη του γονικού φακέλου στο sys.path για να βρει τα modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import init_routes
from core.ml_operations.loader import load_codebert_model


class FlaskAPITests(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)  # Δημιουργία test Flask app
        # Mocking του load_codebert_model, ωστε να μην το φορτώνουμε κατά τα tests
        with patch('core.ml_operations.loader.load_codebert_model') as mock_load_model:
            mock_load_model.return_value = MagicMock()  # Επιστρέφει ένα mock object
            init_routes(self.app)  # Αρχικοποίηση routes στο *test* app

        self.client = self.app.test_client()

        # Δοκιμαστικά δεδομένα
        self.sample_repo_name = "test_repo"
        self.sample_repo_url = "https://github.com/apache/kafka"  # Πραγματική διεύθυνση (προσοχή στις αλλαγές!)
        self.sample_commits = [
            {"commit": "commit_id_1", "author": "author1", "filename": "file1.py",
             "timestamp": "2024-01-01 10:00:00", "sha": "sha1"},
            {"commit": "commit_id_2", "author": "author2", "filename": "file2.py",
             "timestamp": "2024-01-02 12:00:00", "sha": "sha2"}
        ]

    @patch('api.routes.save_commits_to_db')
    @patch('api.routes.extract_contributions')
    @patch('api.routes.pull_repo')
    @patch('api.routes.repo_exists')
    @patch('api.routes.clone_repo')
    def test_list_commits(self, mock_clone, mock_repo_exists, mock_pull, mock_extract, mock_save_commits):
        """
        Title: Testing repository commit listing functionality
        Description: This test verifies that the /commits endpoint correctly handles repository
        commit listing by cloning a new repository or pulling an existing one, extracting commit
        information, and returning the correct response. It tests both scenarios: when a repository
        doesn't exist (requiring cloning) and when it already exists (requiring pulling).
        Related methods: app.clone_repo, app.repo_exists, app.extract_contributions,
        app.pull_repo, app.save_commits_to_db
        """
        # Σενάριο 1: Το repo ΔΕΝ υπάρχει -> clone
        print("test")
        mock_repo_exists.return_value = False
        mock_extract.return_value = self.sample_commits  # return sample commits
        mock_save_commits.return_value = None

        response = self.client.post('/commits', json={"repo_url": self.sample_repo_url, "limit": 10})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, self.sample_commits)
        mock_clone.assert_called_once()
        mock_extract.assert_called_once()
        mock_save_commits.assert_called_once()

        # Σενάριο 2: Το repo υπάρχει -> pull
        mock_repo_exists.return_value = True
        mock_extract.return_value = self.sample_commits
        mock_save_commits.return_value = None
        mock_pull.reset_mock()  # Reset Mock
        mock_extract.reset_mock()
        mock_save_commits.reset_mock()

        response = self.client.post('/commits', json={"repo_url": self.sample_repo_url})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, self.sample_commits)
        mock_pull.assert_called_once()
        mock_extract.assert_called_once()
        mock_save_commits.assert_called_once()

    @patch('api.routes.save_repo_to_db')
    def test_create_repo(self, mock_save_repo):
        """
        Title: Testing repository creation endpoint
        Description: This test verifies that the /repos POST endpoint correctly creates a new
        repository entry in the database with the provided information. It tests both successful
        creation scenario and error handling when an exception occurs during database operation.
        Related methods: app.save_repo_to_db
        """
        # Test successful creation
        mock_save_repo.return_value = True

        response = self.client.post('/repos',
                                    json={
                                        "repo_name": self.sample_repo_name,
                                        "url": self.sample_repo_url,
                                        "description": "Test repo",
                                        "comments": "Test comment"
                                    })

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "Repository created successfully")
        mock_save_repo.assert_called_once_with(
            self.sample_repo_name, self.sample_repo_url, None, "Test repo", "Test comment"
        )

        # Test with exception
        mock_save_repo.side_effect = Exception("Database error")
        response = self.client.post('/repos',
                                    json={"repo_name": self.sample_repo_name, "url": self.sample_repo_url,
                                          "description": "Test repo",
                                          "comments": "Test comment"})  # Πρέπει να περιέχει όλα τα πεδία
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("error", data)
        mock_save_repo.side_effect = None

    @patch('api.routes.getdetected_kus')
    def test_get_detected_kus(self, mock_get_kus):
        """
        Title: Testing retrieval of detected knowledge units
        Description: This test verifies that the /detected_kus endpoint correctly retrieves
        the list of detected knowledge units from the database. It tests successful retrieval,
        handling of None results, and proper error handling when database exceptions occur.
        Related methods: app.getdetected_kus
        """
        # Test successful retrieval
        mock_get_kus.return_value = [{'author': 'author1', 'kus': ['KU1', 'KU2']},
                                     {'author': 'author2', 'kus': ['KU3']}]

        response = self.client.get('/detected_kus')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        self.assertIn('KU1', data[0]['kus'])

        # Test with None return
        mock_get_kus.return_value = None
        response = self.client.get('/detected_kus')
        self.assertEqual(response.status_code, 500)

        # Test with exception
        mock_get_kus.side_effect = Exception("Database error")
        response = self.client.get('/detected_kus')
        self.assertEqual(response.status_code, 500)
        mock_get_kus.side_effect = None

    @patch('api.routes.save_repo_to_db')
    def test_edit_repo(self, mock_save_repo):
        """
        Title: Testing repository information update functionality
        Description: This test verifies that the /repos/<repo_name> PUT endpoint correctly updates
        repository information in the database. It tests both successful update scenario and error
        handling when an exception occurs during the database operation.
        Related methods: app.save_repo_to_db
        """
        # Test successful update
        mock_save_repo.return_value = True

        response = self.client.put(f'/repos/{self.sample_repo_name}',
                                   json={
                                       "url": self.sample_repo_url,
                                       "description": "Updated description",
                                       "comments": "Updated comment"
                                   })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "Repository updated successfully")
        mock_save_repo.assert_called_once_with(
            self.sample_repo_name, self.sample_repo_url, None, "Updated description", "Updated comment"
        )

        # Test with exception
        mock_save_repo.side_effect = Exception("Database error")
        response = self.client.put(f'/repos/{self.sample_repo_name}',
                                   json={"url": self.sample_repo_url, "description": "Updated description",
                                         "comments": "Updated comment"})  # Πρέπει να περιέχει όλα τα πεδία
        self.assertEqual(response.status_code, 500)
        mock_save_repo.side_effect = None

    @patch('api.routes.get_commits_timestamps_from_db')
    def test_get_timestamps(self, mock_get_timestamps):
        """
        Title: Testing commit timestamp retrieval functionality
        Description: This test verifies that the /timestamps endpoint correctly retrieves
        commit timestamps for a specified repository from the database. It tests successful
        retrieval, validation of required parameters, and proper error handling when database
        operations fail.
        Related methods: app.get_commits_timestamps_from_db
        """
        # Test successful retrieval
        timestamps = [
            {"commit_id": "123", "timestamp": "2023-01-01T10:00:00"},
            {"commit_id": "456", "timestamp": "2023-01-02T11:00:00"}
        ]
        mock_get_timestamps.return_value = timestamps

        response = self.client.get(f'/timestamps?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)

        # Test without repo_name
        response = self.client.get('/timestamps')
        self.assertEqual(response.status_code, 400)

        # Test with None return
        mock_get_timestamps.return_value = None
        response = self.client.get(f'/timestamps?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 500)

    @patch('api.routes.get_history_repo')
    def test_historytime(self, mock_get_history):
        """
        Title: Testing repository commit history timeline retrieval
        Description: This test verifies that the /historytime endpoint correctly retrieves
        and formats a timeline of commit dates for a specified repository. It tests successful
        retrieval, validation of required parameters, and proper error handling when exceptions
        occur during history retrieval.
        Related methods: app.get_history_repo
        """
        # Mock return values
        mock_dates = [
            datetime.datetime(2023, 1, 1, 10, 0, 0),
            datetime.datetime(2023, 1, 2, 11, 0, 0)
        ]
        mock_get_history.return_value = mock_dates

        # Test the endpoint
        response = self.client.get(f'/historytime?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        repo_name = self.sample_repo_url.split("/")[-1].replace(".git", "")
        self.assertEqual(data["repo_name"], repo_name)
        self.assertEqual(len(data["commit_dates"]), 2)

        # Test without repo_url
        response = self.client.get('/historytime')
        self.assertEqual(response.status_code, 400)

        # Test with exception
        mock_get_history.side_effect = Exception("Error fetching history")
        response = self.client.get(f'/historytime?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 500)
        mock_get_history.side_effect = None

    @patch('api.routes.delete_repo_from_db')
    def test_delete_repo(self, mock_delete_repo):
        """
        Title: Testing repository deletion functionality
        Description: This test verifies that the /delete_repo/<repo_name> endpoint correctly
        removes a repository from the database. It tests successful deletion scenario and proper
        error handling when database exceptions occur during the deletion process.
        Related methods: app.delete_repo_from_db
        """
        mock_delete_repo.return_value = True  # Mock successful deletion
        response = self.client.delete(f'/delete_repo/{self.sample_repo_name}')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", data["message"])  # Check success message
        mock_delete_repo.assert_called_once_with(self.sample_repo_name)

        # Test failure case with specific error message:
        mock_delete_repo.side_effect = Exception("Database connection error")
        response = self.client.delete(f'/delete_repo/{self.sample_repo_name}')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(data.get("error"), "Database connection error")  # Έλεγχος για το μύνημα
        mock_delete_repo.side_effect = None

    @patch('api.routes.get_all_repos_from_db')
    def test_list_repos(self, mock_get_all_repos):
        """
        Title: Testing repository listing functionality
        Description: This test verifies that the /repos GET endpoint correctly retrieves all
        repositories from the database. It tests successful retrieval scenario and proper error
        handling when database exceptions occur during the retrieval process.
        Related methods: app.get_all_repos_from_db
        """
        mock_repos = [
            {"name": "Test Repo 1", "url": "testurl1", "organization": None, "description": "", "comments": "",
             "created_at": None, "updated_at": None, "analysis_status": None, "analysis_start_time": None,
             "analysis_end_time": None, "analysis_progress": None, "analysis_error_message": None},
            {"name": "Test Repo 2", "url": "testurl2", "organization": None, "description": "", "comments": "",
             "created_at": None, "updated_at": None, "analysis_status": None, "analysis_start_time": None,
             "analysis_end_time": None, "analysis_progress": None, "analysis_error_message": None}
        ]
        mock_get_all_repos.return_value = mock_repos
        response = self.client.get('/repos')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['name'], "Test Repo 1")
        self.assertEqual(data[1]['url'], "testurl2")

        # Test for DB error
        mock_get_all_repos.side_effect = Exception("Database error")
        response = self.client.get('/repos')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(data['error'], 'Database error')
        mock_get_all_repos.side_effect = None

    @patch('api.routes.background_task_executor.submit')
    @patch('api.routes.read_files_from_dict_list')
    @patch('api.routes.get_commits_from_db')
    @patch('api.routes.get_analysis_status')
    def test_analyze_endpoint(self, mock_get_status, mock_get_commits, mock_read_files, mock_submit):
        """
        Title: Testing repository code analysis functionality (NEW ASYNC LOGIC)
        Description: This test verifies that the /analyze endpoint correctly initiates a background task
        and handles various scenarios like missing parameters or an already running analysis.
        Related methods: get_analysis_status, get_commits_from_db, read_files_from_dict_list, background_task_executor.submit
        """
        # --- Scenario 1: Successful start of analysis ---
        mock_get_status.return_value = None  # No analysis is currently running
        mock_get_commits.return_value = self.sample_commits
        mock_read_files.return_value = {"file1.py": MagicMock()}

        response = self.client.get(f'/analyze?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 202)  # 202 Accepted is the correct response
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Analysis started in the background.')
        expected_repo_name = self.sample_repo_url.split("/")[-1].replace(".git", "")
        self.assertEqual(data['repo_name'], expected_repo_name)
        mock_submit.assert_called_once()  # Verify that the task was submitted

        # --- Scenario 2: Missing repo_url parameter ---
        response = self.client.get('/analyze')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Repository URL is required')

        # --- Scenario 3: Analysis already in progress ---
        mock_get_status.return_value = {'status': 'in-progress'}
        response = self.client.get(f'/analyze?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 409)  # 409 Conflict
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Analysis is already in progress for this repository.')

        # --- Scenario 4: No commits found for the repository ---
        mock_get_status.return_value = None  # Reset status
        mock_get_commits.return_value = []  # No commits found
        response = self.client.get(f'/analyze?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No commits found for the repository')

    @patch('api.routes.get_analysis_status')
    def test_analysis_status_endpoint(self, mock_get_status):
        """
        Title: Testing analysis status retrieval functionality
        Description: This test verifies that the /analysis_status endpoint correctly retrieves
        the current status of a code analysis process for a specified repository. It tests
        successful retrieval, parameter validation, and handling of non-existent analysis records.
        Related methods: app.get_analysis_status
        """
        # Mock data
        status_info = {
            "status": "completed",
            "progress": 100,
            "start_time": "2023-01-01T10:00:00",
            "end_time": "2023-01-01T10:05:00",
            "error_message": None
        }

        # Test successful retrieval
        mock_get_status.return_value = status_info

        response = self.client.get(f'/analysis_status?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "completed")

        # Test without repo_name
        response = self.client.get('/analysis_status')
        self.assertEqual(response.status_code, 400)

        # Test with no status found
        mock_get_status.return_value = None
        response = self.client.get(f'/analysis_status?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 200)  # CORRECT: Should return 200 with default status
        data = json.loads(response.data)
        self.assertEqual(data, {"status": "not_started", "progress": 0})

    @patch('api.routes.get_analysis_from_db')
    def test_analyzedb_endpoint(self, mock_get_analysis):
        """
        Title: Testing repository analysis results retrieval
        Description: This test verifies that the /analyzedb endpoint correctly retrieves
        the stored analysis results for a specified repository from the database. It tests
        successful retrieval, parameter validation, and proper error handling for database
        operations.
        Related methods: app.get_analysis_from_db
        """
        # Mock data
        analysis_data = [
            {"filename": "file1.py", "detected_kus": ["KU1", "KU2"], "author": "", "timestamp": None, "sha": "",
             "elapsed_time": ""},
            {"filename": "file2.py", "detected_kus": ["KU3"], "author": "", "timestamp": None, "sha": "",
             "elapsed_time": ""}
        ]

        # Test successful retrieval
        mock_get_analysis.return_value = analysis_data

        response = self.client.get(f'/analyzedb?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)

        # Test without repo_name
        response = self.client.get('/analyzedb')
        self.assertEqual(response.status_code, 400)

        # Test with None return
        mock_get_analysis.return_value = None
        response = self.client.get(f'/analyzedb?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 500)

        # Test with exception
        mock_get_analysis.side_effect = Exception("Database error")
        response = self.client.get(f'/analyzedb?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 500)
        mock_get_analysis.side_effect = None

    @patch('api.routes.get_allanalysis_from_db')
    def test_analyzeall_endpoint(self, mock_get_all_analysis):
        """
        Title: Testing retrieval of analysis results for all repositories
        Description: This test verifies that the /analyzeall endpoint correctly retrieves
        analysis results for all repositories from the database. It tests successful retrieval
        and proper error handling for database operations.
        Related methods: app.get_allanalysis_from_db
        """
        # Mock data
        all_analysis = [
            {"repo_name": "repo1", "files": [{"filename": "file1.py"}]},
            {"repo_name": "repo2", "files": [{"filename": "file2.py"}]}
        ]

        # Test successful retrieval
        mock_get_all_analysis.return_value = all_analysis

        response = self.client.get('/analyzeall')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)

        # Test with None return
        mock_get_all_analysis.return_value = None
        response = self.client.get('/analyzeall')
        self.assertEqual(response.status_code, 500)

        # Test with exception
        mock_get_all_analysis.side_effect = Exception("Database error")
        response = self.client.get('/analyzeall')
        self.assertEqual(response.status_code, 500)
        mock_get_all_analysis.side_effect = None

    @patch('api.routes.get_ku_counts_from_db')
    def test_get_ku_statistics(self, mock_get_counts):
        """
        Title: Testing KU statistics endpoint
        """
        mock_data = [{"ku_id": "KU_1", "count": 15}, {"ku_id": "KU_2", "count": 10}]
        mock_get_counts.return_value = mock_data

        response = self.client.get('/ku_statistics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)

        # Test with exception
        mock_get_counts.side_effect = Exception("DB Error")
        response = self.client.get('/ku_statistics')
        self.assertEqual(response.status_code, 500)
        mock_get_counts.side_effect = None

    @patch('api.routes.get_organization_project_counts')
    def test_get_organization_statistics(self, mock_get_counts):
        """
        Title: Testing organization statistics endpoint
        """
        mock_data = [{"organization": "apache", "count": 5}, {"organization": "google", "count": 3}]
        mock_get_counts.return_value = mock_data

        response = self.client.get('/organization_stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)

        # Test with exception
        mock_get_counts.side_effect = Exception("DB Error")
        response = self.client.get('/organization_stats')
        self.assertEqual(response.status_code, 500)
        mock_get_counts.side_effect = None

    @patch('api.routes.get_ku_counts_by_organization')
    def test_get_ku_by_organization_stats(self, mock_get_data):
        """
        Title: Testing KU statistics by organization endpoint
        """
        mock_data = [{
            "organization": "apache",
            "ku_counts": [{"ku_id": "KU_1", "count": 10}]
        }]
        mock_get_data.return_value = mock_data

        response = self.client.get('/ku_by_organization')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)

        # Test with exception
        mock_get_data.side_effect = Exception("DB Error")
        response = self.client.get('/ku_by_organization')
        self.assertEqual(response.status_code, 500)
        mock_get_data.side_effect = None

    @patch('api.routes.get_monthly_analysis_counts_by_org')
    def test_get_monthly_analysis_statistics(self, mock_get_data):
        """
        Title: Testing monthly analysis statistics endpoint
        """
        mock_data = [{
            "organization": "apache",
            "monthly_counts": [{"month": "2024-05", "count": 100}]
        }]
        mock_get_data.return_value = mock_data

        response = self.client.get('/monthly_analysis_stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)

        # Test with exception
        mock_get_data.side_effect = Exception("DB Error")
        response = self.client.get('/monthly_analysis_stats')
        self.assertEqual(response.status_code, 500)
        mock_get_data.side_effect = None


if __name__ == '__main__':
    unittest.main()
