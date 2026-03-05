import unittest
from unittest.mock import patch, MagicMock
import json
import os
import datetime
from flask import Flask
import sys
import logging

# Suppress logging during tests
logging.disable(logging.CRITICAL)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import init_routes
from core.ml_operations.loader import load_codebert_model

# -----------------------------------------------------------------------
# Σταθερά header που απαιτείται από όλα τα endpoints
# -----------------------------------------------------------------------
ORG_HEADER = {"X-User-Organization": "test_org"}


class FlaskAPITests(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        with patch('core.ml_operations.loader.load_codebert_model') as mock_load_model:
            mock_load_model.return_value = MagicMock()
            init_routes(self.app)

        self.client = self.app.test_client()

        self.sample_repo_name = "test_repo"
        self.sample_repo_url = "https://github.com/apache/kafka"
        self.sample_commits = [
            {"commit": "commit_id_1", "author": "author1", "filename": "file1.py",
             "timestamp": "2024-01-01 10:00:00", "sha": "sha1"},
            {"commit": "commit_id_2", "author": "author2", "filename": "file2.py",
             "timestamp": "2024-01-02 12:00:00", "sha": "sha2"}
        ]

    # ------------------------------------------------------------------
    # /commits
    # ------------------------------------------------------------------
    @patch('api.routes.save_commits_to_db')
    @patch('api.routes.extract_contributions')
    @patch('api.routes.pull_repo')
    @patch('api.routes.repo_exists')
    @patch('api.routes.clone_repo')
    def test_list_commits(self, mock_clone, mock_repo_exists, mock_pull, mock_extract, mock_save_commits):
        """
        Title: Testing repository commit listing functionality
        Description: Verifies the /commits endpoint handles cloning a new repo or pulling
        an existing one, extracting commit info, and returning the correct response.
        Also verifies that missing X-User-Organization header returns 400.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.post('/commits', json={"repo_url": self.sample_repo_url})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

        # Σενάριο 1: Το repo ΔΕΝ υπάρχει -> clone
        mock_repo_exists.return_value = False
        mock_extract.return_value = self.sample_commits
        mock_save_commits.return_value = None

        response = self.client.post(
            '/commits',
            json={"repo_url": self.sample_repo_url, "limit": 10},
            headers=ORG_HEADER
        )
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
        mock_pull.reset_mock()
        mock_extract.reset_mock()
        mock_save_commits.reset_mock()

        response = self.client.post(
            '/commits',
            json={"repo_url": self.sample_repo_url},
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, self.sample_commits)
        mock_pull.assert_called_once()
        mock_extract.assert_called_once()
        mock_save_commits.assert_called_once()

    # ------------------------------------------------------------------
    # /repos POST
    # ------------------------------------------------------------------
    @patch('api.routes.save_repo_to_db')
    def test_create_repo(self, mock_save_repo):
        """
        Title: Testing repository creation endpoint
        Description: Verifies the /repos POST endpoint creates a new repository entry.
        Tests missing header, successful creation, and database exception handling.
        NOTE: The organization is taken exclusively from the X-User-Organization header,
        NOT from the JSON body.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.post('/repos', json={"repo_name": self.sample_repo_name,
                                                    "url": self.sample_repo_url})
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής δημιουργία
        mock_save_repo.return_value = True

        response = self.client.post(
            '/repos',
            json={
                "repo_name": self.sample_repo_name,
                "url": self.sample_repo_url,
                "description": "Test repo",
                "comments": "Test comment"
            },
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "Repository created successfully")
        # Ο οργανισμός περνά από το header ("test_org"), ΟΧΙ από το body
        mock_save_repo.assert_called_once_with(
            self.sample_repo_name, self.sample_repo_url, "test_org", "Test repo", "Test comment"
        )

        # Σενάριο 2: Exception -> 500
        mock_save_repo.side_effect = Exception("Database error")
        response = self.client.post(
            '/repos',
            json={"repo_name": self.sample_repo_name, "url": self.sample_repo_url,
                  "description": "Test repo", "comments": "Test comment"},
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("error", data)
        mock_save_repo.side_effect = None

    # ------------------------------------------------------------------
    # /detected_kus
    # ------------------------------------------------------------------
    @patch('api.routes.getdetected_kus')
    def test_get_detected_kus(self, mock_get_kus):
        """
        Title: Testing retrieval of detected knowledge units
        Description: Verifies the /detected_kus endpoint retrieves KUs correctly.
        Tests missing header, successful retrieval, None result, and exception handling.
        NOTE: The organization is passed as a keyword argument from the header.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/detected_kus')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_get_kus.return_value = [{'author': 'author1', 'kus': ['KU1', 'KU2']},
                                     {'author': 'author2', 'kus': ['KU3']}]

        response = self.client.get('/detected_kus', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        self.assertIn('KU1', data[0]['kus'])
        # Ο οργανισμός πρέπει να περαστεί ως keyword argument
        mock_get_kus.assert_called_once_with(organization="test_org")

        # Σενάριο 2: None return -> 500
        mock_get_kus.return_value = None
        response = self.client.get('/detected_kus', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)

        # Σενάριο 3: Exception -> 500
        mock_get_kus.side_effect = Exception("Database error")
        response = self.client.get('/detected_kus', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_kus.side_effect = None

    # ------------------------------------------------------------------
    # /repos/<repo_name> PUT
    # ------------------------------------------------------------------
    @patch('api.routes.save_repo_to_db')
    def test_edit_repo(self, mock_save_repo):
        """
        Title: Testing repository information update functionality
        Description: Verifies the /repos/<repo_name> PUT endpoint updates repo info.
        Tests missing header, successful update, and exception handling.
        NOTE: Organization comes from the header, not the body.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.put(f'/repos/{self.sample_repo_name}',
                                   json={"url": self.sample_repo_url})
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ενημέρωση
        mock_save_repo.return_value = True

        response = self.client.put(
            f'/repos/{self.sample_repo_name}',
            json={
                "url": self.sample_repo_url,
                "description": "Updated description",
                "comments": "Updated comment"
            },
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "Repository updated successfully")
        mock_save_repo.assert_called_once_with(
            self.sample_repo_name, self.sample_repo_url, "test_org", "Updated description", "Updated comment"
        )

        # Σενάριο 2: Exception -> 500
        mock_save_repo.side_effect = Exception("Database error")
        response = self.client.put(
            f'/repos/{self.sample_repo_name}',
            json={"url": self.sample_repo_url, "description": "Updated description",
                  "comments": "Updated comment"},
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 500)
        mock_save_repo.side_effect = None

    # ------------------------------------------------------------------
    # /timestamps
    # ------------------------------------------------------------------
    @patch('api.routes.get_commits_timestamps_from_db')
    def test_get_timestamps(self, mock_get_timestamps):
        """
        Title: Testing commit timestamp retrieval functionality
        Description: Verifies the /timestamps endpoint retrieves timestamps correctly.
        Tests missing header, missing repo_name, successful retrieval, and None result.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get(f'/timestamps?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Λείπει το repo_name -> 400
        response = self.client.get('/timestamps', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)

        # Σενάριο 2: Επιτυχής ανάκτηση
        timestamps = [
            {"commit_id": "123", "timestamp": "2023-01-01T10:00:00"},
            {"commit_id": "456", "timestamp": "2023-01-02T11:00:00"}
        ]
        mock_get_timestamps.return_value = timestamps

        response = self.client.get(
            f'/timestamps?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)

        # Σενάριο 3: None return -> 500
        mock_get_timestamps.return_value = None
        response = self.client.get(
            f'/timestamps?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 500)

    # ------------------------------------------------------------------
    # /historytime
    # ------------------------------------------------------------------
    @patch('api.routes.get_history_repo')
    def test_historytime(self, mock_get_history):
        """
        Title: Testing repository commit history timeline retrieval
        Description: Verifies the /historytime endpoint formats commit dates correctly.
        Tests missing header, missing repo_url, successful retrieval, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get(f'/historytime?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Λείπει το repo_url -> 400
        response = self.client.get('/historytime', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)

        # Σενάριο 2: Επιτυχής ανάκτηση
        mock_dates = [
            datetime.datetime(2023, 1, 1, 10, 0, 0),
            datetime.datetime(2023, 1, 2, 11, 0, 0)
        ]
        mock_get_history.return_value = mock_dates

        response = self.client.get(
            f'/historytime?repo_url={self.sample_repo_url}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        repo_name = self.sample_repo_url.split("/")[-1].replace(".git", "")
        self.assertEqual(data["repo_name"], repo_name)
        self.assertEqual(len(data["commit_dates"]), 2)

        # Σενάριο 3: Exception -> 500
        mock_get_history.side_effect = Exception("Error fetching history")
        response = self.client.get(
            f'/historytime?repo_url={self.sample_repo_url}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 500)
        mock_get_history.side_effect = None

    # ------------------------------------------------------------------
    # /delete_repo/<repo_name>
    # ------------------------------------------------------------------
    @patch('api.routes.delete_repo_from_db')
    def test_delete_repo(self, mock_delete_repo):
        """
        Title: Testing repository deletion functionality
        Description: Verifies the /delete_repo/<repo_name> endpoint deletes repos correctly.
        Tests missing header, successful deletion, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.delete(f'/delete_repo/{self.sample_repo_name}')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής διαγραφή
        mock_delete_repo.return_value = True
        response = self.client.delete(
            f'/delete_repo/{self.sample_repo_name}',
            headers=ORG_HEADER
        )
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", data["message"])
        mock_delete_repo.assert_called_once_with(self.sample_repo_name)

        # Σενάριο 2: Exception -> 500
        mock_delete_repo.side_effect = Exception("Database connection error")
        response = self.client.delete(
            f'/delete_repo/{self.sample_repo_name}',
            headers=ORG_HEADER
        )
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(data.get("error"), "Database connection error")
        mock_delete_repo.side_effect = None

    # ------------------------------------------------------------------
    # /repos GET
    # ------------------------------------------------------------------
    @patch('api.routes.get_all_repos_from_db')
    def test_list_repos(self, mock_get_all_repos):
        """
        Title: Testing repository listing functionality
        Description: Verifies the /repos GET endpoint retrieves repos for an organization.
        Tests missing header, successful retrieval, and exception handling.
        NOTE: The organization now comes exclusively from the header. The previous
        ?organization= query parameter filter is no longer used by the endpoint.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/repos')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση με header
        mock_repos = [
            {"name": "Apache Repo", "url": "testurl1", "organization": "test_org",
             "description": "", "comments": "", "created_at": None, "updated_at": None,
             "analysis_status": None, "analysis_start_time": None, "analysis_end_time": None,
             "analysis_progress": None, "analysis_error_message": None},
            {"name": "Another Repo", "url": "testurl2", "organization": "test_org",
             "description": "", "comments": "", "created_at": None, "updated_at": None,
             "analysis_status": None, "analysis_start_time": None, "analysis_end_time": None,
             "analysis_progress": None, "analysis_error_message": None}
        ]
        mock_get_all_repos.return_value = mock_repos

        response = self.client.get('/repos', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        # Ο οργανισμός περνά από το header ("test_org")
        mock_get_all_repos.assert_called_once_with(organization="test_org")

        # Σενάριο 2: Exception -> 500
        mock_get_all_repos.reset_mock()
        mock_get_all_repos.side_effect = Exception("Database error")

        response = self.client.get('/repos', headers=ORG_HEADER)
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(data['error'], 'Database error')
        mock_get_all_repos.side_effect = None

    # ------------------------------------------------------------------
    # /analyze
    # ------------------------------------------------------------------
    @patch('api.routes.background_task_executor.submit')
    @patch('api.routes.read_files_from_dict_list')
    @patch('api.routes.get_commits_from_db')
    @patch('api.routes.get_analysis_status')
    def test_analyze_endpoint(self, mock_get_status, mock_get_commits, mock_read_files, mock_submit):
        """
        Title: Testing repository code analysis functionality (async)
        Description: Verifies the /analyze endpoint initiates a background task correctly.
        Tests missing header, missing repo_url, already running analysis, no commits found,
        and successful task submission.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get(f'/analyze?repo_url={self.sample_repo_url}')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Λείπει το repo_url -> 400
        response = self.client.get('/analyze', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Repository URL is required')

        # Σενάριο 2: Επιτυχής εκκίνηση ανάλυσης -> 202
        mock_get_status.return_value = None
        mock_get_commits.return_value = self.sample_commits
        mock_read_files.return_value = {"file1.py": MagicMock()}

        response = self.client.get(
            f'/analyze?repo_url={self.sample_repo_url}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 202)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Analysis started in the background.')
        expected_repo_name = self.sample_repo_url.split("/")[-1].replace(".git", "")
        self.assertEqual(data['repo_name'], expected_repo_name)
        mock_submit.assert_called_once()

        # Σενάριο 3: Ανάλυση ήδη σε εξέλιξη -> 409
        mock_get_status.return_value = {'status': 'in-progress'}
        response = self.client.get(
            f'/analyze?repo_url={self.sample_repo_url}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 409)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Analysis is already in progress for this repository.')

        # Σενάριο 4: Δεν βρέθηκαν commits -> 400
        mock_get_status.return_value = None
        mock_get_commits.return_value = []
        response = self.client.get(
            f'/analyze?repo_url={self.sample_repo_url}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No commits found for the repository')

    # ------------------------------------------------------------------
    # /analysis_status
    # ------------------------------------------------------------------
    @patch('api.routes.get_analysis_status')
    def test_analysis_status_endpoint(self, mock_get_status):
        """
        Title: Testing analysis status retrieval functionality
        Description: Verifies the /analysis_status endpoint retrieves analysis status.
        Tests missing header, missing repo_name, successful retrieval, and not_started case.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get(f'/analysis_status?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Λείπει το repo_name -> 400
        response = self.client.get('/analysis_status', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)

        # Σενάριο 2: Επιτυχής ανάκτηση
        status_info = {
            "status": "completed",
            "progress": 100,
            "start_time": "2023-01-01T10:00:00",
            "end_time": "2023-01-01T10:05:00",
            "error_message": None
        }
        mock_get_status.return_value = status_info

        response = self.client.get(
            f'/analysis_status?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "completed")

        # Σενάριο 3: Δεν βρέθηκε status -> 200 με "not_started"
        mock_get_status.return_value = None
        response = self.client.get(
            f'/analysis_status?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, {"status": "not_started", "progress": 0})

    # ------------------------------------------------------------------
    # /analyzedb
    # ------------------------------------------------------------------
    @patch('api.routes.get_analysis_from_db')
    def test_analyzedb_endpoint(self, mock_get_analysis):
        """
        Title: Testing repository analysis results retrieval
        Description: Verifies the /analyzedb endpoint retrieves stored analysis results.
        Tests missing header, missing repo_name, successful retrieval, None result,
        and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get(f'/analyzedb?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Λείπει το repo_name -> 400
        response = self.client.get('/analyzedb', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)

        # Σενάριο 2: Επιτυχής ανάκτηση
        analysis_data = [
            {"filename": "file1.py", "detected_kus": ["KU1", "KU2"], "author": "",
             "timestamp": None, "sha": "", "elapsed_time": ""},
            {"filename": "file2.py", "detected_kus": ["KU3"], "author": "",
             "timestamp": None, "sha": "", "elapsed_time": ""}
        ]
        mock_get_analysis.return_value = analysis_data

        response = self.client.get(
            f'/analyzedb?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)

        # Σενάριο 3: None return -> 500
        mock_get_analysis.return_value = None
        response = self.client.get(
            f'/analyzedb?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 500)

        # Σενάριο 4: Exception -> 500
        mock_get_analysis.side_effect = Exception("Database error")
        response = self.client.get(
            f'/analyzedb?repo_name={self.sample_repo_name}',
            headers=ORG_HEADER
        )
        self.assertEqual(response.status_code, 500)
        mock_get_analysis.side_effect = None

    # ------------------------------------------------------------------
    # /analyzeall
    # ------------------------------------------------------------------
    @patch('api.routes.get_allanalysis_from_db')
    def test_analyzeall_endpoint(self, mock_get_all_analysis):
        """
        Title: Testing retrieval of analysis results for all repositories
        Description: Verifies the /analyzeall endpoint retrieves analysis for all repos.
        Tests missing header, successful retrieval, None result, and exception handling.
        NOTE: The organization is now passed from the header, not as a query parameter.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/analyzeall')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        all_analysis = [
            {"repo_name": "repo1", "files": [{"filename": "file1.py"}]},
            {"repo_name": "repo2", "files": [{"filename": "file2.py"}]}
        ]
        mock_get_all_analysis.return_value = all_analysis

        response = self.client.get('/analyzeall', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        mock_get_all_analysis.assert_called_once_with(organization="test_org")

        # Σενάριο 2: None return -> 500
        mock_get_all_analysis.return_value = None
        response = self.client.get('/analyzeall', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)

        # Σενάριο 3: Exception -> 500
        mock_get_all_analysis.side_effect = Exception("Database error")
        response = self.client.get('/analyzeall', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_all_analysis.side_effect = None

    # ------------------------------------------------------------------
    # /ku_statistics
    # ------------------------------------------------------------------
    @patch('api.routes.get_ku_counts_from_db')
    def test_get_ku_statistics(self, mock_get_counts):
        """
        Title: Testing KU statistics endpoint
        Description: Verifies the /ku_statistics endpoint returns KU counts for an org.
        Tests missing header, successful retrieval, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/ku_statistics')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_data = [{"ku_id": "KU_1", "count": 15}, {"ku_id": "KU_2", "count": 10}]
        mock_get_counts.return_value = mock_data

        response = self.client.get('/ku_statistics', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)
        mock_get_counts.assert_called_once_with(organization="test_org")

        # Σενάριο 2: Exception -> 500
        mock_get_counts.side_effect = Exception("DB Error")
        response = self.client.get('/ku_statistics', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_counts.side_effect = None

    # ------------------------------------------------------------------
    # /organization_stats
    # ------------------------------------------------------------------
    @patch('api.routes.get_organization_project_counts')
    def test_get_organization_statistics(self, mock_get_counts):
        """
        Title: Testing organization statistics endpoint
        Description: Verifies the /organization_stats endpoint returns org project counts.
        Tests missing header, successful retrieval, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/organization_stats')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_data = [{"organization": "apache", "count": 5}, {"organization": "google", "count": 3}]
        mock_get_counts.return_value = mock_data

        response = self.client.get('/organization_stats', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)
        mock_get_counts.assert_called_once_with(organization="test_org")

        # Σενάριο 2: Exception -> 500
        mock_get_counts.side_effect = Exception("DB Error")
        response = self.client.get('/organization_stats', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_counts.side_effect = None

    # ------------------------------------------------------------------
    # /ku_by_organization
    # ------------------------------------------------------------------
    @patch('api.routes.get_ku_counts_by_organization')
    def test_get_ku_by_organization_stats(self, mock_get_data):
        """
        Title: Testing KU statistics by organization endpoint
        Description: Verifies the /ku_by_organization endpoint returns KU counts per org.
        Tests missing header, successful retrieval, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/ku_by_organization')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_data = [{
            "organization": "apache",
            "ku_counts": [{"ku_id": "KU_1", "count": 10}]
        }]
        mock_get_data.return_value = mock_data

        response = self.client.get('/ku_by_organization', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)
        mock_get_data.assert_called_once_with(organization="test_org")

        # Σενάριο 2: Exception -> 500
        mock_get_data.side_effect = Exception("DB Error")
        response = self.client.get('/ku_by_organization', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_data.side_effect = None

    # ------------------------------------------------------------------
    # /monthly_analysis_stats
    # ------------------------------------------------------------------
    @patch('api.routes.get_monthly_analysis_counts_by_org')
    def test_get_monthly_analysis_statistics(self, mock_get_data):
        """
        Title: Testing monthly analysis statistics endpoint
        Description: Verifies the /monthly_analysis_stats endpoint returns monthly counts.
        Tests missing header, successful retrieval, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/monthly_analysis_stats')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_data = [{
            "organization": "apache",
            "monthly_counts": [{"month": "2024-05", "count": 100}]
        }]
        mock_get_data.return_value = mock_data

        response = self.client.get('/monthly_analysis_stats', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)
        mock_get_data.assert_called_once_with(organization="test_org")

        # Σενάριο 2: Exception -> 500
        mock_get_data.side_effect = Exception("DB Error")
        response = self.client.get('/monthly_analysis_stats', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_data.side_effect = None

    # ------------------------------------------------------------------
    # /cluster_repos
    # ------------------------------------------------------------------
    @patch('api.routes.cluster_repositories_by_kus')
    def test_cluster_repos(self, mock_cluster):
        """
        Title: Testing repository clustering endpoint
        Description: Verifies the /cluster_repos POST endpoint performs K-Means clustering.
        Tests missing header, missing/invalid num_clusters, successful clustering,
        and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.post('/cluster_repos', json={"num_clusters": 3})
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Λείπει το num_clusters -> 400
        response = self.client.post('/cluster_repos', json={}, headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)

        # Σενάριο 2: num_clusters < 2 -> 400
        response = self.client.post('/cluster_repos', json={"num_clusters": 1}, headers=ORG_HEADER)
        self.assertEqual(response.status_code, 400)

        # Σενάριο 3: Επιτυχής ομαδοποίηση
        mock_data = [{"cluster": 0, "repos": ["repo1"]}, {"cluster": 1, "repos": ["repo2"]}]
        mock_cluster.return_value = mock_data

        response = self.client.post('/cluster_repos', json={"num_clusters": 3}, headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)
        mock_cluster.assert_called_once_with(3, organization="test_org")

        # Σενάριο 4: Exception -> 500
        mock_cluster.side_effect = Exception("Clustering error")
        response = self.client.post('/cluster_repos', json={"num_clusters": 3}, headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_cluster.side_effect = None

    # ------------------------------------------------------------------
    # /ku_risk
    # ------------------------------------------------------------------
    @patch('api.routes.calculate_risks')
    def test_get_ku_risk(self, mock_calculate_risks):
        """
        Title: Testing KU risk endpoint
        Description: Verifies the /ku_risk endpoint calculates and returns KU risk data.
        Tests missing header, successful retrieval, error in risk data, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/ku_risk')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_calculate_risks.return_value = {
            "ku_risk": {"KU_1": {"risk": 0.8}, "KU_2": {"risk": 0.5}},
            "employee_risk": {}
        }
        response = self.client.get('/ku_risk', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        ku_names = [item["ku_name"] for item in data]
        self.assertIn("KU_1", ku_names)
        mock_calculate_risks.assert_called_once_with(organization="test_org")

        # Σενάριο 2: Επιστροφή σφάλματος μέσα στα δεδομένα -> 500
        mock_calculate_risks.return_value = {"error": "Insufficient data"}
        response = self.client.get('/ku_risk', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)

        # Σενάριο 3: Exception -> 500
        mock_calculate_risks.side_effect = Exception("Risk error")
        response = self.client.get('/ku_risk', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_calculate_risks.side_effect = None

    # ------------------------------------------------------------------
    # /employee_risk
    # ------------------------------------------------------------------
    @patch('api.routes.calculate_risks')
    def test_get_employee_risk(self, mock_calculate_risks):
        """
        Title: Testing employee risk endpoint
        Description: Verifies the /employee_risk endpoint returns per-employee risk data.
        Tests missing header, successful retrieval, error in risk data, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/employee_risk')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_calculate_risks.return_value = {
            "ku_risk": {},
            "employee_risk": {
                "alice": {"absolute_risk": 0.9},
                "bob": {"absolute_risk": 0.4}
            }
        }
        response = self.client.get('/employee_risk', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        employee_names = [item["employee_name"] for item in data]
        self.assertIn("alice", employee_names)

        # Σενάριο 2: Επιστροφή σφάλματος μέσα στα δεδομένα -> 500
        mock_calculate_risks.return_value = {"error": "Insufficient data"}
        response = self.client.get('/employee_risk', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)

        # Σενάριο 3: Exception -> 500
        mock_calculate_risks.side_effect = Exception("Risk error")
        response = self.client.get('/employee_risk', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_calculate_risks.side_effect = None

    # ------------------------------------------------------------------
    # /organizationskills
    # ------------------------------------------------------------------
    @patch('api.routes.get_ku_skills_by_organization')
    def test_get_organization_skills(self, mock_get_skills):
        """
        Title: Testing organization skills endpoint
        Description: Verifies the /organizationskills endpoint returns KU skill summaries.
        Tests missing header, successful retrieval, None result, and exception handling.
        """
        # Σενάριο 0: Λείπει το header -> 400
        response = self.client.get('/organizationskills')
        self.assertEqual(response.status_code, 400)

        # Σενάριο 1: Επιτυχής ανάκτηση
        mock_data = [{"ku_id": "KU_1", "unique_files": 10, "unique_authors": 3}]
        mock_get_skills.return_value = mock_data

        response = self.client.get('/organizationskills', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, mock_data)
        mock_get_skills.assert_called_once_with("test_org")

        # Σενάριο 2: None return -> 500
        mock_get_skills.return_value = None
        response = self.client.get('/organizationskills', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)

        # Σενάριο 3: Exception -> 500
        mock_get_skills.side_effect = Exception("Skills error")
        response = self.client.get('/organizationskills', headers=ORG_HEADER)
        self.assertEqual(response.status_code, 500)
        mock_get_skills.side_effect = None


if __name__ == '__main__':
    unittest.main()