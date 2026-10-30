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

class DatabaseOperationsTests(unittest.TestCase):
    def setUp(self):
        self.sample_repo_name = "test_repo"
        self.sample_repo_url = "https://github.com/test/repo"
        self.sample_commit = {
            "sha": "abc123",
            "author": "test_author",
            "file_content": "test content",
            "changed_lines": [1, 2, 3],
            "temp_filepath": "/tmp/test",
            "timestamp": datetime.datetime.now()
        }
        self.sample_analysis = {
            "filename": "test.py",
            "author": "test_author",
            "timestamp": datetime.datetime.now(),
            "sha": "abc123",
            "detected_kus": {"ku1": "value1"},
            "elapsed_time": 1.23
        }

    @patch('psycopg2.connect')
    def test_create_tables(self, mock_connect):
        """
        Test that create_tables executes the correct SQL commands
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (False,)  # Tables don't exist

        from api.data_db import create_tables
        create_tables()

        # Verify all table creation commands were executed
        self.assertEqual(mock_cursor.execute.call_count, 4)

    @patch('psycopg2.connect')
    def test_save_repo_to_db(self, mock_connect):
        """
        Test saving repository to database
        """
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        from api.data_db import save_repo_to_db
        save_repo_to_db(self.sample_repo_name, self.sample_repo_url, "Test repo", "Test comment")

        mock_conn.cursor().execute.assert_called_once()
        mock_conn.commit.assert_called_once()


class APITests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        with patch('core.ml_operations.loader.load_codebert_model') as mock_load_model:
            mock_load_model.return_value = MagicMock()
            from api.routes import init_routes
            init_routes(self.app)

        self.client = self.app.test_client()
        self.sample_repo_name = "test_repo"
        self.sample_repo_url = "https://github.com/test/repo"

    @patch('api.routes.save_commits_to_db')
    @patch('api.routes.extract_contributions')
    @patch('api.routes.pull_repo')
    @patch('api.routes.repo_exists')
    def test_list_commits(self, mock_repo_exists, mock_pull, mock_extract, mock_save):
        """
        Test listing commits from a repository
        """
        mock_repo_exists.return_value = True
        mock_pull.return_value = True
        mock_extract.return_value = [{"sha": "abc123", "author": "test"}]
        mock_save.return_value = None

        response = self.client.post('/commits', json={
            "repo_url": self.sample_repo_url,
            "limit": 10
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        mock_pull.assert_called_once()

    @patch('api.routes.save_repo_to_db')
    def test_create_repo(self, mock_save):
        """
        Test creating a repository entry
        """
        response = self.client.post('/repos', json={
            "repo_name": self.sample_repo_name,
            "url": self.sample_repo_url
        })
        self.assertEqual(response.status_code, 201)

    @patch('api.routes.getdetected_kus')
    def test_get_detected_kus(self, mock_get):
        """
        Test getting detected knowledge units
        """
        mock_get.return_value = [{"ku": "test"}]
        response = self.client.get('/detected_kus')
        self.assertEqual(response.status_code, 200)

    @patch('psycopg2.connect')
    def test_save_commits_to_db(self, mock_connect):
        """Test saving commits to database"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        from api.data_db import save_commits_to_db
        commits = [{
            "sha": "abc123",
            "author": "test",
            "file_content": "content",
            "changed_lines": [1, 2],
            "temp_filepath": "/tmp/test",
            "timestamp": datetime.datetime.now()
        }]

        save_commits_to_db(self.sample_repo_name, commits)
        mock_conn.cursor().execute.assert_called()
        mock_conn.commit.assert_called_once()

    @patch('psycopg2.connect')
    def test_update_analysis_status(self, mock_connect):
        """Test updating analysis status"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        from api.data_db import update_analysis_status
        update_analysis_status(
            self.sample_repo_name,
            "completed",
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
            progress=100
        )

        mock_conn.cursor().execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('api.routes.get_analysis_from_db')
    def test_analyzedb_endpoint(self, mock_get_analysis):
        """Test analysis results endpoint"""
        mock_get_analysis.return_value = [{
            "filename": "test.py",
            "detected_kus": ["KU1"],
            "author": "test",
            "timestamp": datetime.datetime.now().isoformat(),
            "sha": "abc123",
            "elapsed_time": 1.23
        }]

        response = self.client.get(f'/analyzedb?repo_name={self.sample_repo_name}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)

if __name__ == '__main__':
    unittest.main()
