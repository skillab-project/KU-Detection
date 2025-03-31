# Testing the API Routes Component (`api/routes.py`)

## 1. Introduction

This document describes the testing strategy and procedures for the Flask routes component located in `api/routes.py`. The goal is to ensure the correct operation, robustness, and expected behavior of the API endpoints.

The primary methodology used is **Unit Testing** utilizing Python's built-in `unittest` and `unittest.mock` libraries.

## 2. Testing Philosophy

The tests for the routes (`api/test_routes.py`) are designed to be:

*   **Isolated:** Each test focuses on a specific endpoint or operational scenario. External dependencies, primarily calls to the database (`api/data_db.py`) and time-consuming operations (like Git repo cloning or loading ML models), are **bypassed (mocked)**.
*   **Fast:** Bypassing external dependencies allows for rapid test execution, facilitating integration into CI/CD pipelines and providing immediate feedback during development.
*   **Repeatable:** Tests do not depend on the state of the actual database or file system and must yield the same result every time they are executed.

**Important:** These unit tests **do not interact with the actual PostgreSQL database**. They use mocks to simulate database responses, ensuring the production database remains unaffected.

### 2.1 Testing Strategy

For the API routes component (`api/routes.py`), the following types of testing are applied:

1.  **Unit Testing:** (Primary Method) Testing each API endpoint in isolation, using mocks to isolate from external dependencies (database, Git, ML model). Focuses on the internal logic of the route, request/response handling, and calling the appropriate (mocked) external functions.
2.  **Functional Testing:** Verifying that the endpoints behave as expected based on specifications for specific inputs and usage scenarios.
3.  **Input Validation Testing:** Checking the behavior of endpoints when receiving valid, invalid, boundary, or incomplete input data (URL parameters, query parameters, JSON payloads).
4.  **Error Handling Testing:** Ensuring the application correctly handles errors (e.g., database failure, invalid repository URL, analysis errors) and returns appropriate HTTP status codes and error messages.
5.  **Asynchronous Operation Testing (Simulated):** Testing the initiation and response structure (e.g., Server-Sent Events for `/analyze`) of background operations by simulating their behavior with mocks.

*(Potential future types of testing, complementary to the existing unit tests, could include):*

6.  **Integration Testing (Database):**
    *   **Description:** Testing the interaction of API routes with a **real** (but separate, for testing) PostgreSQL database. This would verify that data is written and read correctly, database constraints work as expected, and more complex queries (if any) return correct results.
    *   **Goal:** Ensure the application functions correctly in conjunction with the actual database system.

7.  **Integration Testing (Git Operations):**
    *   **Description:** Testing the Git-interacting functions (`clone_repo`, `pull_repo`, `extract_contributions`, `get_history_repo`) by executing **real** Git commands against one or more *real* (perhaps temporary or sample) Git repositories.
    *   **Goal:** Verify the application can correctly handle Git repositories, including cases like clone failure (e.g., wrong URL, private repo), history changes, etc.

8.  **Integration Testing (ML Model Analysis):**
    *   **Description:** Testing the full analysis flow (`/analyze` endpoint and `codebert_sliding_window`) by **loading and using the actual CodeBERT model**. Could check if results are produced (without necessarily judging the *correctness* of KUs) for sample code and if errors during model loading or execution are handled gracefully.
    *   **Goal:** Ensure the analysis system works end-to-end with the real ML model. These tests might be slow.

9.  **End-to-End (E2E) Testing:**
    *   **Description:** Simulating complete user scenarios from the perspective of an external client calling the API. For example: a) Add repo, b) Fetch commits, c) Start analysis, d) Check status, e) Retrieve results.
    *   **Goal:** Verify that the different components (routes, database, git operations, analysis) cooperate correctly to complete a full workflow.

10. **Load / Performance Testing:**
    *   **Description:** Using tools (e.g., Locust, k6, JMeter) to simulate multiple concurrent users calling the API endpoints, especially demanding ones like `/analyze` and `/commits`. Measure response times, error rates, and resource usage (CPU, memory, database connections) under load.
    *   **Goal:** Identify bottlenecks, evaluate application scalability, and ensure acceptable performance under expected load.

11. **Concurrency Testing:**
    *   **Description:** Targeted testing of scenarios where multiple operations run simultaneously, especially around the background analysis (`/analyze`). Check for potential race conditions (e.g., updating analysis status in the database from multiple threads), deadlocks, or resource exhaustion (e.g., excessive number of threads).
    *   **Goal:** Ensure application stability and correctness when multiple analyses run concurrently.


## 3. Running the Unit Tests

To run the specific unit tests for the API routes:

1.  Ensure you have installed the necessary dependencies (usually via `pip install -r requirements.txt` in the project's root directory) and activated your virtual environment.
2.  Execute the following command from the project's **root** directory:

    ```bash
    python -m unittest api.test_routes
    ```

## 4. Existing Test Suite Overview

The current tests in `api/test_routes.py` cover the basic functionality of the following endpoints:

*   `/commits` (POST): Retrieve commits, handle clone/pull.
*   `/repos` (POST): Create a new repository entry.
*   `/detected_kus` (GET): Retrieve detected KUs.
*   `/repos/<repo_name>` (PUT): Edit an existing repository entry.
*   `/timestamps` (GET): Retrieve commit timestamps.
*   `/historytime` (GET): Retrieve commit history timeline.
*   `/delete_repo/<repo_name>` (DELETE): Delete a repository entry.
*   `/repos` (GET): List all repository entries.
*   `/analyze` (GET): Start the analysis process (streaming response).
*   `/analysis_status` (GET): Retrieve analysis status.
*   `/analyzedb` (GET): Retrieve stored analysis results for a repo.
*   `/analyzeall` (GET): Retrieve all stored analysis results.

A detailed description of each test case is provided in the [Test Case Catalog](#6-test-case-catalog) below.

## 5. Adding New Unit Tests

When adding new endpoints or modifying existing ones in `api/routes.py`, corresponding unit tests should also be added or updated in `api/test_routes.py`. Follow these steps:

1.  **Create Test Function:** Add a new method to the `FlaskAPITests` class that starts with `test_` (e.g., `test_new_endpoint_success`).
2.  **Use `@patch`:** Use the `@patch` decorator to mock *all* external dependencies called by your route (e.g., functions from `api.data_db`, `core.git_operations`, etc.). Target the function *where it is used* (usually imported into `api.routes`). The mocks are passed as arguments to your test function (in reverse order of the decorators).
    ```python
    @patch('api.routes.some_db_function')
    @patch('api.routes.another_external_call')
    def test_new_endpoint_success(self, mock_external_call, mock_db_function):
        # Mocks are passed in reverse order:
        # mock_external_call corresponds to @patch('api.routes.another_external_call')
        # mock_db_function corresponds to @patch('api.routes.some_db_function')
        # ...
    ```
3.  **Configure Mocks:** Inside your test function, set the return values (`return_value`) or side effects (`side_effect`) of the mock objects to simulate various scenarios (e.g., successful data retrieval, database error, empty results).
    ```python
    mock_db_function.return_value = [{'id': 1, 'data': 'sample'}]
    mock_external_call.side_effect = Exception("Network Error")
    ```
4.  **Call Endpoint:** Use `self.client` (the Flask test client, available in `FlaskAPITests`) to send the appropriate HTTP request to the endpoint you want to test.
    ```python
    response = self.client.get('/new_endpoint?param=value')
    response = self.client.post('/another_endpoint', json={'key': 'value'})
    ```
5.  **Assertions:** Use `unittest`'s `assert` methods (e.g., `self.assertEqual`, `self.assertTrue`, `self.assertIn`) to verify:
    *   The response status code: `self.assertEqual(response.status_code, 200)`
    *   The content type: `self.assertEqual(response.content_type, 'application/json')`
    *   The response data: `data = json.loads(response.data); self.assertEqual(data['message'], 'Success')`
    *   Whether the mocked functions were called correctly: `mock_db_function.assert_called_once_with(...)` or `mock_external_call.assert_called_once()`

## 6. Test Case Catalog

This catalog details the existing and proposed unit tests for the API routes.

### 6.1 Existing Test Cases (`api/test_routes.py`)

---

**ID:** `TC_LIST_COMMITS`
**Description:** Verifies the functionality of the `/commits` (POST) endpoint for retrieving a list of commits from a repository. Tests two scenarios: a) when the repo does not exist locally (requires `clone_repo`), and b) when the repo exists (requires `pull_repo`). Verifies the status code (200) and the correctness of the returned data. Also checks that the appropriate functions (clone/pull, extract, save) are called.
**Category:** Functional Testing, Integration Testing (Simulated - Git Operations & DB)
**Dependencies (Mocks):** `api.routes.save_commits_to_db`, `api.routes.extract_contributions`, `api.routes.pull_repo`, `api.routes.repo_exists`, `api.routes.clone_repo`

---

**ID:** `TC_CREATE_REPO`
**Description:** Verifies the functionality of the `/repos` (POST) endpoint for creating a new repository entry. Checks for successful creation (status 201, success message) and proper error handling during database saving (status 500, error message).
**Category:** Functional Testing, Error Handling
**Dependencies (Mocks):** `api.routes.save_repo_to_db`

---

**ID:** `TC_GET_DETECTED_KUS`
**Description:** Verifies the functionality of the `/detected_kus` (GET) endpoint for retrieving the list of detected knowledge units. Checks for successful retrieval (status 200, data check), the case where the database returns no data (status 500), and handling of database exceptions (status 500).
**Category:** Functional Testing, Error Handling
**Dependencies (Mocks):** `api.routes.getdetected_kus`

---

**ID:** `TC_EDIT_REPO`
**Description:** Verifies the functionality of the `/repos/<repo_name>` (PUT) endpoint for updating information of an existing repository. Checks for successful update (status 200, success message) and error handling during database saving (status 500).
**Category:** Functional Testing, Error Handling
**Dependencies (Mocks):** `api.routes.save_repo_to_db`

---

**ID:** `TC_GET_TIMESTAMPS`
**Description:** Verifies the functionality of the `/timestamps` (GET) endpoint for retrieving commit timestamps for a specific repository. Checks for successful retrieval (status 200, data check), the requirement of the `repo_name` parameter (status 400), and handling cases where the database returns no data (status 500).
**Category:** Functional Testing, Input Validation, Error Handling
**Dependencies (Mocks):** `api.routes.get_commits_timestamps_from_db`

---

**ID:** `TC_HISTORYTIME`
**Description:** Verifies the functionality of the `/historytime` (GET) endpoint for retrieving the timeline of commit dates for a repository. Checks for successful retrieval (status 200, structure and data check), the requirement of the `repo_url` parameter (status 400), and handling exceptions during history retrieval (status 500).
**Category:** Functional Testing, Input Validation, Error Handling
**Dependencies (Mocks):** `api.routes.get_history_repo`

---

**ID:** `TC_DELETE_REPO`
**Description:** Verifies the functionality of the `/delete_repo/<repo_name>` (DELETE) endpoint for deleting a repository entry. Checks for successful deletion (status 200, success message) and handling exceptions during database deletion (status 500).
**Category:** Functional Testing, Error Handling
**Dependencies (Mocks):** `api.routes.delete_repo_from_db`

---

**ID:** `TC_LIST_REPOS`
**Description:** Verifies the functionality of the `/repos` (GET) endpoint for retrieving the list of all repositories. Checks for successful retrieval (status 200, structure and data check) and handling of database exceptions (status 500).
**Category:** Functional Testing, Error Handling
**Dependencies (Mocks):** `api.routes.get_all_repos_from_db`

---

**ID:** `TC_ANALYZE_ENDPOINT`
**Description:** Verifies the functionality of the `/analyze` (GET) endpoint that starts code analysis and returns a streaming response. Checks for successful initiation (status 200, content-type 'text/event-stream'), the requirement of the `repo_url` parameter (status 400), and the initial fetching of commits and reading of files before starting the background task.
**Category:** Functional Testing, Input Validation, Asynchronous Operation Testing (Simulated Start)
**Dependencies (Mocks):** `api.routes.analyze_repository_background`, `api.routes.get_commits_from_db`, `api.routes.read_files_from_dict_list`

---

**ID:** `TC_ANALYSIS_STATUS_ENDPOINT`
**Description:** Verifies the functionality of the `/analysis_status` (GET) endpoint for retrieving the status of an analysis. Checks for successful retrieval (status 200, data check), the requirement of the `repo_name` parameter (status 400), and the case where no status is found for the repo (status 404).
**Category:** Functional Testing, Input Validation, State Verification
**Dependencies (Mocks):** `api.routes.get_analysis_status`

---

**ID:** `TC_ANALYZEDB_ENDPOINT`
**Description:** Verifies the functionality of the `/analyzedb` (GET) endpoint for retrieving stored analysis results for a repo. Checks for successful retrieval (status 200, data check), the requirement of the `repo_name` parameter (status 400), and handling cases where the database returns no data or raises an exception (status 500).
**Category:** Functional Testing, Input Validation, Error Handling
**Dependencies (Mocks):** `api.routes.get_analysis_from_db`

---

**ID:** `TC_ANALYZEALL_ENDPOINT`
**Description:** Verifies the functionality of the `/analyzeall` (GET) endpoint for retrieving analysis results for *all* repositories. Checks for successful retrieval (status 200, data check) and handling cases where the database returns no data or raises an exception (status 500).
**Category:** Functional Testing, Error Handling
**Dependencies (Mocks):** `api.routes.get_allanalysis_from_db`

---

### 6.2 Proposed New Test Cases

These are ideas for additional unit tests that could improve coverage.

---

**ID:** `TC_LIST_COMMITS_INVALID_URL`
**Description:** Test the behavior of `/commits` (POST) when an invalid or inaccessible `repo_url` is provided. It is expected to return an appropriate error code (e.g., 500 or 400) and a clear error message, as `clone_repo` or `pull_repo` will likely fail.
**Category:** Error Handling, Input Validation
**Difficulty:** Medium (Requires setting up the mock to simulate a failed git command)
**Mock Example:** `mock_clone.side_effect = Exception("Git clone failed: repository not found")`

---

**ID:** `TC_LIST_COMMITS_LIMIT_PARAM`
**Description:** Test the `limit` parameter of `/commits` (POST). Try with `limit=0`, `limit=1`, and `limit` greater than the number of commits returned by the `extract_contributions` mock. Verify that the number of returned commits is as expected.
**Category:** Functional Testing, Input Validation
**Difficulty:** Easy
**Example:** Add `limit` to the request JSON and check `len(json.loads(response.data))`.

---

**ID:** `TC_CREATE_REPO_MISSING_FIELDS`
**Description:** Test the behavior of `/repos` (POST) when mandatory fields (if any, e.g., `repo_name`) or optional fields are missing from the JSON payload. Verify that the endpoint either uses default values (as it seems to do now) or returns a 400 error if a field is strictly required.
**Category:** Input Validation, Robustness Testing
**Difficulty:** Easy
**Example:** `self.client.post('/repos', json={"repo_name": "only_name"})`

---

**ID:** `TC_EDIT_REPO_NOT_FOUND`
**Description:** Test the behavior of `/repos/<repo_name>` (PUT) when the `repo_name` provided in the URL does not correspond to an existing repository. The current implementation with `ON CONFLICT DO UPDATE` in `save_repo_to_db` would create a new repo. Consider if this is the desired behavior for an *edit* endpoint or if it should return 404. If it should return 404, the test must verify this (requires changes to the route logic or `save_repo_to_db`).
**Category:** Functional Testing, Edge Case Testing
**Difficulty:** Medium (May reveal the need for changes in application logic)

---

**ID:** `TC_ANALYZE_NO_COMMITS`
**Description:** Test the behavior of `/analyze` (GET) when `get_commits_from_db` returns an empty list for the given `repo_name`. The endpoint is expected to immediately return an error (e.g., 400 or 404) with an appropriate message, instead of proceeding to read files or start the background task.
**Category:** Error Handling, Edge Case Testing
**Difficulty:** Easy
**Mock Example:** `mock_get_commits.return_value = []`

---

**ID:** `TC_ANALYZE_STREAM_ERROR`
**Description:** Test the stream of `/analyze` (GET) when `analyze_repository_background` produces an error during its execution (e.g., after sending some progress updates). Verify that the last message in the stream contains the error information.
**Category:** Asynchronous Operation Testing, Error Handling
**Difficulty:** Medium (Requires setting up the mock generator to `yield` data and then `yield` an error or `raise Exception`)
**Mock Example:** `mock_analyze_background.return_value = iter([b'data: {"progress": 10}\n\n', b'data: {"error": "Analysis failed"}\n\n'])`

---

**ID:** `TC_ANALYZE_STREAM_CONTENT`
**Description:** Test the *content* of the messages sent via the stream from `/analyze` (GET). For a successful scenario, verify that the JSON messages contain the expected fields (`progress`, `file_data` or `message`, `repoUrl`) and that the progress increases logically, eventually reaching 100%.
**Category:** Functional Testing, Asynchronous Operation Testing, Data Validation
**Difficulty:** Medium
**Example:** Analyze the chunks returned from `response.stream`.

---

**ID:** `TC_INVALID_REPO_NAME_FORMAT`
**Description:** Test endpoints accepting `repo_name` as part of the URL (e.g., `/repos/<repo_name>`, `/delete_repo/<repo_name>`, `/analysis_status?repo_name=...`, `/analyzedb?repo_name=...`) with invalid names (e.g., containing `/`, `..`, special characters). Depending on how `repo_name` is used downstream (hopefully not directly in paths), they should either be rejected (400/404) or handled safely.
**Category:** Input Validation, Security Testing (Basic)
**Difficulty:** Easy/Medium

---