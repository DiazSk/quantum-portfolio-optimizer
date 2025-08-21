Title: Fix `tests/conftest.py` mock fixture (pytest.mock.patch -> correct usage)

Description

The fixture `mock_yfinance` in `tests/conftest.py` uses `pytest.mock.patch`, which is not a pytest API and causes import/runtime errors in CI.

Steps to reproduce

1. Run `pytest -q` locally or in CI. Tests fail during conftest import or when the fixture is used.

Proposed fix

- Replace `pytest.mock.patch` usage with `from unittest.mock import patch` and apply `patch` as a context manager or decorator, or rewrite the fixture to use the `monkeypatch` pytest fixture.

Acceptance criteria

- Tests import cleanly and the fixture provides the same mocked responses as before.
- Unit & integration tests referencing `mock_yfinance` pass locally.
- CI pipeline passes the initial pytest stage.

Assignee: @james (please reassign as needed)
Labels: bug, tests, high-priority
Estimate: 1 SP
