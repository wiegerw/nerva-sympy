#!/usr/bin/env bash
# Run all tests in the tests folder.
# - Uses pytest if available (preferred for nicer output)
# - Falls back to Python's unittest discovery otherwise
# Any extra arguments are forwarded to the underlying test runner.

set -euo pipefail

# Resolve repository root (directory of this script)/..
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure src/ is on PYTHONPATH so tests import the local package
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Default test directory
TEST_DIR="${REPO_ROOT}/tests"

# Default pytest flags can be overridden with NERVA_PYTEST_FLAGS
PYTEST_FLAGS_DEFAULT="-v -ra --disable-warnings"
PYTEST_FLAGS="${NERVA_PYTEST_FLAGS:-$PYTEST_FLAGS_DEFAULT}"

if command -v pytest >/dev/null 2>&1; then
  echo "Detected pytest. Running tests with pytest..." >&2
  exec pytest ${PYTEST_FLAGS} "${TEST_DIR}" "$@"
else
  echo "pytest not found. Falling back to unittest discovery..." >&2
  # -b buffers stdout/stderr from tests (only shown on failure)
  exec python3 -m unittest discover -b -s "${TEST_DIR}" -p "test_*.py" -v "$@"
fi
