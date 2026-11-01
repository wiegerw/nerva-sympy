# One-Layer Forward/Backward Test Cases

This directory contains generated test cases used by `tests/test_one_layer.py`.

## Generate test cases
- Default location: The generator writes here by default.

```bash
python scripts/generate_one_layer_tests.py
```

- Custom options (optional):

```bash
python scripts/generate_one_layer_tests.py --Ns 2,4 --Ds 3,5 --Ks 2,3
```

- Explicit output directory (optional):

```bash
python scripts/generate_one_layer_tests.py --out-dir tests/one_layer_cases
```

## Run the tests
- Quiet mode (default):

```bash
pytest -q tests/test_one_layer.py
```

- Debug output (recommended to run via python so the environment variable is honored):

```bash
ONE_LAYER_DEBUG=1 python tests/test_one_layer.py
ONE_LAYER_DEBUG=2 python tests/test_one_layer.py
```

- On Windows PowerShell:

```powershell
$Env:ONE_LAYER_DEBUG=1; python tests/test_one_layer.py
$Env:ONE_LAYER_DEBUG=2; python tests/test_one_layer.py
```

- Note on pytest and environment variables:
  Some environments may ignore inline environment variables when invoking pytest directly. If you do not see debug output with pytest, use the python invocation shown above.

## Notes
- The generated cases are committed to the repository, so the test can be run without generating them first.
  Regenerating them requires `nerva-torch`. With the default options the output is reproducible (seed 12345).
- The generator also copies the cases to the `tests/one_layer_cases` directories of sibling checkouts of the other
  backends, whose `test_one_layer.py` compare their implementation with the same cases.
- If no `manifest.json` is present in this directory, the test will be skipped.
- The debug levels are:
  - 0 or unset: no extra output (default)
  - 1: per-case info and per-assert summary (names, shapes, max|diff|)
  - 2: includes tensor contents for small tensors (<= 200 elements)
