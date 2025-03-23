# SPLAT_ONE Tests

This directory contains tests for the SPLAT_ONE application.

## Running Tests

To run the tests, make sure you have pytest installed:

```bash
pip install pytest pytest-cov
```

Then run the tests from the project root directory:

```bash
pytest tests/
```

For coverage reports:

```bash
pytest --cov=app tests/
```

## Test Organization

- `conftest.py`: Common fixtures and test configurations
- `test_*.py`: Test modules for different components
