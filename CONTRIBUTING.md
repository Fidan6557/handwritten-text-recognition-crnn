# Contributing

Thank you for your interest in contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/handwritten-text-recognition-crnn.git
   cd handwritten-text-recognition-crnn
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/ app/
```

## Project Structure

See `README.md` for the full project layout and architecture overview.

## Pull Request Guidelines

- Open an issue before starting significant work
- Write tests for new functionality
- Update `CHANGELOG.md` under `[Unreleased]`
- Keep commits small and focused
