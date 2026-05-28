# Contributing

Thank you for your interest in contributing to this project.

## Getting Started

1. Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/handwritten-text-recognition-crnn.git
cd handwritten-text-recognition-crnn
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

Run tests before submitting:

```bash
make test
```

Lint your code:

```bash
make lint
```

## Pull Request Guidelines

- Keep changes focused — one feature or fix per PR.
- Write or update tests for any changed behaviour.
- Ensure `make test` and `make lint` pass before opening a PR.
- Write a clear PR description explaining what changed and why.

## Reporting Issues

Open an issue on GitHub and include:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Python and PyTorch versions
