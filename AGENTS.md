# Darwin Gödel Machine (DGM)

## Project Overview

Darwin Gödel Machine (DGM) is a self-improving system that iteratively modifies its own code and validates changes using SWE-bench and Polyglot benchmarks.

## Technology Stack

- **Language**: Python 3.10+
- **Testing**: pytest
- **Containerization**: Docker
- **LLM Providers**: Anthropic Claude, OpenAI (GPT-4o, o1, o3-mini), DeepSeek

## Build/Test Commands

### Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_bash_tool.py

# Run specific test
pytest tests/test_bash_tool.py::TestBashTool::test_simple_command

# Run with verbose output (default in pytest.ini)
pytest -v
```

### Running the System
```bash
# Main DGM loop
python DGM_outer.py

# With custom arguments
python DGM_outer.py --max_generation 80 --selfimprove_size 2

# Single self-improvement step
python self_improve_step.py --parent_commit initial --entry django__django-10999 --output_dir ./output

# Evaluation only
python test_swebench.py --test_med --num_samples 50
```

## Code Style Guidelines

### Imports
```python
# Standard library imports first
import argparse
import json
import os
from typing import List, Dict, Optional

# Third-party imports
import anthropic
import openai
import pytest

# Local imports (use relative imports within packages)
from tools.bash import tool_function
from utils.common_utils import load_json_file
```

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `get_response_from_llm`, `max_tokens`)
- **Classes**: `PascalCase` (e.g., `AgenticSystem`, `BashSession`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_OUTPUT_TOKENS`, `CLAUDE_MODEL`)
- **Private**: `_leading_underscore` for internal use

### Type Hints
Use type hints for function signatures when non-obvious:
```python
def create_client(model: str) -> tuple:
def load_json_file(file_path: str) -> dict:
```

### Error Handling
- Use try/except with specific exceptions
- Always log errors with context using thread-local loggers
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### Documentation
- Docstrings for public functions and classes
- Use triple quotes even for single-line docstrings
```python
def tool_info():
    """Return tool metadata as JSON."""
    return {...}
```

### Tool Development (tools/)
Each tool module must implement:
```python
def tool_info():
    return {
        "name": "tool_name",
        "description": "What the tool does",
        "input_schema": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "..."}
            },
            "required": ["param"]
        }
    }

def tool_function(param):
    """Implementation here."""
    pass
```

### Thread Safety
Use thread-local loggers for parallel execution:
```python
import threading
thread_local = threading.local()

def get_thread_logger():
    return getattr(thread_local, 'logger', None)
```

### Testing Patterns
- Test files: `test_*.py` or `*_test.py`
- Test classes: `class TestFeature:`
- Test methods: `def test_specific_behavior(self):`
- Use fixtures in `conftest.py` for shared setup

## Project Structure

```
dgm/
├── DGM_outer.py              # Main evolutionary loop
├── coding_agent.py           # SWE-bench agent
├── coding_agent_polyglot.py  # Polyglot agent
├── llm.py                    # LLM client creation
├── llm_withtools.py          # Tool-use functionality
├── self_improve_step.py      # Single self-improve step
├── tools/                    # Agent tools (bash, edit)
├── prompts/                  # LLM prompts
├── utils/                    # Utilities (docker, git, eval)
├── tests/                    # Unit tests
├── swe_bench/               # SWE-bench framework
└── polyglot/                # Polyglot benchmark
```

## Security Notes

**WARNING**: This repo executes untrusted model-generated code in Docker containers.
- Docker provides isolation but not absolute safety
- Review all code changes before production use
- Model-generated code may behave destructively

## Environment Variables

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export AWS_REGION_NAME='...'          # Optional (Bedrock)
export AWS_ACCESS_KEY_ID='...'        # Optional (Bedrock)
export AWS_SECRET_ACCESS_KEY='...'    # Optional (Bedrock)
```

## Key Configuration Files

- `pytest.ini`: Test configuration (verbose by default, discovers in tests/)
- `requirements.txt`: Core dependencies
- `requirements_dev.txt`: Analysis/plotting dependencies
- `Dockerfile`: Sandbox container definition

## Agent Development

1. Main agent classes: `AgenticSystem` (coding_agent.py, coding_agent_polyglot.py)
2. Entry point: `forward()` method
3. Use `llm_withtools.chat_with_agent()` for LLM interactions
4. Tools auto-discovered via `tools/__init__.py`

## LLM Models Used

- **Claude**: `bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0` (coding)
- **OpenAI**: `o3-mini-2025-01-31` (self-improvement)
- **Diagnosis**: `o1-2024-12-17` (problem diagnosis)
