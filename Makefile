.PHONY: install run test eval eval-mock lint clean

# Prefer the project virtual environment on Windows; allow callers to override
# this with `make PYTHON=python` (or another interpreter).
ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= python3
endif

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

test:
	$(PYTHON) -m pytest -v

eval:
	$(PYTHON) -X utf8 eval.py --delay 13 --save-report eval_report.json

eval-mock:
	$(PYTHON) -X utf8 eval.py --mock

lint:
	$(PYTHON) -m ruff check app tests eval.py --select E4,E7,E9,F

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; paths = ['.pytest_cache', '__pycache__', 'app/__pycache__', 'app/providers/__pycache__', 'tests/__pycache__', 'eval_report.json']; [shutil.rmtree(p) if Path(p).is_dir() else Path(p).unlink(missing_ok=True) for p in paths]"
