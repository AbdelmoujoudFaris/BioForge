.PHONY: install lint format test test-integration cov serve dashboard docker-up docker-down docs clean

install:
	pip install -e ./engines/pharmaforge_core
	pip install -e ".[dev,scale,tracking,physics,frontend]"
	pre-commit install

lint:
	ruff check src tests engines/pharmaforge_core/pharmaforge_core
	ruff format --check src tests engines/pharmaforge_core/pharmaforge_core

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest tests/unit -q -m "not integration"

test-integration:
	pytest tests/integration -q -m integration

cov:
	pytest tests -q --cov=src/bioforge --cov-report=term-missing --cov-report=html

serve:
	python -m bioforge.cli serve $(SERVICE)

dashboard:
	streamlit run frontend/streamlit_app/app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

docs:
	sphinx-build -b html docs docs/_build/html

clean:
	find . -type d -name __pycache__ -not -path "*/node_modules/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml docs/_build
