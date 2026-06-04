.PHONY: help install test eval gate clean

help:
	@echo "make install  - install pinned dependencies"
	@echo "make test     - run safety-gate + adversarial suites (no API calls)"
	@echo "make eval     - run full pipeline over eval set -> predictions.jsonl + metrics.json"
	@echo "make gate      - print deterministic gate validation on the training set"

install:
	pip install -r requirements.txt

test:
	pytest -q

eval:
	python3 eval.py

gate:
	python3 -m src.safety_gate

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
