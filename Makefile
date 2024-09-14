.PHONY: dev run-tests

dev:
	python -m black open_dubbing/ tests/
	python -m flake8 open_dubbing/ tests/

run-tests:
	python -m pytest


