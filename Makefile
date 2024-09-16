.PHONY: dev run-tests

dev:
	python -m black open_dubbing/ tests/ e2e-tests/
	python -m flake8 open_dubbing/ tests/ e2e-tests/

run-tests:
	python -m pytest tests/

run-e2e-tests:
	CT2_USE_MKL="False" CT2_FORCE_CPU_ISA='GENERIC' KMP_DUPLICATE_LIB_OK="TRUE" TOKENIZERS_PARALLELISM="false" python -m pytest e2e-tests/

