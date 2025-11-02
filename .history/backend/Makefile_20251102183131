SHELL := /bin/bash

PY := python3
VENV := venv
VENV_PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PORT ?= 10000

.PHONY: venv reenv install run health clean check

venv:
	@if [ ! -x "$(VENV_PY)" ]; then \
		$(PY) -m venv $(VENV); \
	fi

reenv:
	rm -rf $(VENV)
	$(PY) -m venv $(VENV)

install: venv
	$(VENV_PY) -m pip install --upgrade pip setuptools wheel
	$(VENV_PY) -m pip install -r requirements.txt

run:
	PORT=$(PORT) $(VENV_PY) app.py

health:
	@echo "GET http://localhost:$(PORT)/api/health" && \
	curl -sS http://localhost:$(PORT)/api/health | sed 's/{/{\n  /;s/,/,\n  /g;s/}/\n}/'

check: venv
	$(VENV_PY) scripts/check_imports.py

clean:
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__ .pytest_cache
	rm -rf .cache/huggingface