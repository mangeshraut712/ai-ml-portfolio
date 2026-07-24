.PHONY: help install test test-mlfs demo verify-vad-pass lab-llm-eval lab-llm-eval-test verify-all fetch-vad-samples run-vad-ui run-llm-eval-ui

VENV ?= .venv
PY := $(VENV)/bin/python

help:
	@echo "AI/ML Portfolio monorepo"
	@echo "  make install           - venv + all lab deps"
	@echo "  make test-mlfs         - NumPy ML from-scratch tests"
	@echo "  make demo              - MLFS demos (bias-variance, ROC, …)"
	@echo "  make verify-vad-pass   - VAD challenge FULL_PASS gate"
	@echo "  make lab-llm-eval      - offline RAG / LLM evaluation lab"
	@echo "  make lab-llm-eval-test - LLM eval smoke tests"
	@echo "  make verify-all        - mlfs + vad + llm-eval (CI-equivalent)"
	@echo "  make fetch-vad-samples - re-download 50× 16 kHz WAVs"
	@echo "  make run-vad-ui        - Streamlit VAD UI"
	@echo "  make run-llm-eval-ui   - Streamlit LLM eval report UI"

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

test-mlfs:
	$(PY) -m pytest tests -q

test: test-mlfs lab-llm-eval-test
	$(PY) -m pytest labs/vad/test_vad_pipeline.py -q

demo:
	PYTHONPATH=. $(PY) -m mlfs.demos.run_all

fetch-vad-samples:
	$(PY) labs/vad/fetch_samples.py

verify-vad-pass:
	@test -f sample_data/sample_audio/sample_001.wav || $(MAKE) fetch-vad-samples
	$(PY) labs/vad/challenge_pass.py

lab-llm-eval:
	$(PY) labs/llm-eval/src/run_lab.py

lab-llm-eval-test:
	$(PY) -m pytest labs/llm-eval/tests -q

verify-all: test-mlfs lab-llm-eval-test verify-vad-pass lab-llm-eval
	@echo "[+] verify-all OK (mlfs + VAD FULL_PASS + LLM eval)"

run-vad-ui:
	@test -f sample_data/sample_audio/sample_001.wav || $(MAKE) fetch-vad-samples
	$(PY) -m streamlit run labs/vad/streamlit_app.py

run-llm-eval-ui:
	@test -f labs/llm-eval/reports/latest_report.json || $(MAKE) lab-llm-eval
	$(PY) -m streamlit run labs/llm-eval/ui/streamlit_app.py
