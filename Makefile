PY ?= python
DEMO_ENV = SCTC_DATA_DIR=./demo_data

.PHONY: help install demo vaso septic figures test all

help:
	@echo "make install   install dependencies"
	@echo "make demo      generate synthetic data and run both tasks on it"
	@echo "make vaso      run Task 1 (vasopressor escalation)"
	@echo "make septic    run Task 2 (septic-shock onset)"
	@echo "make figures   regenerate the result figures"
	@echo "make test      run the unit tests"
	@echo "make all       run both tasks, the figures, and the tests"

install:
	$(PY) -m pip install -r requirements.txt

demo:
	$(PY) make_demo_data.py
	$(DEMO_ENV) $(PY) run_vaso.py
	$(DEMO_ENV) $(PY) run_septic.py
	$(DEMO_ENV) $(PY) reproduce_figures.py

vaso:
	$(PY) run_vaso.py

septic:
	$(PY) run_septic.py

figures:
	$(PY) reproduce_figures.py

test:
	$(PY) tests/test_sctc.py

all: vaso septic figures test
