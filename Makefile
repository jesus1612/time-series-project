# Time Series Project — monorepo orchestration
# Delegates to time-series-library (TSLib) and tslib-shiny-app (UI).

.PHONY: help install-lib install-lib-spark test-lib install-ui run-ui sampler check-java

help:
	@echo "Time Series Project (monorepo)"
	@echo "=============================="
	@echo ""
	@echo "Prerequisites: Python 3.9+, Java 17+ for PySpark / Spark paths in TSLib."
	@echo ""
	@echo "Targets:"
	@echo "  install-lib       - TSLib dev install (venv inside time-series-library/)"
	@echo "  install-lib-spark - TSLib + PySpark (requires Java 17+ on PATH)"
	@echo "  test-lib          - Run TSLib test suite"
	@echo "  install-ui        - Shiny app venv + pip (editable TSLib via ../time-series-library)"
	@echo "  run-ui            - Start Shiny on http://0.0.0.0:8000"
	@echo "  sampler           - Regenerate CSV datasets under sampler/datasets/"
	@echo "  check-java        - Verify Java 17+ (from UI Makefile)"
	@echo ""

install-lib:
	$(MAKE) -C time-series-library install-dev

install-lib-spark:
	$(MAKE) -C time-series-library install-spark

test-lib:
	$(MAKE) -C time-series-library test

install-ui:
	$(MAKE) -C tslib-shiny-app install

run-ui:
	$(MAKE) -C tslib-shiny-app run

sampler:
	@if [ -x time-series-library/venv/bin/python ]; then \
		time-series-library/venv/bin/python sampler/generate_datasets.py; \
	else \
		python3 sampler/generate_datasets.py; \
	fi

check-java:
	$(MAKE) -C tslib-shiny-app check-java
