# TSLib - Time Series Library Makefile
# Comprehensive build and development automation

.PHONY: help venv prepare-test install install-spark install-dev test test-coverage test-spark benchmark clean format lint docs examples check-version setup-spark-parallel demo-parallel

# Default target
help:
	@echo "TSLib - Time Series Library"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  install              - Install basic dependencies and library"
	@echo "  install-spark        - Install with PySpark support"
	@echo "  install-dev          - Install with development dependencies"
	@echo "  setup-spark-parallel - Setup everything for parallel ARIMA workflow"
	@echo "  demo-parallel        - Run parallel ARIMA workflow demo"
	@echo "  venv           - Create ./venv if missing (python3 -m venv)"
	@echo "  prepare-test   - venv + pip deps (core, PySpark, pytest/statsmodels dev)"
	@echo "  test           - prepare-test + full pytest (tests/benchmark, Java auto)"
	@echo "  test-coverage  - prepare-test + pytest with coverage"
	@echo "  test-spark     - Run Spark-specific tests"
	@echo "  benchmark      - Run performance benchmarks"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with flake8"
	@echo "  docs           - Generate documentation"
	@echo "  examples       - Run all example scripts (AR, MA, ARMA, ARIMA)"
	@echo "  clean          - Clean build artifacts"
	@echo "  check-version  - Check Python version compatibility"
	@echo ""

# Check Python version compatibility
check-version:
	@echo "Checking Python version compatibility..."
	@python3 -c "import sys; version = sys.version_info; assert version >= (3, 9), f'Python 3.9+ required, found {version.major}.{version.minor}'; print(f'✓ Python {version.major}.{version.minor}.{version.micro} is compatible')"

# Check Java version compatibility for Spark
check-java:
	@echo "Checking Java version compatibility for Spark..."
	@if command -v java >/dev/null 2>&1; then \
		JAVA_VERSION=$$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1); \
		if [ "$$JAVA_VERSION" -ge 17 ]; then \
			echo "✓ Java $$JAVA_VERSION is compatible with PySpark 4.0.1"; \
		else \
			echo "✗ Java $$JAVA_VERSION is not compatible. PySpark 4.0.1 requires Java 17+"; \
			echo "  Install Java 17+ using: make install-java-macos"; \
			exit 1; \
		fi; \
	else \
		echo "✗ Java not found. PySpark 4.0.1 requires Java 17+"; \
		echo "  Install Java 17+ using: make install-java-macos"; \
		exit 1; \
	fi

# Virtual environment (required for install-* and test)
venv: check-version
	@if [ ! -x venv/bin/python ]; then \
		echo "Creando entorno virtual en ./venv ..."; \
		python3 -m venv venv; \
	fi

# Everything needed to run the full test suite (no Java install: use make install-java-macos if Spark tests skip)
prepare-test: venv
	@echo "Preparando entorno de pruebas (pip: requirements + PySpark + herramientas dev)..."
	@venv/bin/pip install -q -U pip setuptools wheel
	@venv/bin/pip install -q -r requirements.txt
	@venv/bin/pip install -q -r requirements-spark.txt
	@venv/bin/pip install -q -e ".[dev]"
	@echo "✓ Dependencias de test listas (pytest, pyspark, statsmodels opcional para benchmarks)."

# Installation targets
install: venv check-version
	@echo "Installing TSLib with basic dependencies..."
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -e .

install-spark: venv check-version check-java
	@echo "Installing TSLib with PySpark support..."
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements-spark.txt
	venv/bin/pip install -e .

install-dev: venv check-version
	@echo "Installing TSLib with development dependencies..."
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements-spark.txt
	venv/bin/pip install -e ".[dev]"

# Testing targets (pytest runs via run_pytest_with_java.sh so Spark benchmarks use the same JDK discovery as lib_java_spark.sh)
test: prepare-test
	@echo "Ejecutando toda la suite (unitarios + integración + tests/benchmark)..."
	bash scripts/run_pytest_with_java.sh tests/ -v

test-coverage: prepare-test
	@echo "Ejecutando tests con cobertura..."
	bash scripts/run_pytest_with_java.sh tests/ --cov=tslib --cov-report=html --cov-report=term-missing -v

test-spark: prepare-test check-java
	@echo "Running Spark-specific tests..."
	bash scripts/run_pytest_with_java.sh tests/test_spark_comparison.py tests/test_performance_benchmark.py -v

benchmark: prepare-test check-java
	@echo "Running benchmark suite (linear vs Spark vs statsmodels per model family)..."
	bash scripts/run_pytest_with_java.sh tests/benchmark/ -m benchmark -s --tb=short -v
	@echo "--- Legacy Spark vs normal ARIMA (optional) ---"
	bash scripts/run_pytest_with_java.sh tests/test_performance_benchmark.py::TestPerformanceBenchmark::test_comprehensive_benchmark -v -s

# Development targets
format: venv check-version
	@echo "Formatting code with black..."
	venv/bin/black tslib/ tests/ examples/ --line-length 88

lint: venv check-version
	@echo "Linting code with flake8..."
	venv/bin/flake8 tslib/ tests/ examples/ --max-line-length 88 --ignore E203,W503

# Documentation targets
docs: check-version
	@echo "Generating documentation..."
	@if [ -d "docs" ]; then \
		echo "Documentation generation not yet implemented"; \
	else \
		echo "Creating basic documentation structure..."; \
		mkdir -p docs; \
		echo "# TSLib Documentation" > docs/README.md; \
		echo "Documentation will be generated here." >> docs/README.md; \
	fi

# Example targets
examples: venv check-version
	@echo "Running example scripts..."
	@echo "Running AR model example..."
	venv/bin/python examples/ar_example.py
	@echo ""
	@echo "Running MA model example..."
	venv/bin/python examples/ma_example.py
	@echo ""
	@echo "Running ARMA model example..."
	venv/bin/python examples/arma_example.py
	@echo ""
	@echo "Running basic ARIMA example..."
	venv/bin/python examples/basic_arima.py
	@echo ""
	@echo "Running Spark parallel ARIMA example..."
	@if venv/bin/python -c "import pyspark" 2>/dev/null; then \
		venv/bin/python examples/spark_parallel_arima.py; \
	else \
		echo "PySpark not available, skipping Spark example"; \
	fi
	@echo ""
	@echo "Running internal parallelization demo..."
	venv/bin/python examples/parallel_internal_demo.py

# Cleanup targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

quick-test: prepare-test
	@echo "Running quick test suite..."
	bash scripts/run_pytest_with_java.sh tests/test_arima.py -v

full-test: test test-spark benchmark
	@echo "Full test suite completed!"

# Performance analysis
profile: venv check-version
	@echo "Running performance profiling..."
	venv/bin/python -m cProfile -o profile_output.prof examples/basic_arima.py
	@echo "Profile saved to profile_output.prof"

# Version and compatibility checks
check-deps: venv check-version
	@echo "Checking dependency versions..."
	venv/bin/pip list | grep -E "(numpy|scipy|pandas|matplotlib|pyspark)"

# Docker support (if needed)
docker-build:
	@echo "Building Docker image..."
	@if [ -f "Dockerfile" ]; then \
		docker build -t tslib .; \
	else \
		echo "Dockerfile not found"; \
	fi

# CI/CD helpers
ci-test: prepare-test test-coverage lint
	@echo "CI test pipeline completed!"

# Release helpers
version-check:
	@echo "Current version information:"
	@venv/bin/python -c "import tslib; print(f'TSLib version: {getattr(tslib, \"__version__\", \"unknown\")}')" 2>/dev/null || echo "TSLib not installed"

# Help for specific targets
help-install:
	@echo "Installation Options:"
	@echo "  make install        - Basic installation (core dependencies only)"
	@echo "  make install-spark  - With PySpark for distributed computing"
	@echo "  make install-dev    - With development tools (pytest, black, etc.)"

help-test:
	@echo "Testing Options:"
	@echo "  make prepare-test   - Install venv + deps needed for tests (run once or after pull)"
	@echo "  make test           - prepare-test + full pytest (Spark si hay JDK 17+)"
	@echo "  make test-coverage  - prepare-test + pytest with coverage report"
	@echo "  make test-spark     - Run Spark-specific tests only (requires Java 17+)"
	@echo "  make benchmark      - Run performance benchmarks (requires Java 17+)"
	@echo "  make quick-test     - Run basic tests only"

# Java installation commands
install-java-macos:
	@echo "Installing Java 17+ on macOS..."
	@if command -v brew >/dev/null 2>&1; then \
		if [ -d "/opt/homebrew/opt/openjdk@17" ] || [ -d "/usr/local/opt/openjdk@17" ]; then \
			echo "✓ Java 17 already installed"; \
		else \
			brew install openjdk@17 || true; \
		fi; \
		if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
			echo "✓ Java 17 installed at /opt/homebrew/opt/openjdk@17"; \
			echo "To use in this session, run:"; \
			echo '  export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH"'; \
			echo '  export JAVA_HOME="/opt/homebrew/opt/openjdk@17"'; \
		elif [ -d "/usr/local/opt/openjdk@17" ]; then \
			echo "✓ Java 17 installed at /usr/local/opt/openjdk@17"; \
			echo "To use in this session, run:"; \
			echo '  export PATH="/usr/local/opt/openjdk@17/bin:$$PATH"'; \
			echo '  export JAVA_HOME="/usr/local/opt/openjdk@17"'; \
		else \
			echo "✗ Java 17 installation failed. Please install manually:"; \
			echo "  https://adoptium.net/temurin/releases/?version=17"; \
		fi; \
	else \
		echo "Homebrew not found. Please install Java 17+ manually:"; \
		echo "  https://adoptium.net/temurin/releases/?version=17"; \
	fi

# Setup function to find and configure Java
setup-java-env:
	@echo "Configuring Java environment..."
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && \
		echo "✓ Using Java at /opt/homebrew/opt/openjdk@17" && \
		java -version 2>&1 | head -n 1; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17" && \
		echo "✓ Using Java at /usr/local/opt/openjdk@17" && \
		java -version 2>&1 | head -n 1; \
	else \
		echo "✗ Java 17 not found. Installing..."; \
		$(MAKE) install-java-macos; \
	fi

install-java-linux:
	@echo "Installing Java 17+ on Linux..."
	@echo "Ubuntu/Debian: sudo apt update && sudo apt install openjdk-17-jdk"
	@echo "CentOS/RHEL: sudo yum install java-17-openjdk-devel"
	@echo "Or download from: https://adoptium.net/temurin/releases/?version=17"

install-java-windows:
	@echo "Installing Java 17+ on Windows..."
	@echo "1. Download from: https://adoptium.net/temurin/releases/?version=17"
	@echo "2. Run the installer"
	@echo "3. Set JAVA_HOME environment variable to installation directory"

help-dev:
	@echo "Development Options:"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code with flake8"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make dev-setup      - Complete development setup"

# Setup everything for parallel Spark workflow
setup-spark-parallel: venv check-version
	@echo "=========================================="
	@echo "Setting up Parallel ARIMA Workflow"
	@echo "=========================================="
	@echo ""
	@echo "Step 1: Checking/Installing Java 17+..."
	@$(MAKE) install-java-macos
	@echo ""
	@echo "Step 2: Verifying Java setup..."
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && \
		java -version 2>&1 | head -n 1 && \
		echo "✓ Java configured correctly"; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17" && \
		java -version 2>&1 | head -n 1 && \
		echo "✓ Java configured correctly"; \
	else \
		echo "✗ Java 17 not found. Please install manually."; \
		exit 1; \
	fi
	@echo ""
	@echo "Step 3: Installing dependencies..."
	@venv/bin/pip install -q -r requirements.txt
	@venv/bin/pip install -q -r requirements-spark.txt
	@venv/bin/pip install -q -e .
	@echo "✓ Dependencies installed"
	@echo ""
	@echo "Step 4: Verifying PySpark installation..."
	@if venv/bin/python -c "import pyspark; print(f'✓ PySpark {pyspark.__version__} installed')" 2>/dev/null; then \
		echo ""; \
		echo "=========================================="; \
		echo "Setup Complete!"; \
		echo "=========================================="; \
		echo ""; \
		echo "To run the parallel workflow demo, use:"; \
		echo "  make demo-parallel"; \
	else \
		echo "✗ PySpark not installed correctly"; \
		exit 1; \
	fi

# Run the parallel ARIMA workflow demo
demo-parallel: venv check-version
	@echo "=========================================="
	@echo "Running Parallel ARIMA Workflow Demo"
	@echo "=========================================="
	@echo ""
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && \
		echo "Using Java 17 from /opt/homebrew/opt/openjdk@17" && \
		venv/bin/python examples/spark_parallel_arima_workflow_demo.py; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17" && \
		echo "Using Java 17 from /usr/local/opt/openjdk@17" && \
		venv/bin/python examples/spark_parallel_arima_workflow_demo.py; \
	else \
		echo "✗ Java 17 not found. Please run 'make setup-spark-parallel' first."; \
		exit 1; \
	fi

# Update examples target to include parallel workflow demo
examples-parallel: venv check-version
	@echo "Running parallel ARIMA workflow example..."
	@if [ -d "/opt/homebrew/opt/openjdk@17" ] || [ -d "/usr/local/opt/openjdk@17" ]; then \
		if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
			export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
			export JAVA_HOME="/opt/homebrew/opt/openjdk@17"; \
		else \
			export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
			export JAVA_HOME="/usr/local/opt/openjdk@17"; \
		fi && \
		venv/bin/python examples/spark_parallel_arima_workflow_demo.py; \
	else \
		echo "Java 17 not found. Run 'make setup-spark-parallel' first."; \
	fi
