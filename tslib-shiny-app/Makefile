.PHONY: install run clean clean-all format help setup-venv check-venv check-java setup-java-env install-java-macos install-java

# Default target
help:
	@echo "Available commands:"
	@echo "  install         - Install dependencies (creates venv if needed)"
	@echo "  run             - Run the Shiny application (requires venv, checks Java for ARIMA parallel)"
	@echo "  check-java      - Check if Java 17+ is installed and configured"
	@echo "  install-java    - Install Java 17+ (macOS only, uses Homebrew)"
	@echo "  setup-java-env  - Configure Java environment variables"
	@echo "  clean           - Clean cache and temporary files"
	@echo "  clean-all       - Clean everything including virtual environment"
	@echo "  format          - Format Python code with black"
	@echo "  setup-venv      - Create virtual environment manually"
	@echo "  help            - Show this help message"

# Check if virtual environment exists
check-venv:
	@if [ ! -d "venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv venv; \
		echo "Virtual environment created."; \
	fi

# Check Java version compatibility for Spark
check-java:
	@echo "Checking Java installation..."
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && \
		JAVA_VERSION=$$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1) && \
		if [ "$$JAVA_VERSION" -ge 17 ]; then \
			echo "✓ Java $$JAVA_VERSION found at /opt/homebrew/opt/openjdk@17"; \
		else \
			echo "✗ Java version $$JAVA_VERSION is not compatible (requires 17+)"; \
			exit 1; \
		fi; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17" && \
		JAVA_VERSION=$$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1) && \
		if [ "$$JAVA_VERSION" -ge 17 ]; then \
			echo "✓ Java $$JAVA_VERSION found at /usr/local/opt/openjdk@17"; \
		else \
			echo "✗ Java version $$JAVA_VERSION is not compatible (requires 17+)"; \
			exit 1; \
		fi; \
	elif command -v java >/dev/null 2>&1; then \
		JAVA_VERSION=$$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1) && \
		if [ "$$JAVA_VERSION" -ge 17 ]; then \
			echo "✓ Java $$JAVA_VERSION found in PATH"; \
		else \
			echo "✗ Java version $$JAVA_VERSION is not compatible (requires 17+)"; \
			echo "💡 Run 'make install-java' to install Java 17+"; \
			exit 1; \
		fi; \
	else \
		echo "✗ Java not found. PySpark requires Java 17+"; \
		echo "💡 Run 'make install-java' to install Java 17+"; \
		exit 1; \
	fi

# Setup Java environment variables
setup-java-env:
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17"; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17"; \
	fi

# Install Java 17+ (alias for install-java-macos)
install-java: install-java-macos

# Install Java 17+ on macOS
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
		elif [ -d "/usr/local/opt/openjdk@17" ]; then \
			echo "✓ Java 17 installed at /usr/local/opt/openjdk@17"; \
		else \
			echo "✗ Java 17 installation failed. Please install manually:"; \
			echo "  https://adoptium.net/temurin/releases/?version=17"; \
		fi; \
	else \
		echo "Homebrew not found. Please install Java 17+ manually:"; \
		echo "  https://adoptium.net/temurin/releases/?version=17"; \
	fi

# Create virtual environment manually
setup-venv:
	python3 -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate' to activate it."

# Install dependencies (auto-creates venv if needed)
install: check-venv
	@echo "Installing dependencies..."
	@if [ -f "venv/bin/activate" ]; then \
		. venv/bin/activate && pip install -r requirements.txt; \
	else \
		python3 -m pip install -r requirements.txt; \
	fi
	@echo "Dependencies installed successfully!"

# Run the application
run: check-venv
	@if [ ! -f "venv/bin/activate" ]; then \
		echo "❌ Error: El entorno virtual está corrupto."; \
		echo "💡 Solución: Ejecuta 'make clean && make install' para recrear el entorno."; \
		exit 1; \
	fi
	@echo "Verificando Java para modelos paralelos..."
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17"; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17"; \
	fi
	@echo "✅ Ejecutando aplicación con entorno virtual (hot reload activado)..."
	@if [ -d "/opt/homebrew/opt/openjdk@17" ]; then \
		export PATH="/opt/homebrew/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/opt/homebrew/opt/openjdk@17" && \
		. venv/bin/activate && shiny run --reload --host 0.0.0.0 --port 8000 app:app; \
	elif [ -d "/usr/local/opt/openjdk@17" ]; then \
		export PATH="/usr/local/opt/openjdk@17/bin:$$PATH" && \
		export JAVA_HOME="/usr/local/opt/openjdk@17" && \
		. venv/bin/activate && shiny run --reload --host 0.0.0.0 --port 8000 app:app; \
	else \
		. venv/bin/activate && shiny run --reload --host 0.0.0.0 --port 8000 app:app; \
	fi

# Clean cache and temporary files
clean:
	@echo "🧹 Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Limpieza completada."

# Clean everything including virtual environment
clean-all: clean
	@echo "🗑️ Eliminando entorno virtual..."
	rm -rf venv
	@echo "✅ Entorno virtual eliminado."

# Format code
format:
	black . --line-length 88
